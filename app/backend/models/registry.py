"""Build the four models from app/config.yaml. The app only talks to the shared interfaces."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import AppConfig, ModelSpec
from .detectors import CenterDetector, HaarDetector, YuNetDetector, ensure_file
from .dummy import DummyEmotion, DummyLiveness, PixelEmbedder

DEFAULT_RECOGNITION_THRESHOLD = 0.5


@dataclass
class ModelBundle:
    detector: Any
    liveness: Any
    embedder: Any
    emotion: Any
    info: dict[str, dict[str, Any]]
    checkpoint_recognition_threshold: float
    # torch module không nên chạy song song từ nhiều thread
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedder.embedding_dim)

    @property
    def model_version(self) -> str:
        return str(self.embedder.model_version)

    @property
    def embed_crop(self) -> tuple[int, float]:
        return embed_crop_params(self.embedder)


def embed_crop_params(embedder: Any) -> tuple[int, float]:
    if hasattr(embedder, "preprocess_config"):
        cfg = embedder.preprocess_config
        return int(cfg.size), float(cfg.crop_scale)
    if hasattr(embedder, "preprocessing"):
        cfg = embedder.preprocessing
        return int(cfg.get("size", 112)), float(cfg.get("scale", 1.0))
    return int(getattr(embedder, "crop_size", 112)), float(
        getattr(embedder, "crop_scale", 1.0)
    )


def _require(config: AppConfig, spec: ModelSpec) -> Path:
    if not spec.path:
        raise ValueError(f"model type {spec.type!r} needs a path")
    path = config.path(spec.path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def build_detector(config: AppConfig, spec: ModelSpec):
    if spec.type == "yunet":
        path = config.path(spec.path or "app/weights/face_detection_yunet_2023mar.onnx")
        extra = spec.model_extra or {}
        ensure_file(path, extra.get("url"), extra.get("sha256"))
        return YuNetDetector(path, float(extra.get("score_threshold", 0.8)))
    if spec.type == "haar":
        return HaarDetector()
    if spec.type == "center":
        return CenterDetector()
    raise ValueError(f"unknown detector type {spec.type!r}")


def build_liveness(config: AppConfig, spec: ModelSpec):
    if spec.type == "dummy":
        return DummyLiveness()
    if spec.type == "liveness":
        from src.liveness.predictor import LivenessPredictor

        return LivenessPredictor(_require(config, spec), device=config.device)
    raise ValueError(f"unknown liveness type {spec.type!r}")


def build_embedder(config: AppConfig, spec: ModelSpec) -> tuple[Any, float]:
    """Trả về (embedder, threshold lưu trong checkpoint)."""
    if spec.type == "dummy":
        dim = int((spec.model_extra or {}).get("dim", 128))
        return PixelEmbedder(dim), DEFAULT_RECOGNITION_THRESHOLD
    path = _require(config, spec)
    if spec.type == "recognition":
        from src.recognition.classification.predictor import RecognitionEmbedder

        embedder = RecognitionEmbedder(path, device=config.device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metrics = payload.get("val_metrics") or {}
        threshold = metrics.get("val_score_threshold", DEFAULT_RECOGNITION_THRESHOLD)
        return embedder, float(threshold)
    if spec.type == "metric":
        from src.recognition.identification.predictor import MetricEmbedder

        embedder = MetricEmbedder(path, device=config.device)
        metrics = embedder.payload.get("val_metrics") or {}
        legacy = embedder.payload.get("dev_metrics") or {}
        threshold = metrics.get(
            "dev_threshold", legacy.get("threshold", DEFAULT_RECOGNITION_THRESHOLD)
        )
        return embedder, float(threshold)
    raise ValueError(f"unknown embedder type {spec.type!r}")


def build_emotion(config: AppConfig, spec: ModelSpec):
    if spec.type == "dummy":
        return DummyEmotion()
    if spec.type == "emotion_gru":
        from .emotion import EmotionGRUPredictor

        return EmotionGRUPredictor(_require(config, spec), device=config.device)
    raise ValueError(f"unknown emotion type {spec.type!r}")


def _warm_up(bundle: ModelBundle) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (220, 140, 420, 340)
    from src.common.face_crop import crop_face

    bundle.detector.detect(frame)
    bundle.liveness.predict_frame(frame, box)
    size, scale = bundle.embed_crop
    bundle.embedder.embed(crop_face(frame, box, size, scale))
    bundle.emotion.predict(
        crop_face(frame, box, bundle.emotion.img_size, bundle.emotion.crop_scale)
    )


def build_models(config: AppConfig) -> ModelBundle:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    specs = config.models
    started = time.perf_counter()
    detector = build_detector(config, specs.detector)
    liveness = build_liveness(config, specs.liveness)
    embedder, recognition_threshold = build_embedder(config, specs.embedder)
    emotion = build_emotion(config, specs.emotion)
    info = {
        "detector": {"type": specs.detector.type, "path": specs.detector.path},
        "liveness": {
            "type": specs.liveness.type,
            "path": specs.liveness.path,
            "checkpoint_threshold": float(liveness.threshold),
            "img_size": liveness.img_size,
            "crop_scale": liveness.crop_scale,
        },
        "embedder": {
            "type": specs.embedder.type,
            "path": specs.embedder.path,
            "embedding_dim": int(embedder.embedding_dim),
            "model_version": str(embedder.model_version),
            "checkpoint_threshold": recognition_threshold,
        },
        "emotion": {
            "type": specs.emotion.type,
            "path": specs.emotion.path,
            "labels": list(emotion.labels),
        },
    }
    bundle = ModelBundle(detector, liveness, embedder, emotion, info, recognition_threshold)
    _warm_up(bundle)
    info["load_seconds"] = round(time.perf_counter() - started, 2)
    return bundle
