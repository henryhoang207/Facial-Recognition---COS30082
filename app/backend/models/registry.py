"""Build the four models from app/config.yaml. The app only talks to the shared interfaces."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import AppConfig, ModelSpec
from .detectors import CenterDetector, HaarDetector, YuNetDetector, ensure_file
from .dummy import DummyEmotion, DummyLiveness, DummyMask, PixelEmbedder

DEFAULT_RECOGNITION_THRESHOLD = 0.5
log = logging.getLogger("faceguard")


@dataclass
class ModelBundle:
    detector: Any
    liveness: Any
    embedder: Any
    emotion: Any
    info: dict[str, dict[str, Any]]
    checkpoint_recognition_threshold: float
    mask: Any = None
    mask_detector_score: float | None = None
    attendance: Any = None
    individual: dict[str, Any] = field(default_factory=dict)
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
    if path.exists():
        return path
    # máy mỗi người có thể đặt tên checkpoint khác nhau: thử lần lượt các tên dự phòng trong config
    for fallback in (spec.model_extra or {}).get("fallback_paths", []):
        if config.path(fallback).exists():
            log.warning("checkpoint %s not found, using %s", spec.path, fallback)
            spec.path = fallback  # để /api/health hiện đúng file đang dùng
            return config.path(fallback)
    raise FileNotFoundError(f"checkpoint not found: {path}")


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


def build_mask(config: AppConfig, spec: ModelSpec | None):
    """Trả về (model khẩu trang | None, ngưỡng score detector khi chạy mode mask)."""
    if spec is None:
        return None, None
    score = (spec.model_extra or {}).get("detector_score")
    score = None if score is None else float(score)
    if spec.type == "dummy":
        return DummyMask(), score
    if spec.type == "mask":
        path = config.path(spec.path or "")
        if not spec.path or not path.exists():
            # model khẩu trang là tính năng phụ: thiếu checkpoint thì app vẫn chạy, chỉ tắt tab Mask
            log.warning("mask checkpoint not found (%s); mask mode disabled", path)
            return None, score
        from src.mask.predictor import MaskPredictor

        return MaskPredictor(path, device=config.device), score
    raise ValueError(f"unknown mask type {spec.type!r}")


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
    if bundle.mask is not None:
        bundle.mask.predict_frame(frame, box)


def build_attendance(config: AppConfig, spec: ModelSpec | None, model_version: str):
    if spec is None:
        return None
    if spec.type != "mlp":
        raise ValueError(f"unknown attendance model type {spec.type!r}")
    from src.recognition.attendance.model import AttendancePredictor

    try:
        predictor = AttendancePredictor(_require(config, spec), encoder_sha256=model_version)
        if predictor.frame_count != 5:
            raise ValueError("attendance MLP must use five frames")
    except (FileNotFoundError, ValueError) as exc:
        log.warning("attendance MLP unavailable: %s", exc)
        return None
    return predictor


def build_models(config: AppConfig) -> ModelBundle:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    specs = config.models
    started = time.perf_counter()
    detector = build_detector(config, specs.detector)
    liveness = build_liveness(config, specs.liveness)
    embedder, recognition_threshold = build_embedder(config, specs.embedder)
    emotion = build_emotion(config, specs.emotion)
    mask, mask_detector_score = build_mask(config, specs.mask)
    attendance = build_attendance(config, specs.attendance, str(embedder.model_version))
    from ..individual import build_individual

    individual, individual_info = build_individual(config, specs.individual)
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
        "mask": None
        if mask is None
        else {
            "type": specs.mask.type,
            "path": specs.mask.path,
            "arch": mask.arch,
            "checkpoint_threshold": float(mask.threshold),
            "img_size": mask.img_size,
            "crop_scale": mask.crop_scale,
            "detector_score": mask_detector_score,
        },
    }
    if attendance is not None:
        info["attendance"] = {
            "type": "mlp",
            "path": specs.attendance.path,
            "threshold": attendance.threshold,
            "frame_count": attendance.frame_count,
            "gallery_size": attendance.gallery_size,
        }
    bundle = ModelBundle(
        detector, liveness, embedder, emotion, info, recognition_threshold,
        mask, mask_detector_score, attendance, individual,
    )
    info["individual"] = individual_info
    _warm_up(bundle)
    info["load_seconds"] = round(time.perf_counter() - started, 2)
    return bundle
