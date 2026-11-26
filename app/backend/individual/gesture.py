"""Đức Anh: dynamic ASL signs (J, Z, HELLO, SORRY, THANK_YOU, YES, NO) from hand landmarks.

Uses Đức Anh's code in src/hand-gestures: the MediaPipe hand tracker (landmarks.py), the
sequence features (features.py) and the MLP + GRU classifier (models.py), the same path as his
camera preview (test-cam.py). Faces are not detected in this mode.
"""

from __future__ import annotations

import time
from collections import deque
from contextlib import ExitStack
from importlib import import_module
from typing import Any

import numpy as np
import torch

from . import Feature as BaseFeature

WINDOW_SECONDS = 2.0  # cửa sổ chuyển động, giống mặc định --window của test-cam.py


def _module(name: str):
    return import_module(f"src.hand-gestures.{name}")


class DynamicGestureModel:
    def __init__(self, path, hand_model):
        import_module("mediapipe")  # báo lỗi sớm nếu chưa cài requirements/hand-gestures.txt
        features, models = _module("features"), _module("models")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("checkpoint_kind") != "inference" or saved.get("format_version") != 1:
            raise ValueError(f"not an inference gesture checkpoint: {path}")
        metadata = saved["metadata"]
        if metadata.get("model") != "mlp_gru":
            raise ValueError(f"expected an mlp_gru run, got {metadata.get('model')!r}")
        self.settings = metadata["preprocess"]
        if self.settings["include_body"]:
            raise ValueError("this run needs body anchors; only hand-only runs are supported")
        columns = features.sequence_columns(False)
        if list(columns) != metadata["columns"]:
            raise ValueError("checkpoint feature columns do not match src/hand-gestures/features.py")
        self.classes: list[str] = list(metadata["classes"])
        net = models.GestureGRU(len(columns), len(self.classes))
        net.load_state_dict({k.removeprefix("model."): v for k, v in saved["task"].items()})
        self.net = net.eval()
        self.hand_model = hand_model.read_bytes()
        self.val_metrics = {k: v for k, v in (saved.get("val_metrics") or {}).items()}

    @torch.inference_mode()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.net(torch.from_numpy(sequence[None])).softmax(dim=1)[0].numpy()


class Feature(BaseFeature):
    uses_faces = False

    @staticmethod
    def load(config, spec):
        if spec.type != "asl_dynamic":
            raise ValueError(f"unknown gesture model type {spec.type!r}")
        extra = spec.model_extra or {}
        hand_model = config.path(extra.get("hand_model", ""))
        if not hand_model.is_file():
            raise FileNotFoundError(f"MediaPipe hand model not found: {hand_model}")
        model = DynamicGestureModel(config.path(spec.path or ""), hand_model)
        return model, {
            "classes": model.classes,
            "window_s": WINDOW_SECONDS,
            "sequence_length": model.settings["sequence_length"],
            "val_accuracy": model.val_metrics.get("val_accuracy"),
            "val_clip_macro_f1": model.val_metrics.get("val_clip_macro_f1"),
        }

    def __init__(self, ctx, model, pipeline):
        super().__init__(ctx, model, pipeline)
        self.features = _module("features")
        self._stack = ExitStack()
        # mỗi stream một bộ theo dõi tay riêng (MediaPipe VIDEO cần timestamp tăng dần)
        self._detect = self._stack.enter_context(_module("landmarks").video_landmarker(model.hand_model))
        self._started = time.monotonic()
        self._last_ms = -1
        self._history: deque = deque()

    def close(self) -> None:
        self._stack.close()

    def frame(self, frame, clock) -> dict[str, Any]:
        features, settings = self.features, self.model.settings
        elapsed = time.monotonic() - self._started
        self._last_ms = max(self._last_ms + 1, int(elapsed * 1000))
        points, side, _, anchors = clock.run("hand", self._detect, frame, self._last_ms)
        self._history.append((elapsed, points, side, anchors))
        while self._history[0][0] < elapsed - WINDOW_SECONDS:
            self._history.popleft()

        height, width = frame.shape[:2]
        times, hands, sides, bodies = zip(*self._history, strict=True)
        clip = features.LandmarkClip(
            np.array(times), np.stack(hands), np.array(sides), np.stack(bodies), width, height
        )
        result: dict[str, Any] = {
            "status": "no_hand",
            "label": None,
            "confidence": None,
            "top": [],
            "progress": 0.0,
            "handedness": side or None,
            "points": None if not np.isfinite(points).all() else np.round(points[:, :2], 4).tolist(),
        }
        if not clip.observed()[-1]:
            return result

        # webcam qua trình duyệt thường chậm hơn 20 fps lúc train: nới khoảng hở cho phép theo fps thật
        gaps = np.diff(clip.timestamps)
        max_gap = settings["max_hand_gap_sec"]
        if len(gaps):
            max_gap = max(max_gap, 2.5 * float(np.median(gaps)))
        track = clip.select(features.track_spans(clip, max_gap)[-1])
        duration = track.timestamps[-1] - track.timestamps[0]
        result.update(status="collecting", progress=round(min(1.0, duration / (WINDOW_SECONDS / 2)), 2))
        if duration < WINDOW_SECONDS / 2:
            return result
        try:
            sequence = features.sequence_features(
                track,
                length=settings["sequence_length"],
                min_coverage=settings["min_hand_coverage"],
                max_gap=max_gap,
                include_body=False,
            )
        except ValueError:
            return result
        probs = clock.run("gesture", self.model.predict, sequence)
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        label = self.model.classes[best]
        result.update(
            status="other" if label == "OTHER" else "detected",
            label=label,
            confidence=round(float(probs[best]), 4),
            top=[
                {"label": self.model.classes[i], "prob": round(float(probs[i]), 4)}
                for i in order[:3]
            ],
            progress=1.0,
        )
        return result
