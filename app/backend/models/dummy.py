"""Dummy models with the same contract as the real predictors (mock mode, tests)."""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from src.common.face_crop import crop_face
from src.common.interfaces import Box

EMOTION_LABELS = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]


class DummyLiveness:
    img_size, crop_scale, threshold = 224, 2.0, 0.5

    def __init__(self) -> None:
        self.force_spoof = False

    def predict(self, face: np.ndarray) -> tuple[str, float]:
        prob = 0.05 if self.force_spoof else 0.95
        return ("live" if prob >= self.threshold else "spoof"), prob

    def predict_frame(self, frame: np.ndarray, box: Box) -> tuple[str, float]:
        return self.predict(crop_face(frame, box, self.img_size, self.crop_scale))


class PixelEmbedder:
    """Ảnh xám 32x32 chiếu ngẫu nhiên (seed cố định) xuống `dim` chiều, giống baseline 'no learning'."""

    crop_size, crop_scale = 112, 1.0

    def __init__(self, dim: int = 128, seed: int = 0):
        self.embedding_dim = dim
        rng = np.random.default_rng(seed)
        self.projection = rng.standard_normal((32 * 32, dim)).astype(np.float32)
        self.model_version = f"dummy-pixel-{dim}-{seed}"
        self.preprocessing_version = hashlib.sha256(b"gray32").hexdigest()

    def embed(self, face: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        pixels = cv2.resize(gray, (32, 32)).astype(np.float32).ravel()
        pixels = (pixels - pixels.mean()) / (pixels.std() + 1e-6)
        vector = pixels @ self.projection
        return (vector / (np.linalg.norm(vector) + 1e-12)).astype(np.float32)


class DummyEmotion:
    img_size, crop_scale = 224, 1.0
    labels = EMOTION_LABELS

    def predict(self, face: np.ndarray) -> tuple[str, dict[str, float]]:
        # xác suất giả nhưng ổn định theo độ sáng khuôn mặt
        level = float(face.mean()) / 255.0
        logits = np.array([np.cos(level * 6 + i) for i in range(len(self.labels))])
        probs = np.exp(logits * 3) / np.exp(logits * 3).sum()
        scores = {label: float(p) for label, p in zip(self.labels, probs, strict=True)}
        return max(scores, key=scores.get), scores

    def predict_track(self, track_id: int, face: np.ndarray):
        return self.predict(face)

    def drop_track(self, track_id: int) -> None:
        pass
