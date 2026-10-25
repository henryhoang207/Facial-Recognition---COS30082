"""Interfaces every model exposes so the web app can plug it in.

Each owner implements one of these in their own package and keeps the
signature unchanged. `face` is always the output of `face_crop.crop_face`.
"""

from typing import Protocol

import numpy as np

# (x1, y1, x2, y2) in pixel coordinates of the original frame
Box = tuple[int, int, int, int]


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Box]:
        """Return one box per face found in a BGR webcam frame."""
        ...


class Embedder(Protocol):
    def embed(self, face: np.ndarray) -> np.ndarray:
        """Return a 1-D, L2-normalised face embedding."""
        ...


class AntiSpoofingModel(Protocol):
    def predict(self, face: np.ndarray) -> tuple[str, float]:
        """Return ("live" | "spoof", probability that the face is live)."""
        ...


class EmotionModel(Protocol):
    def predict(self, face: np.ndarray) -> tuple[str, dict[str, float]]:
        """Return (top emotion label, {label: probability} for every class)."""
        ...
