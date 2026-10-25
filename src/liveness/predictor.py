"""Liveness model for the web app. Implements `LivenessModel` from src/common/interfaces.py.

    liveness = LivenessPredictor("checkpoints/liveness_mobilenetv3_2026-11-01.pt")
    face = crop_face(frame, box, size=liveness.img_size, scale=liveness.crop_scale)
    label, prob_live = liveness.predict(face)

Liveness uses a wider crop than recognition, so crop with the predictor's own
img_size and crop_scale (or call predict_frame with the full frame).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from src.common.face_crop import crop_face
from src.common.interfaces import Box

from .data import eval_transform
from .evaluate import load_checkpoint


class LivenessPredictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, cfg, self.threshold = load_checkpoint(Path(checkpoint), self.device)
        self.img_size, self.crop_scale = cfg["img_size"], cfg["crop_scale"]
        self.transform = eval_transform()

    @torch.no_grad()
    def predict(self, face: np.ndarray) -> tuple[str, float]:
        """`face`: BGR crop of img_size x img_size. Returns ("live" | "spoof", P(live))."""
        x = self.transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        prob = float(self.model.score(self.model(x))[0])
        return ("live" if prob >= self.threshold else "spoof"), prob

    def predict_frame(self, frame: np.ndarray, box: Box) -> tuple[str, float]:
        return self.predict(crop_face(frame, box, size=self.img_size, scale=self.crop_scale))
