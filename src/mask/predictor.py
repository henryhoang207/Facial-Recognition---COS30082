"""Mask model for the web app.

    mask = MaskPredictor("model/Mask/mask_mobilenetv3s_2026-11-13.pt")
    label, prob_mask = mask.predict_frame(frame, box)   # crops with its own size and scale
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.common.face_crop import crop_face
from src.common.interfaces import Box

from .model import eval_transform, load_checkpoint


class MaskPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, config = load_checkpoint(Path(checkpoint), self.device)
        self.arch: str = config["arch"]
        self.labels: list[str] = config["classes"]
        self.img_size: int = config["img_size"]
        self.crop_scale: float = config["crop_scale"]
        self.threshold: float = config["threshold"]
        self.transform = eval_transform(self.img_size)

    @torch.inference_mode()
    def predict(self, face: np.ndarray) -> tuple[str, float]:
        """`face`: BGR crop. Returns ("mask" | "no_mask", P(mask))."""
        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        prob_mask = float(torch.softmax(self.model(tensor), dim=1)[0, 0])
        return ("mask" if prob_mask >= self.threshold else "no_mask"), prob_mask

    def predict_frame(self, frame: np.ndarray, box: Box) -> tuple[str, float]:
        return self.predict(crop_face(frame, box, size=self.img_size, scale=self.crop_scale))
