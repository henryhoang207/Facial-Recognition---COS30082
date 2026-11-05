"""Checksum-pinned YuNet detection in original image coordinates."""

import hashlib
import math
import os
import urllib.request
from dataclasses import asdict, dataclass
from uuid import uuid4

import cv2
import numpy as np

from .config import DetectorConfig


def ensure_model(config: DetectorConfig) -> None:
    if config.model.exists():
        if hashlib.sha256(config.model.read_bytes()).hexdigest() != config.sha256:
            raise ValueError(f"detector checksum mismatch: {config.model}")
        return
    config.model.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.model.with_suffix(f".{uuid4().hex}.download")
    try:
        with urllib.request.urlopen(config.url, timeout=30) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != config.sha256:
            raise ValueError("downloaded detector checksum mismatch")
        temporary.write_bytes(content)
        os.replace(temporary, config.model)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class FaceCandidate:
    box: tuple[int, int, int, int]
    confidence: float
    landmarks: tuple[tuple[float, float], ...]


class YuNetDetector:
    """One instance per worker; original images are letterboxed without distortion."""

    def __init__(self, config: DetectorConfig):
        self.config = config
        self.detector = cv2.FaceDetectorYN.create(
            str(config.model),
            "",
            (config.input_size, config.input_size),
            config.score_threshold,
            config.nms_threshold,
            config.top_k,
        )

    def candidates(self, frame: np.ndarray) -> list[FaceCandidate]:
        height, width = frame.shape[:2]
        side = self.config.input_size
        scale = min(side / width, side / height)
        resized_w, resized_h = max(1, round(width * scale)), max(
            1, round(height * scale)
        )
        left, top = (side - resized_w) // 2, (side - resized_h) // 2
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[top : top + resized_h, left : left + resized_w] = cv2.resize(
            frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA
        )
        _, faces = self.detector.detect(canvas)
        result = []
        for face in [] if faces is None else faces:
            x, y, w, h = map(float, face[:4])
            # Rounding each resized side can change its exact scale by one pixel.
            sx, sy = resized_w / width, resized_h / height
            box = (
                max(0, min(width, math.floor((x - left) / sx))),
                max(0, min(height, math.floor((y - top) / sy))),
                max(0, min(width, math.ceil((x + w - left) / sx))),
                max(0, min(height, math.ceil((y + h - top) / sy))),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            landmarks = tuple(
                ((float(px) - left) / sx, (float(py) - top) / sy)
                for px, py in face[4:14].reshape(5, 2)
            )
            result.append(FaceCandidate(box, float(face[-1]), landmarks))
        return sorted(
            result, key=lambda candidate: (-candidate.confidence, candidate.box)
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        return [candidate.box for candidate in self.candidates(frame)]


def inspect_face(
    frame, candidates: list[FaceCandidate], config: DetectorConfig
) -> dict:
    """Conservatively accept one plausible face; never guess between identities."""
    height, width = frame.shape[:2]
    status = "no_face" if not candidates else "multiple_faces"
    selected = None
    if len(candidates) == 1:
        selected = candidates[0]
        x1, y1, x2, y2 = selected.box
        fraction = (x2 - x1) * (y2 - y1) / (width * height)
        distance = math.hypot(
            (x1 + x2) / (2 * width) - 0.5, (y1 + y2) / (2 * height) - 0.5
        )
        if min(x2 - x1, y2 - y1) < config.min_face_pixels:
            status = "small_face"
        elif fraction < config.min_face_area_fraction:
            status = "small_face"
        elif distance > config.max_center_distance:
            status = "off_center"
        else:
            status = "accepted"
    return {
        "width": width,
        "height": height,
        "status": status,
        "box": list(selected.box) if status == "accepted" else None,
        "confidence": selected.confidence if selected else None,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
