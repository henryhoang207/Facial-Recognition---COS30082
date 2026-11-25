"""Face detectors implementing src.common.interfaces.Detector."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.common.interfaces import Box


def ensure_file(path: Path, url: str | None, sha256: str | None) -> None:
    if path.exists():
        if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"checksum mismatch: {path}")
        return
    if not url:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    if sha256 and hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("downloaded detector checksum mismatch")
    path.write_bytes(content)


def _clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> Box:
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(width, int(round(x + w))), min(height, int(round(y + h)))
    return x1, y1, x2, y2


def _by_area(boxes: list[Box]) -> list[Box]:
    valid = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    return sorted(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


class YuNetDetector:
    name = "yunet"

    def __init__(self, path: Path, score_threshold: float = 0.8):
        self.path = path
        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, 0.3, 50
        )
        self._size: tuple[int, int] | None = None

    def detect(self, frame: np.ndarray, score_threshold: float | None = None) -> list[Box]:
        height, width = frame.shape[:2]
        if self._size != (width, height):
            self.detector.setInputSize((width, height))
            self._size = (width, height)
        if score_threshold is None:
            _, faces = self.detector.detect(frame)
        else:
            # mặt đeo khẩu trang có score thấp hơn: hạ ngưỡng cho lần detect này rồi trả lại
            self.detector.setScoreThreshold(score_threshold)
            try:
                _, faces = self.detector.detect(frame)
            finally:
                self.detector.setScoreThreshold(self.score_threshold)
        if faces is None:
            return []
        return _by_area([_clip_box(*face[:4], width, height) for face in faces])


class HaarDetector:
    name = "haar"

    def __init__(self) -> None:
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade))

    def detect(self, frame: np.ndarray, score_threshold: float | None = None) -> list[Box]:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return _by_area([_clip_box(*face, width, height) for face in faces])


class CenterDetector:
    """Trả về một box cố định ở giữa frame; chỉ dùng cho test và chạy không có camera."""

    name = "center"

    def detect(self, frame: np.ndarray, score_threshold: float | None = None) -> list[Box]:
        height, width = frame.shape[:2]
        side = int(min(width, height) * 0.5)
        x1, y1 = (width - side) // 2, (height - side) // 2
        return [(x1, y1, x1 + side, y1 + side)]
