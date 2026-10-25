"""Shared face crop, used for training and in the live app. Owner: Thành."""

import cv2
import numpy as np

from .interfaces import Box


def crop_face(frame: np.ndarray, box: Box, size: int = 112, scale: float = 1.0) -> np.ndarray:
    """Crop one face from a BGR frame and resize it to `size` x `size`.

    The crop is a square centred on the box, with side `scale` x the longer box side.
    Recognition and emotion use the tight crop (scale 1.0); anti-spoofing uses a wider one
    so spoof cues around the face (phone edges, paper borders) stay in view.
    Parts of the crop outside the frame are filled with black.

    Every model is trained on this output, so changing it means retraining.
    """
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, 1) * scale
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    right, bottom = left + int(round(side)), top + int(round(side))

    h, w = frame.shape[:2]
    pad = max(0, -left, -top, right - w, bottom - h)
    if pad:
        frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        left, top, right, bottom = left + pad, top + pad, right + pad, bottom + pad

    face = frame[top:bottom, left:right]
    interp = cv2.INTER_AREA if face.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(face, (size, size), interpolation=interp)
