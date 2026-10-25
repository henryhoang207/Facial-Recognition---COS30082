"""Shared face crop, used for training and in the live app. Owner: Thành."""

import numpy as np

from .interfaces import Box


def crop_face(frame: np.ndarray, box: Box, size: int = 112) -> np.ndarray:
    """Crop one face from a BGR frame and resize it to `size` x `size`.

    Every model is trained on this output, so changing it means retraining.
    """
    raise NotImplementedError("Thành: implement the shared face crop")
