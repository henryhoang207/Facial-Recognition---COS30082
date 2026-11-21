"""Batched landmark augmentation and the shared static pose representation."""

import math

import numpy as np
import torch

from .features import MIN_SCALE


def class_weights(labels: np.ndarray, class_count: int) -> np.ndarray:
    """Bound the correction for class imbalance after landmark extraction."""
    counts = np.bincount(labels, minlength=class_count)
    if len(counts) != class_count or not counts.all():
        raise ValueError(
            "usable training landmarks are required for each alphabet class"
        )
    return np.clip(len(labels) / (class_count * counts), 0.5, 2).astype(np.float32)


def augment_points(
    points: torch.Tensor,
    flip_probability: float,
    rotation_degrees: float,
    jitter_std: float,
) -> torch.Tensor:
    centered = points.float() - points[:, :1].float()
    batch = len(centered)
    angle = (2 * torch.rand(batch, 1, device=points.device) - 1) * math.radians(
        rotation_degrees
    )
    mirrored = torch.rand(batch, 1, device=points.device) < flip_probability
    x, y, z = centered.unbind(-1)
    x = torch.where(mirrored, -x, x)
    sin, cos = angle.sin(), angle.cos()
    rotated = torch.stack((x * cos - y * sin, x * sin + y * cos, z), dim=-1)
    if jitter_std:
        scale = torch.linalg.vector_norm(centered[:, 9], dim=-1)[:, None, None]
        rotated = rotated + torch.randn_like(rotated) * (jitter_std * scale)
    return rotated


def pose_features(points: torch.Tensor) -> torch.Tensor:
    """Match the existing NumPy static feature order with FP32 geometry."""
    hand = points.float() - points[:, :1].float()
    scale = torch.linalg.vector_norm(hand[:, 9], dim=-1).clamp_min(MIN_SCALE)
    hand = hand / scale[:, None, None]
    apertures = torch.linalg.vector_norm(
        hand[:, [4, 4, 8]] - hand[:, [8, 12, 12]], dim=-1
    )
    digits = hand[:, 1:].reshape(-1, 5, 4, 3)
    bones = digits.diff(dim=2)
    left, right = -bones[:, :, :-1], bones[:, :, 1:]
    divisor = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    valid = divisor > MIN_SCALE
    cosine = (left * right).sum(dim=-1) / divisor.clamp_min(MIN_SCALE)
    angles = torch.where(valid, cosine.clamp(-1, 1).acos(), 0)
    return torch.cat(
        (hand.flatten(1), apertures, angles.flatten(1), valid.flatten(1)), dim=1
    )
