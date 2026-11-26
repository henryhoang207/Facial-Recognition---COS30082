"""Static pose parity and geometry for device-side augmentation."""

from importlib import import_module

import numpy as np
import torch

alphabet = import_module("src.hand-gestures.alphabet_features")
features = import_module("src.hand-gestures.features")


def test_geometry_matches_numpy_and_flips_only_horizontal_coordinates():
    points = np.random.default_rng(42).uniform(0, 1, (8, 21, 3)).astype(np.float32)
    points[0, 2] = points[0, 1]
    original = torch.from_numpy(points)
    pose = alphabet.pose_features(original).numpy()
    np.testing.assert_allclose(pose, features.local_pose(points), atol=1e-5)
    flipped = alphabet.augment_points(original, 1, 0, 0)
    mirrored_pose = alphabet.pose_features(flipped).numpy()
    expected = pose.copy()
    expected[:, :63:3] *= -1
    np.testing.assert_allclose(mirrored_pose, expected, atol=1e-5)
    np.testing.assert_array_equal(original.numpy(), points)
