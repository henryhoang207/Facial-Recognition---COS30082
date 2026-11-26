from importlib import import_module

import numpy as np

landmarks = import_module("src.hand-gestures.landmarks")


def test_cache_requires_matching_metadata_and_arrays(tmp_path):
    path = tmp_path / "clip.npz"
    metadata = {"extractor": "model1", "rotation": 90, "sha256": "clip"}
    arrays = {
        "timestamps": np.array([0.0, 0.05]),
        "hand": np.zeros((2, 21, 3)),
        "handedness": np.array(["Right", "Right"]),
        "handedness_score": np.ones(2),
        "anchors": np.full((2, 4, 2), np.nan),
        "observed": np.ones(2, dtype=bool),
        "anchor_valid": np.zeros((2, 4), dtype=bool),
        "width": 480,
        "height": 640,
    }
    np.savez_compressed(path, **metadata, **arrays)
    cached = landmarks._check_cache(path, metadata)
    np.testing.assert_array_equal(cached["hand"], arrays["hand"])
    assert landmarks._check_cache(path, {**metadata, "rotation": 0}) is None
    assert landmarks._check_cache(path, {**metadata, "extractor": "model2"}) is None
    np.savez_compressed(path, **metadata, **{**arrays, "timestamps": np.zeros(2)})
    assert landmarks._check_cache(path, metadata) is None
