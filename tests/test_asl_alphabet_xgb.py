"""Verify alphabet booster training and native artifact loading."""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

alphabet_xgb = import_module("src.hand-gestures.alphabet_xgb")
features = import_module("src.hand-gestures.features")


def test_alphabet_booster_round_trip(tmp_path):
    xgb = pytest.importorskip("xgboost")
    config = SimpleNamespace(
        seed=42,
        training=SimpleNamespace(device="cpu", batch_size=8),
        flip_probability=0.5,
        rotation_degrees=8,
        jitter_std=0.008,
        xgb_rounds=2,
        xgb_patience=2,
    )
    labels = np.tile(np.arange(3), 12)
    points = np.random.default_rng(42).random((len(labels), 21, 3), dtype=np.float32)
    splits = np.repeat(["train", "val", "test"], 12)
    metadata = {
        "classes": ["A", "B", "C"],
        "columns": features.STATIC_COLUMNS,
        "schema": "test",
    }
    alphabet_xgb.train(config, points, labels, splits, metadata, tmp_path)
    predict = alphabet_xgb.load_alphabet_xgb(tmp_path, metadata)
    probabilities = predict(points[splits == "test"])
    assert probabilities.shape == (12, 3)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    if torch.cuda.is_available() and xgb.build_info().get("USE_CUDA"):
        predict_cuda = alphabet_xgb.load_alphabet_xgb(tmp_path, metadata, "cuda")
        np.testing.assert_allclose(
            predict_cuda(points[splits == "test"]), probabilities, atol=1e-6
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        alphabet_xgb.load_alphabet_xgb(tmp_path, {**metadata, "schema": "other"})
