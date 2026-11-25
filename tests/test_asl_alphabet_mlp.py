"""Verify reusable ASL alphabet training and held-out checkpoint inference."""

import csv
import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training.config import TrainingConfig

mlp = import_module("src.hand-gestures.alphabet_mlp")


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_alphabet_mlp_checkpoint_and_clean_evaluation(tmp_path, device):
    cuda = device == "cuda"
    if cuda and (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        pytest.skip("native CUDA BF16 is unavailable")
    settings = TrainingConfig(
        epochs=2,
        batch_size=4,
        lr=0.005,
        weight_decay=0.0005,
        optimizer="adamw",
        scheduler="none",
        seed=42,
        device=device,
        amp=cuda,
        amp_dtype="bf16",
        deterministic=True,
        num_workers=0,
        monitor="val_macro_f1",
        mode="max",
        progress=False,
    )
    config = SimpleNamespace(
        training=settings,
        dropout=0.15,
        flip_probability=0.5,
        rotation_degrees=8,
        jitter_std=0.008,
    )
    points = np.random.default_rng(42).normal(size=(20, 21, 3)).astype(np.float32)
    labels = np.array([0, 1] * 8 + [-1] * 4, dtype=np.int64)
    splits = np.repeat(["train", "val", "test"], [12, 4, 4])
    metadata = {"model": "mlp", "classes": ["A", "B"]}
    mlp.train(config, points, labels, splits, metadata, tmp_path)
    saved, predict = mlp.load_alphabet_predictor(tmp_path)
    scores = predict(points[-4:])
    assert saved == metadata and scores.shape == (4, 2)
    np.testing.assert_allclose(scores.sum(1), 1, atol=1e-6)
    np.testing.assert_array_equal(scores, predict(points[-4:]))
    status = json.loads((tmp_path / "environment.json").read_text())
    assert status["precision"] == ("bf16-mixed" if cuda else "fp32")
    with (tmp_path / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert {"train_clean_accuracy", "val_accuracy", "val_macro_f1"} <= rows[0].keys()
