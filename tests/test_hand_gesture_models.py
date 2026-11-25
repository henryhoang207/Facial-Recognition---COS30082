"""Small model and clip aggregation checks without videos or landmark assets."""

import json
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training.config import TrainingConfig

evaluation = import_module("src.hand-gestures.evaluate")
models = import_module("src.hand-gestures.models")
training = import_module("src.hand-gestures.train")


def sample_fixture(model, split):
    classes = tuple(f"L{i}" for i in range(24 if model == "static_xgb" else 8))
    labels = np.repeat(np.arange(len(classes), dtype=np.int64), 2)
    inputs = np.eye(len(classes), dtype=np.float32)[labels]
    if model == "mlp_gru":
        inputs = np.repeat(inputs[:, None, :], 32, axis=1)
    return SimpleNamespace(
        inputs=inputs,
        labels=labels,
        classes=classes,
        columns=classes,
        clip_ids=tuple(f"{split}/{label}" for label in labels),
        window_ids=tuple(f"{split}/{row}" for row in range(len(labels))),
        weights=np.full(len(labels), 0.5, dtype=np.float32),
        handedness=("Right",) * len(labels),
        anchor_coverage=np.ones(len(labels), dtype=np.float32),
        segments=("reviewed",) * len(labels),
        rejected={},
    )


def test_clip_aggregation_preserves_support_and_averages_probabilities():
    samples = sample_fixture("motion_xgb", "val")
    probabilities = np.eye(8, dtype=np.float32)[samples.labels]
    probabilities[0] = np.array([0.3, 0.7, 0, 0, 0, 0, 0, 0])
    report, predictions = evaluation.classification_report(samples, probabilities)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["per_class"]["L0"] == {"clips": 1, "windows": 2, "recall": 1.0}
    assert len(predictions) == 8
    assert predictions[0]["confidence"] == pytest.approx(0.65)


@pytest.mark.parametrize("model", models.MODEL_NAMES)
def test_artifact_produces_finite_normalized_probabilities(model, tmp_path):
    torch.set_num_threads(1)
    train = sample_fixture(model, "train")
    val = sample_fixture(model, "val")
    schema = evaluation.schema_digest(train)
    metadata = {
        "version": models.ARTIFACT_VERSION,
        "model": model,
        "columns": list(train.columns),
        "classes": list(train.classes),
        "schema": schema,
    }
    if model == "mlp_gru":
        settings = TrainingConfig(
            epochs=1,
            batch_size=4,
            lr=0.005,
            weight_decay=0,
            optimizer="adamw",
            scheduler="none",
            seed=42,
            device="cpu",
            amp=False,
            deterministic=True,
            num_workers=0,
            monitor="val_clip_macro_f1",
            mode="max",
            progress=False,
        )
        training.train_gru(
            SimpleNamespace(training=settings), train, val, tmp_path, metadata, None
        )
        saved = torch.load(tmp_path / "best.pt", weights_only=True)
        task = training.GestureTask(
            models.GestureGRU(len(train.columns), len(train.classes))
        )
        task.load_state_dict(saved["task"])
        task.eval()
        with torch.inference_mode():
            expected = task.model(torch.from_numpy(val.inputs)).softmax(1).numpy()
    else:
        xgb = pytest.importorskip("xgboost")
        training.train_xgboost(
            SimpleNamespace(seed=42, xgb_rounds=3), train, val, tmp_path, schema
        )
        booster = xgb.Booster()
        booster.load_model(tmp_path / "model.json")
        expected = booster.predict(xgb.DMatrix(val.inputs))
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    saved_metadata, predictor = models.load_predictor(tmp_path)
    assert saved_metadata == metadata
    actual = predictor(val.inputs)
    assert actual.shape == (len(val.labels), len(val.classes))
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.sum(1), 1, atol=1e-6)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
