"""Common-checkpoint inference and shared verification adapter contracts."""

import copy
import json

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from src.recognition.verification import predictor
from src.recognition.verification.config import VerificationConfig
from src.recognition.verification.run import (
    EvaluationConfig,
    ModelSpec,
    VerificationRunner,
)


class TinyEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.fc = nn.Linear(3, embedding_dim)
        with torch.no_grad():
            self.fc.weight.zero_()
            self.fc.weight[:3].copy_(torch.eye(3))
            self.fc.bias.zero_()
            self.fc.bias[3] = 1
        self.last_input = None

    def forward(self, images):
        self.last_input = images.detach().cpu().clone()
        return nn.functional.normalize(self.fc(images.mean((2, 3))), dim=1)


@pytest.fixture
def common_payload(monkeypatch):
    monkeypatch.setattr(predictor, "IResNet34", TinyEncoder)
    return {
        "format_version": 1,
        "checkpoint_kind": "inference",
        "task": {
            **{
                f"model.{key}": value
                for key, value in TinyEncoder().state_dict().items()
            },
            "_extra_state": {"strategy": "random", "gate_passes": 0},
        },
        "training_config": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "none",
            "seed": 42,
            "device": "cpu",
            "amp": False,
            "deterministic": False,
            "num_workers": 0,
            "monitor": "val_tar",
            "mode": "max",
        },
        "metadata": {
            "task": "recognition_metric_learning",
            "experiment_config": {"method": "triplet", "embedding_dim": 256},
            "method": "triplet",
            "margin": 0.3,
            "seed": 42,
            "class_count": 4,
            "embedding_dim": 256,
            "preprocessing": copy.deepcopy(predictor.METRIC_PREPROCESSING),
            "signature": "test-run-signature",
            "environment_compatibility": {"torch": str(torch.__version__)},
        },
        "val_metrics": {"dev_threshold": 0.4},
    }


@pytest.mark.parametrize("method", ["triplet", "arcface"])
def test_common_checkpoint_embeds_bgr_crop_and_ignores_loss_head(
    tmp_path, monkeypatch, common_payload, method
):
    common_payload["metadata"]["method"] = method
    common_payload["metadata"]["experiment_config"]["method"] = method
    if method == "arcface":
        common_payload["task"]["head.weight"] = torch.zeros(4, 256)
    checkpoint = tmp_path / "best.pt"
    torch.save(common_payload, checkpoint)
    original_load = torch.load
    load_options = []

    def safe_load(*args, **kwargs):
        load_options.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(predictor.torch, "load", safe_load)
    before = torch.get_rng_state().clone()
    embedder = predictor.MetricEmbedder(checkpoint)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(options["weights_only"] is True for options in load_options)
    face = np.zeros((112, 112, 3), dtype=np.uint8)
    face[:, :, 2] = 255
    vector = embedder.embed(face)
    torch.testing.assert_close(
        embedder.encoder.last_input[0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0])
    )
    assert vector.shape == (256,) and vector.dtype == np.float32
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)
    assert embedder.embedding_dim == 256
    assert not embedder.encoder.training
    assert len(embedder.model_version) == len(embedder.preprocessing_version) == 64
    assert embedder.payload["val_metrics"]["dev_threshold"] == 0.4


def test_training_checkpoint_requires_explicit_internal_opt_in(
    tmp_path, common_payload
):
    common_payload["checkpoint_kind"] = "training"
    checkpoint = tmp_path / "last.pt"
    torch.save(common_payload, checkpoint)
    with pytest.raises(ValueError, match="inference checkpoint"):
        predictor.MetricEmbedder(checkpoint)
    encoder, payload = predictor.load_metric_encoder(checkpoint, allow_training=True)
    assert isinstance(encoder, TinyEncoder)
    assert payload["checkpoint_kind"] == "training"


def test_encoder_only_export_matches_dataset_normalization(tmp_path, common_payload):
    common_payload["task"].pop("_extra_state")
    checkpoint = tmp_path / "embedding.pt"
    torch.save(common_payload, checkpoint)
    embedder = predictor.MetricEmbedder(checkpoint)
    face = np.random.default_rng(72).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    embedder.embed(face)
    rgb = np.ascontiguousarray(face[:, :, ::-1])
    expected = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    torch.testing.assert_close(
        embedder.encoder.last_input[0], torch.from_numpy(expected), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "format_version", 0, "inference checkpoint"),
        (None, "checkpoint_kind", "unknown", "inference checkpoint"),
        (None, "checkpoint_kind", {}, "inference checkpoint"),
        (None, "metadata", None, "metric-learning training artifact"),
        (None, "training_config", None, "common training_config"),
        (None, "task", {}, "metric task state"),
        ("metadata", "task", "recognition_classification", "training artifact"),
        ("metadata", "method", "classification", "metadata"),
        ("metadata", "method", {}, "metadata"),
        ("metadata", "embedding_dim", 128, "metadata"),
        ("metadata", "class_count", 1, "metadata"),
        ("metadata", "margin", float("nan"), "metadata"),
        ("metadata", "preprocessing", {"size": 224}, "preprocessing"),
        ("metadata", "signature", "", "metadata"),
        ("metadata", "environment_compatibility", None, "metadata"),
    ],
)
def test_rejects_incompatible_schema_or_metadata(
    tmp_path, common_payload, section, key, value, message
):
    target = common_payload if section is None else common_payload[section]
    target[key] = value
    checkpoint = tmp_path / "invalid.pt"
    torch.save(common_payload, checkpoint)
    with pytest.raises(ValueError, match=message):
        predictor.MetricEmbedder(checkpoint)


def test_rejects_legacy_metric_checkpoint_and_wrong_encoder_state(
    tmp_path, common_payload
):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": TinyEncoder().state_dict(), "embedding_dim": 256}, checkpoint)
    with pytest.raises(ValueError, match="inference checkpoint"):
        predictor.MetricEmbedder(checkpoint)
    common_payload["task"].pop("model.fc.weight")
    torch.save(common_payload, checkpoint)
    with pytest.raises(ValueError, match="weights do not match"):
        predictor.MetricEmbedder(checkpoint)
    common_payload["task"]["unrelated.weight"] = torch.ones(1)
    torch.save(common_payload, checkpoint)
    with pytest.raises(ValueError, match="task state layout"):
        predictor.MetricEmbedder(checkpoint)


@pytest.mark.parametrize(
    "face",
    [
        None,
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((112, 112), dtype=np.uint8),
        np.zeros((112, 112, 4), dtype=np.uint8),
        np.zeros((224, 224, 3), dtype=np.uint8),
        np.zeros((112, 112, 3), dtype=np.float32),
        np.full((112, 112, 3), np.nan, dtype=np.float32),
    ],
)
def test_rejects_invalid_shared_face_crop(tmp_path, common_payload, face):
    checkpoint = tmp_path / "best.pt"
    torch.save(common_payload, checkpoint)
    embedder = predictor.MetricEmbedder(checkpoint)
    with pytest.raises(ValueError, match="BGR uint8 crop"):
        embedder.embed(face)


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros(1, 256),
        torch.full((1, 256), float("nan")),
        torch.full((1, 256), float("inf")),
        torch.ones(1, 256),
        torch.ones(256),
        torch.ones(1, 128),
    ],
)
def test_rejects_invalid_encoder_output(tmp_path, common_payload, monkeypatch, output):
    checkpoint = tmp_path / "best.pt"
    torch.save(common_payload, checkpoint)
    embedder = predictor.MetricEmbedder(checkpoint)
    monkeypatch.setattr(embedder.encoder, "forward", lambda images: output)
    with pytest.raises(ValueError, match="embedding"):
        embedder.embed(np.zeros((112, 112, 3), dtype=np.uint8))


def test_shared_verification_runner_loads_metric_factory(tmp_path, common_payload):
    checkpoint = tmp_path / "best.pt"
    torch.save(common_payload, checkpoint)
    for name, color in [("a", (0, 0, 255)), ("b", (0, 0, 255)), ("c", (0, 255, 0))]:
        face = np.full((112, 112, 3), color, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), face)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text("a.png b.png 1\na.png c.png 0\n", encoding="utf-8")
    config = EvaluationConfig(
        verification=VerificationConfig(
            root=tmp_path,
            pairs=pairs,
            pairs_format="whitespace",
            target_fpr=0.01,
            selection_metric="cosine",
        ),
        models={
            "triplet": ModelSpec(
                checkpoint=checkpoint,
                factory="src.recognition.verification.predictor:MetricEmbedder",
            )
        },
        baseline=False,
        split="val",
        size=112,
        crop_scale=1.0,
        device="cpu",
        output_dir=tmp_path / "verification",
        calibration=None,
    )
    directory = VerificationRunner(config).run()
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert report["selected"]["model"] == "triplet"
    assert report["models"]["triplet"]["metrics"]["cosine"]["auc"] == 1
    assert (directory / "roc.png").is_file()
