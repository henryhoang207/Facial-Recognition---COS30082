"""A/C configuration, training, inference and shared verification checks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from src.recognition.classification.config import RecognitionConfig, load_config
from src.recognition.classification.data import FaceTransform, make_loaders
from src.recognition.classification.evaluate import evaluate
from src.recognition.classification.models import ArcFaceHead
from src.recognition.classification.predictor import RecognitionEmbedder
from src.recognition.classification.train import train
from src.recognition.verification import (
    PairEvaluator,
    VerificationConfig,
    score_metrics,
)
from src.recognition.verification.run import EvaluationConfig, VerificationRunner
from src.training.config import read_yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def tiny_config(tmp_path, variant="a", device="cpu"):
    rng = np.random.default_rng(42)
    for split in ("train_data", "val_data", "test_data"):
        for identity in ("alice", "bob"):
            directory = tmp_path / "data" / split / identity
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(2):
                image = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
                assert cv2.imwrite(str(directory / f"{index}.png"), image)
    raw = read_yaml(REPO / f"configs/recognition/model_{variant}.yaml")
    raw["data"].update(root=str(tmp_path / "data"), manifest=None)
    raw["output_dir"] = str(tmp_path / "runs")
    raw["preprocess"].update(size=32, horizontal_flip=0, brightness=0)
    raw["training"].update(
        epochs=1,
        batch_size=2,
        num_workers=0,
        monitor="val_loss",
        mode="min",
        device=device,
        progress=False,
        amp_init_scale=1.0,
    )
    raw["verification"] = None
    return RecognitionConfig.model_validate(raw)


def test_yaml_cli_overrides_do_not_mutate_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(read_yaml(REPO / "configs/recognition/model_a.yaml")),
        encoding="utf-8",
    )
    before = path.read_bytes()
    cfg, _ = load_config(
        [
            "--config",
            str(path),
            "--lr",
            "0.005",
            "--epochs",
            "2",
            "--set",
            "preprocess.horizontal_flip=0.25",
        ]
    )
    assert cfg.training.lr == 0.005 and cfg.training.epochs == 2
    assert cfg.preprocess.horizontal_flip == 0.25
    assert cfg.training.batch_size == 64
    assert path.read_bytes() == before
    grouped, _ = load_config(
        [
            "--config",
            str(path),
            "--optimizer",
            "sgd",
            "--warmup-epochs",
            "3",
            "--accumulation-steps",
            "4",
            "--group-lr",
            "backbone=0.0001",
            "--group-lr",
            "head=0.001",
            "--set",
            "training.sgd_momentum=0.9",
            "--set",
            "training.sgd_nesterov=true",
        ]
    )
    assert grouped.training.warmup_epochs == 3
    assert grouped.training.accumulation_steps == 4
    assert grouped.training.sgd_nesterov is True
    assert [(group.name, group.lr) for group in grouped.training.parameter_groups] == [
        ("backbone", 0.0001),
        ("head", 0.001),
    ]
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(["--config", str(path), "--set", "training.typo=2"])
    with pytest.raises(ValueError, match="unique"):
        path.write_text("training: {}\ntraining: {}\n", encoding="utf-8")
        load_config(["--config", str(path)])


def test_bgr_preprocessing_and_arcface_target_margin(tmp_path):
    cfg = tiny_config(tmp_path)
    face = np.zeros((32, 32, 3), dtype=np.uint8)
    face[:, :, 2] = 255
    tensor = FaceTransform(cfg.preprocess)(face)
    assert tensor[:, 0, 0].tolist() == [1.0, -1.0, -1.0]
    head = ArcFaceHead(2, 2, scale=30, margin=0.3)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    features = torch.tensor([[0.8, 0.6]], requires_grad=True)
    logits = head(features, torch.tensor([0]))
    assert logits[0, 0].item() == pytest.approx(17.6081, abs=0.001)
    assert logits[0, 1].item() == pytest.approx(18, abs=0.001)
    logits.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert head.weight.grad is not None


@pytest.mark.parametrize("variant", ["a", "c"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_train_checkpoint_embed_and_classification_test(tmp_path, variant, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    cfg = tiny_config(tmp_path, variant, device)
    result = train(cfg)
    embedder = RecognitionEmbedder(result.best_checkpoint)
    face = np.random.default_rng(5).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    vector = embedder.embed(face)
    assert vector.shape == (128,) and vector.dtype == np.float32
    assert np.isfinite(vector).all() and np.linalg.norm(vector) == pytest.approx(
        1, abs=1e-5
    )
    scores = evaluate(result.best_checkpoint, split="test", device="cpu")
    assert scores["images"] == 4 and scores["split"] == "test"
    assert 0 <= scores["accuracy"] <= 1
    assert 0 <= result.history[0]["val_accuracy"] <= 1
    assert result.history[0]["train_optimizer_steps"] > 0
    if variant == "c":
        payload = torch.load(result.last_checkpoint, weights_only=True)
        assert "model.head.weight" in payload["task"]


def test_training_selects_checkpoint_using_verification_auc(tmp_path):
    cfg = tiny_config(tmp_path)
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "alice/0.png alice/1.png 1\n"
        "bob/0.png bob/1.png 1\n"
        "alice/0.png bob/0.png 0\n"
        "alice/1.png bob/1.png 0\n",
        encoding="utf-8",
    )
    cfg = cfg.model_copy(
        update={
            "verification": VerificationConfig(
                root=cfg.data.root / "val_data",
                pairs=pairs,
                pairs_format="whitespace",
                target_fpr=0.01,
                selection_metric="cosine",
            ),
            "training": cfg.training.model_copy(
                update={"monitor": "val_auc", "mode": "max"}
            ),
        }
    )
    result = train(cfg)
    payload = torch.load(result.best_checkpoint, weights_only=True)
    assert payload["best"] == result.history[0]["val_auc"]
    assert 0 <= payload["best"] <= 1
    assert result.history[0]["val_fpr"] == 0


def test_unknown_class_and_train_val_path_overlap_are_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    (tmp_path / "data/val_data/bob").rename(tmp_path / "data/val_data/charlie")
    with pytest.raises(ValueError, match="same set"):
        make_loaders(cfg)
    (tmp_path / "data/val_data/charlie").rename(tmp_path / "data/val_data/bob")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,identity,split\ntrain_data/alice/0.png,alice,train\ntrain_data/alice/0.png,alice,val\n",
        encoding="utf-8",
    )
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"manifest": manifest})}
    )
    with pytest.raises(ValueError, match="duplicate"):
        make_loaders(cfg)


def test_roc_auc_direction_threshold_and_fixed_test_threshold():
    labels = np.array([1, 1, 0, 0])
    metrics, _ = score_metrics(labels, np.array([0.9, 0.8, 0.3, 0.1]), target_fpr=0.01)
    assert metrics["auc"] == 1 and metrics["tar"] == 1 and metrics["fpr"] == 0
    assert metrics["score_threshold"] == 0.8
    test, _ = score_metrics(
        labels,
        np.array([0.9, 0.7, 0.85, 0.1]),
        target_fpr=0.01,
        threshold=metrics["score_threshold"],
    )
    assert test["tar"] == 0.5 and test["fpr"] == 0.5
    tied, _ = score_metrics(labels, np.ones(4), target_fpr=0.01)
    assert tied["tar"] == 0 and tied["fpr"] == 0 and tied["score_threshold"] > 1


def verification_fixture(tmp_path):
    # Pure-color crops map to known unit vectors in ColorEmbedder.
    for name, pixel in [
        ("a", (0, 0, 255)),
        ("b", (0, 0, 240)),
        ("c", (0, 255, 0)),
        ("d", (0, 240, 0)),
    ]:
        assert cv2.imwrite(
            str(tmp_path / f"{name}.png"), np.full((32, 32, 3), pixel, np.uint8)
        )
    pairs = tmp_path / "pairs.txt"
    pairs.write_text(
        "a.png b.png 1\nc.png d.png 1\na.png c.png 0\nb.png d.png 0\n", encoding="utf-8"
    )
    return VerificationConfig(
        root=tmp_path,
        pairs=pairs,
        pairs_format="whitespace",
        target_fpr=0.01,
        selection_metric="cosine",
    )


class ColorEmbedder:
    def embed(self, face):
        color = face[0, 0].astype(np.float32)
        return color / np.linalg.norm(color)


def test_shared_evaluator_cosine_euclidean_have_same_auc(tmp_path):
    config = verification_fixture(tmp_path)
    evaluator = PairEvaluator(config, size=32, crop_scale=1, progress=False)
    metrics, _, _, _ = evaluator.evaluate(ColorEmbedder())
    assert metrics["cosine"]["auc"] == 1 and metrics["euclidean"]["auc"] == 1
    assert metrics["cosine"]["similarity_min"] == 1
    assert metrics["euclidean"]["distance_max"] == 0


def test_verification_report_baseline_calibration_and_test_reuse(tmp_path):
    verification = verification_fixture(tmp_path)
    cfg = EvaluationConfig(
        verification=verification,
        models={},
        baseline=True,
        split="val",
        size=32,
        crop_scale=1,
        device="cpu",
        output_dir=tmp_path / "reports",
        calibration=None,
    )
    output = VerificationRunner(cfg).run()
    metrics_path = output / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    assert metrics["models"]["raw_pixels"]["metrics"]["cosine"]["auc"] == 1
    assert (output / "roc.png").stat().st_size > 1000
    test_pairs = tmp_path / "test_pairs.txt"
    test_pairs.write_text(
        "a.png d.png 0\nc.png b.png 0\na.png b.png 1\n", encoding="utf-8"
    )
    cfg = cfg.model_copy(
        update={
            "split": "test",
            "calibration": metrics_path,
            "verification": verification.model_copy(update={"pairs": test_pairs}),
        }
    )
    test_output = VerificationRunner(cfg).run()
    test_metrics = json.loads((test_output / "metrics.json").read_text())
    assert (
        test_metrics["models"]["raw_pixels"]["metrics"]["cosine"]["score_threshold"]
        == 1
    )
    with pytest.raises(ValueError, match="requires"):
        VerificationRunner(cfg.model_copy(update={"calibration": None})).run()
