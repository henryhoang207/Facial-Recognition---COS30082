"""A/C configuration, training, inference and shared verification checks."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.recognition.classification.augmentation import FaceBatchTransform
from src.recognition.classification.cache import cache_budget, cache_loaders
from src.recognition.classification.config import (
    ModelConfig,
    RecognitionConfig,
    load_config,
)
from src.recognition.classification.data import FaceTransform, make_loaders
from src.recognition.classification.evaluate import evaluate
from src.recognition.classification.models import ArcFaceHead, ClassificationModel
from src.recognition.classification.predictor import RecognitionEmbedder, load_task
from src.recognition.classification.task import RecognitionTask
from src.recognition.classification.train import train
from src.recognition.classification.verification import TrainingVerificationEvaluator
from src.recognition.verification import (
    PairEvaluator,
    VerificationConfig,
    score_metrics,
)
from src.recognition.verification.run import EvaluationConfig, VerificationRunner
from src.training import Trainer, make_optimizer, make_scheduler, seed_everything
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
    raw["preprocess"].update(
        size=32,
        horizontal_flip=0,
        brightness=0,
        contrast=0,
        saturation=0,
        rotation_degrees=0,
        translate=0,
    )
    raw["training"].update(
        epochs=1,
        batch_size=2,
        num_workers=0,
        monitor="val_loss",
        mode="min",
        device=device,
        progress=False,
        amp_init_scale=1.0,
        checkpoint_tiebreakers=[],
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
            "--amp-dtype",
            "fp16",
            "--set",
            "preprocess.horizontal_flip=0.25",
        ]
    )
    assert cfg.training.lr == 0.005 and cfg.training.epochs == 2
    assert cfg.training.amp_dtype == "fp16"
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_augmentation_is_per_image_and_eval_matches_inference(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    cfg = tiny_config(tmp_path).preprocess.model_copy(update={"horizontal_flip": 0.5})
    transform = FaceBatchTransform(cfg).to(device)
    image = torch.zeros(1, 3, 32, 32, dtype=torch.uint8, device=device)
    image[:, 0, :, :16] = 255
    batch = image.expand(64, -1, -1, -1).clone()
    original = batch.clone()
    torch.manual_seed(9)
    output = transform(batch)
    assert output.device.type == device and output.dtype == torch.float32
    assert set(output[:, 0, 0, 0].tolist()) == {-1, 1}
    assert not torch.equal(output, transform(batch))
    assert torch.equal(batch, original)
    transform.eval()
    red = np.zeros((32, 32, 3), dtype=np.uint8)
    red[:, :, 2] = 255
    rgb = torch.from_numpy(red[:, :, ::-1].copy()).permute(2, 0, 1)
    expected = FaceTransform(cfg)(red)
    assert torch.equal(transform(rgb[None].to(device))[0].cpu(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_color_and_geometry_augmentation_is_fresh_and_bounded(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    cfg = tiny_config(tmp_path).preprocess.model_copy(
        update={
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "rotation_degrees": 10,
            "translate": 0.05,
        }
    )
    transform = FaceBatchTransform(cfg).to(device)
    images = torch.full((8, 3, 32, 32), 120, dtype=torch.uint8, device=device)
    images[:, 0, :, :16] = 240
    output = transform(images)
    assert torch.isfinite(output).all() and output.min() >= -1 and output.max() <= 1
    assert not torch.equal(output[0], output[1])
    assert not torch.equal(output, transform(images))
    transform.eval()
    assert torch.equal(transform(images), transform(images))


def test_cache_budget_keeps_training_reserve_and_fraction():
    assert cache_budget(10 * 2**30, 0.8, 3) == 7 * 2**30
    assert cache_budget(10 * 2**30, 0.5, 2) == 5 * 2**30
    assert cache_budget(2**30, 0.8, 2) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cuda_cache_contains_unaugmented_crops_and_streams_when_over_budget(
    tmp_path, monkeypatch
):
    cfg = tiny_config(tmp_path, device="cuda")
    train_loader, val_loader, _ = make_loaders(cfg)
    expected = next(iter(val_loader))[0]
    (cached_train, cached_val), report = cache_loaders(
        (train_loader, val_loader), cfg, torch.device("cuda")
    )
    assert report["mode"] == "cuda" and report["cached_images"] == 8
    assert report["required_bytes"] == 8 * 3 * 32**2
    assert cached_train.num_workers == 0 and not cached_val.pin_memory
    assert cached_val.dataset.cached_images.dtype == torch.uint8

    def no_decode(*args):
        raise AssertionError("cached iteration must not decode source images")

    with monkeypatch.context() as patch:
        patch.setattr(cv2, "imread", no_decode)
        images, _ = next(iter(cached_val))
        assert images.is_cuda and torch.equal(images.cpu(), expected)
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"cache_reserve_gib": 1_000_000})}
    )
    a, b, _ = make_loaders(cfg)
    (streaming, _), report = cache_loaders((a, b), cfg, torch.device("cuda"))
    assert report["reason"] == "insufficient_budget" and report["mode"] == "batch"
    assert next(iter(streaming))[0].device.type == "cpu"
    transform = FaceBatchTransform(cfg.preprocess).cuda()
    assert transform(next(iter(streaming))[0].cuda()).is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_cache_allocation_oom_releases_partial_cache(tmp_path, monkeypatch):
    import src.recognition.classification.cache as cache_module

    cfg = tiny_config(tmp_path, device="cuda")
    a, b, _ = make_loaders(cfg)

    def allocation_oom(*args):
        partial = torch.zeros(1024, device="cuda")
        assert partial.is_cuda
        raise torch.cuda.OutOfMemoryError("injected cache allocation failure")

    monkeypatch.setattr(cache_module, "_allocate_and_fill", allocation_oom)
    with pytest.warns(UserWarning, match="cache allocation failed"):
        loaders, report = cache_loaders((a, b), cfg, torch.device("cuda"))
    assert report["reason"] == "cache_cuda_oom"
    assert all(loader.dataset.cached_images is None for loader in loaders)
    assert next(iter(loaders[0]))[0].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
@pytest.mark.parametrize("completed_epochs", [0, 1])
def test_training_oom_retries_without_cache_and_preserves_weights(
    tmp_path, monkeypatch, completed_epochs
):
    cfg = tiny_config(tmp_path, device="cuda")
    cfg = cfg.model_copy(
        update={
            "preprocess": cfg.preprocess.model_copy(
                update={
                    "horizontal_flip": 0.5,
                    "brightness": 0.1,
                    "rotation_degrees": 5,
                    "translate": 0.05,
                }
            )
        }
    )
    cfg = cfg.model_copy(
        update={
            "training": cfg.training.model_copy(
                update={
                    "epochs": 2,
                    "scheduler": "none",
                    "patience": None,
                    "deterministic": True,
                }
            )
        }
    )
    streaming = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"cache": "none"})}
    )
    expected = train(streaming)
    original = Trainer._epoch
    injected = False

    def interrupted_epoch(self, loader, *, train):
        nonlocal injected
        result = original(self, loader, train=train)
        if train and len(self.history) == completed_epochs and not injected:
            injected = True
            raise torch.cuda.OutOfMemoryError("injected training failure")
        return result

    monkeypatch.setattr(Trainer, "_epoch", interrupted_epoch)
    with pytest.warns(UserWarning, match="Releasing the cache"):
        result = train(cfg)
    assert injected
    pipeline = json.loads((result.run_dir / "data_pipeline.json").read_text())
    assert pipeline["mode"] == "batch" and pipeline["reason"] == "training_cuda_oom"
    actual = torch.load(result.last_checkpoint, weights_only=True)
    baseline = torch.load(expected.last_checkpoint, weights_only=True)
    assert len(actual["history"]) == 2
    for key in baseline["task"]:
        assert torch.equal(actual["task"][key], baseline["task"][key]), key


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
    vectors = np.stack(
        [ColorEmbedder().embed(cv2.imread(str(p))) for p in evaluator.paths]
    )
    batch_metrics, _, _, _ = evaluator.evaluate_embeddings(vectors)
    assert batch_metrics == metrics
    with pytest.raises(ValueError, match="one embedding row"):
        evaluator.evaluate_embeddings(vectors[:-1])


def test_verification_reads_the_supplied_target_face_bbox(tmp_path):
    from src.datasets.catalog import write_csv

    config = verification_fixture(tmp_path)
    for path in tmp_path.glob("*.png"):
        face = cv2.imread(str(path))
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[16:48, 16:48] = face
        assert cv2.imwrite(str(path), frame)
    boxes = tmp_path / "boxes.csv"
    rows = [
        {
            "path": f"{name}.png",
            "x1": 16,
            "y1": 16,
            "x2": 48,
            "y2": 48,
            "width": 64,
            "height": 64,
        }
        for name in "abcd"
    ]
    write_csv(boxes, rows, list(rows[0]))
    config = config.model_copy(update={"bbox_manifest": boxes})
    evaluator = PairEvaluator(config, size=32, crop_scale=1, progress=False)
    metrics, _, _, _ = evaluator.evaluate(ColorEmbedder())
    assert metrics["cosine"]["auc"] == 1
    assert evaluator.read_face(tmp_path / "a.png")[0, 0].tolist() == [0, 0, 255]
    assert evaluator.bbox_manifest_sha256 is not None
    write_csv(boxes, rows[:-1], list(rows[0]))
    with pytest.raises(ValueError, match="cover every verification image"):
        PairEvaluator(config, size=32, crop_scale=1, progress=False)


def test_validation_threshold_reuse_rejects_a_changed_bbox_protocol(tmp_path):
    from src.datasets.catalog import write_csv

    verification = verification_fixture(tmp_path)
    boxes = tmp_path / "boxes.csv"
    rows = [
        {"path": f"{name}.png", "x1": 0, "y1": 0, "x2": 32, "y2": 32} for name in "abcd"
    ]
    write_csv(boxes, rows, list(rows[0]))
    verification = verification.model_copy(update={"bbox_manifest": boxes})
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
    pairs = tmp_path / "test.txt"
    pairs.write_text("a.png b.png 1\na.png d.png 0\n", encoding="utf-8")
    rows[0].update(x1=1, y1=1, x2=31, y2=31)
    write_csv(boxes, rows, list(rows[0]))
    test = cfg.model_copy(
        update={
            "split": "test",
            "calibration": output / "metrics.json",
            "verification": verification.model_copy(update={"pairs": pairs}),
        }
    )
    with pytest.raises(ValueError, match="preprocessing differs"):
        VerificationRunner(test).run()


@pytest.mark.parametrize(
    "device,cache", [("cpu", "auto"), ("cuda", "auto"), ("cuda", "none")]
)
def test_training_verification_batches_match_single_image_and_refresh_weights(
    tmp_path, monkeypatch, device, cache
):
    from types import SimpleNamespace

    from torch import nn
    from torch.nn import functional as F

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    cfg = tiny_config(tmp_path, device=device)
    pairs_dir = tmp_path / "verification"
    pairs_dir.mkdir()
    verification = verification_fixture(pairs_dir).model_copy(
        update={
            "batch_size": 3,
            "cache": cache,
        }
    )
    cfg = cfg.model_copy(update={"verification": verification})

    class ColorEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(3))
            self.calls = []

        def embed(self, images):
            assert images.dtype == torch.float32
            self.calls.append(len(images))
            return F.normalize(images.mean((2, 3)) * self.scale, dim=1)

    encoder = ColorEncoder().to(device).eval()
    task = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
    target = torch.device(device)
    evaluator = TrainingVerificationEvaluator(cfg)
    report = evaluator.prepare(target)
    assert report["mode"] == (
        "cuda" if device == "cuda" and cache == "auto" else "batch"
    )
    if report["mode"] == "cuda":
        assert report["cached_images"] == 4
        assert report["cached_bytes"] == 4 * 3 * 32**2
        assert evaluator.loader.num_workers == 0

        def no_decode(*args):
            raise AssertionError("verification cache must survive across epochs")

        with monkeypatch.context() as patch:
            patch.setattr(cv2, "imread", no_decode)
            first = evaluator(task, target)
            with torch.no_grad():
                encoder.scale[0] = 3
            second = evaluator(task, target)
    else:
        first = evaluator(task, target)
        with torch.no_grad():
            encoder.scale[0] = 3
        second = evaluator(task, target)
    assert encoder.calls == [3, 1, 3, 1]
    assert first["val_auc"] == second["val_auc"] == 1

    class Adapter:
        @torch.inference_mode()
        def embed(self, face):
            image = FaceTransform(cfg.preprocess)(face)[None].to(target)
            return encoder.embed(image)[0].cpu().numpy()

    metrics, _, _, _ = evaluator.pairs.evaluate(Adapter())
    assert second["val_score_threshold"] == pytest.approx(
        metrics["cosine"]["score_threshold"], abs=1e-6
    )
    assert first["val_cosine_positive_mean"] > first["val_cosine_negative_mean"]
    encoder.embed = lambda images: torch.ones(len(images), 3, device=target) / 3**0.5
    collapsed = evaluator(task, target)
    assert collapsed["val_embedding_mean_norm"] == pytest.approx(1, abs=1e-6)
    assert collapsed["val_cosine_positive_mean"] == pytest.approx(1, abs=1e-6)
    assert collapsed["val_cosine_negative_mean"] == pytest.approx(1, abs=1e-6)
    assert collapsed["val_cosine_negative_std"] == pytest.approx(0, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_verification_and_classification_share_the_total_cache_budget(
    tmp_path, monkeypatch
):
    cfg = tiny_config(tmp_path, device="cuda")
    pairs_dir = tmp_path / "verification"
    pairs_dir.mkdir()
    cfg = cfg.model_copy(
        update={
            "verification": verification_fixture(pairs_dir),
            "data": cfg.data.model_copy(update={"cache_reserve_gib": 0}),
        }
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40000, 40000))
    evaluator = TrainingVerificationEvaluator(cfg)
    report = evaluator.prepare(torch.device("cuda"))
    assert report["budget_bytes"] == 32000 and report["cached_bytes"] == 12288
    a, b, _ = make_loaders(cfg)
    _, classification = cache_loaders(
        (a, b),
        cfg,
        torch.device("cuda"),
        budget_bytes=report["budget_bytes"] - report["cached_bytes"],
    )
    assert classification["budget_bytes"] == 19712
    assert classification["reason"] == "insufficient_budget"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_training_oom_fallback_also_releases_a_verification_only_cache(
    tmp_path, monkeypatch
):
    cfg = tiny_config(tmp_path, device="cuda")
    pairs_dir = tmp_path / "verification"
    pairs_dir.mkdir()
    cfg = cfg.model_copy(
        update={
            "verification": verification_fixture(pairs_dir),
            "data": cfg.data.model_copy(update={"cache": "none"}),
        }
    )
    original = Trainer._epoch
    interrupted = False

    def once(self, loader, *, train):
        nonlocal interrupted
        metrics = original(self, loader, train=train)
        if train and not interrupted:
            interrupted = True
            raise torch.cuda.OutOfMemoryError("injected verification-cache pressure")
        return metrics

    monkeypatch.setattr(Trainer, "_epoch", once)
    with pytest.warns(UserWarning, match="Releasing the cache"):
        result = train(cfg)
    report = json.loads((result.run_dir / "data_pipeline.json").read_text())
    assert report["mode"] == report["verification"]["mode"] == "batch"
    assert report["verification"]["cached_images"] == 0
    assert result.history[0]["train_optimizer_steps"] > 0


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


def c_model_config(**updates):
    raw = read_yaml(REPO / "configs/recognition/model_c.yaml")["model"]
    raw.update(updates)
    return ModelConfig.model_validate(raw)


def test_margin_warmup_has_exact_hold_ramp_and_final_values():
    config = c_model_config(
        arcface_margin_warmup={"enabled": True, "hold_epochs": 3, "ramp_epochs": 5}
    )
    schedule = config.arcface_margin_warmup
    assert [schedule.value(epoch, 0.3) for epoch in range(1, 11)] == pytest.approx(
        [0, 0, 0, 0.06, 0.12, 0.18, 0.24, 0.3, 0.3, 0.3]
    )
    disabled = c_model_config().arcface_margin_warmup
    assert disabled.value(1, 0.3) == 0.3
    with pytest.raises(ValueError, match="ramp_epochs"):
        c_model_config(arcface_margin_warmup={"enabled": True, "ramp_epochs": 0})
    with pytest.raises(ValueError, match="cannot exceed"):
        c_model_config(arcface_margin_warmup={"enabled": True, "start_margin": 0.4})
    with pytest.raises(ValueError, match="require Model C"):
        c_model_config(variant="a", arcface_gradient="straight_through")


def test_straight_through_preserves_forward_and_changes_only_margin_gradient():
    analytic = ArcFaceHead(2, 2, scale=30, margin=0.3)
    surrogate = ArcFaceHead(
        2, 2, scale=30, margin=0.3, gradient_policy="straight_through"
    )
    with torch.no_grad():
        analytic.weight.copy_(torch.eye(2))
        surrogate.weight.copy_(analytic.weight)
    x = torch.tensor([[0.8, 0.6]], requires_grad=True)
    labels = torch.tensor([0])
    logits = analytic(x, labels)
    adjusted = surrogate(x, labels)
    torch.testing.assert_close(adjusted, logits)
    actual = torch.autograd.grad(F.cross_entropy(adjusted, labels), x)[0]
    unadjusted = surrogate(x)
    coefficients = (adjusted.softmax(1) - F.one_hot(labels, 2)).detach()
    expected = torch.autograd.grad((unadjusted * coefficients).sum(), x)[0]
    torch.testing.assert_close(actual, expected)
    analytical = torch.autograd.grad(F.cross_entropy(logits, labels), x)[0]
    assert not torch.allclose(actual, analytical)
    analytic.margin = 0
    torch.testing.assert_close(analytic(x, labels), analytic(x))


def test_wrong_class_rectification_preserves_target_and_inference():
    plain = ArcFaceHead(2, 2, scale=30, margin=0.3)
    rectified = ArcFaceHead(2, 2, scale=30, margin=0.3, rectify_wrong_class_logits=True)
    with torch.no_grad():
        plain.weight.copy_(torch.eye(2))
        rectified.weight.copy_(plain.weight)
    x, labels = torch.tensor([[0.8, -0.6]]), torch.tensor([0])
    expected, actual = plain(x, labels), rectified(x, labels)
    assert actual[0, 0] == expected[0, 0]
    assert expected[0, 1].item() == pytest.approx(-18)
    assert actual[0, 1] == 0
    torch.testing.assert_close(rectified(x), plain(x))


def test_prototype_regularization_is_added_and_reported_separately():
    config = c_model_config(embedding_dim=2, prototype_regularization_weight=0.5)
    model = ClassificationModel(config, 2)
    model.encoder = nn.Identity()
    task = RecognitionTask(model)
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1.0, 0.0], [0.6, 0.8]]))
    images, labels = torch.eye(2), torch.tensor([0, 1])
    result = task.step((images, labels))
    # Mean of the two unit prototypes is (0.8, 0.4).
    expected_penalty = 0.5 * (0.8**2 + 0.4**2) ** 0.5
    classification = result.metrics["classification_loss"].total / 2
    assert result.loss.item() == pytest.approx(classification + expected_penalty)
    assert result.metrics["prototype_mean_norm"].total / 2 == pytest.approx(0.8**0.5)
    result.loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()


def test_margin_resume_restores_weights_bn_and_epoch_metrics_exactly(tmp_path):
    model_config = c_model_config(
        embedding_dim=4,
        embedding_normalization="batchnorm",
        arcface_gradient="straight_through",
        prototype_regularization_weight=0.1,
        arcface_margin_warmup={"enabled": True, "hold_epochs": 1, "ramp_epochs": 2},
    )
    base = tiny_config(tmp_path, "c")
    cfg = base.training.model_copy(
        update={
            "epochs": 6,
            "scheduler": "cosine",
            "parameter_groups": (),
            "deterministic": True,
            "amp": False,
            "monitor": "val_constant",
            "mode": "min",
            "patience": 1,
        }
    )

    def run(directory, *, resume=None, interrupt=False):
        seed_everything(17, True)
        model = ClassificationModel(model_config, 2)
        model.encoder.backbone = nn.Linear(3, 512)
        task = RecognitionTask(model)
        optimizer = make_optimizer(task, cfg)
        dataset = TensorDataset(
            torch.arange(24).float().reshape(8, 3) / 10, torch.tensor([0, 1] * 4)
        )
        loaders = tuple(
            DataLoader(
                dataset,
                batch_size=4,
                shuffle=train,
                generator=torch.Generator().manual_seed(11),
            )
            for train in (True, False)
        )
        calls = 0

        def evaluator(task, device):
            nonlocal calls
            calls += 1
            if interrupt and calls == 3:
                raise RuntimeError("interrupted after epoch two")
            return {"val_constant": 1.0}

        trainer = Trainer(
            task,
            optimizer,
            cfg,
            directory,
            scheduler=make_scheduler(optimizer, cfg),
            metadata={"model_config": model_config.model_dump(mode="json")},
            evaluator=evaluator,
        )
        return trainer.fit(*loaders, resume=resume)

    full = run(tmp_path / "full-margin")
    assert [row["arcface_margin"] for row in full.history] == pytest.approx(
        [0, 0.15, 0.3, 0.3]
    )
    # Patience 1 does not stop the run before the margin reaches its target.
    assert len(full.history) == 4
    interrupted = tmp_path / "interrupted-margin"
    with pytest.raises(RuntimeError, match="after epoch two"):
        run(interrupted, interrupt=True)
    resumed = run(tmp_path / "resumed-margin", resume=interrupted / "last.pt")
    assert resumed.history == full.history
    expected = torch.load(full.last_checkpoint, weights_only=True)
    actual = torch.load(resumed.last_checkpoint, weights_only=True)
    for key, value in expected["task"].items():
        torch.testing.assert_close(actual["task"][key], value, rtol=0, atol=0)
    assert actual["bad_epochs"] == expected["bad_epochs"] == 1


def test_loading_checkpoint_restores_its_margin_and_bn_design(tmp_path):
    cfg = c_model_config(
        embedding_normalization="batchnorm",
        arcface_margin_warmup={"enabled": True, "hold_epochs": 1, "ramp_epochs": 2},
    )
    task = RecognitionTask(ClassificationModel(cfg, 2))
    checkpoint = tmp_path / "scheduled.pt"
    torch.save(
        {
            "epoch": 2,
            "task": task.state_dict(),
            "metadata": {
                "task": "recognition_classification",
                "model_config": cfg.model_dump(mode="json", exclude_defaults=True),
                "class_to_idx": {"alice": 0, "bob": 1},
            },
        },
        checkpoint,
    )
    loaded, _ = load_task(checkpoint, torch.device("cpu"))
    assert loaded.model.head.margin == pytest.approx(0.15)
    assert isinstance(loaded.model.encoder.embedding_norm, nn.BatchNorm1d)
    assert not loaded.model.encoder.embedding_norm.training


def test_default_objective_metadata_matches_legacy_model_c_schema():
    metadata = c_model_config().model_dump(mode="json", exclude_defaults=True)
    assert metadata == {
        "variant": "c",
        "backbone": "resnet18",
        "embedding_dim": 128,
        "initialization": "random",
        "arcface_scale": 30.0,
        "arcface_margin": 0.3,
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_recovery_controls_train_and_reload_real_encoder(tmp_path, device):
    if device == "cuda" and (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported(including_emulation=False)
    ):
        pytest.skip("Native BF16 GPU unavailable")
    cfg = tiny_config(tmp_path, "c", device)
    model = c_model_config(
        embedding_normalization="batchnorm",
        arcface_gradient="straight_through",
        prototype_regularization_weight=0.25,
        rectify_wrong_class_logits=True,
        arcface_margin_warmup={"enabled": True, "hold_epochs": 0, "ramp_epochs": 2},
    )
    cfg = cfg.model_copy(update={"model": model})
    seed_everything(cfg.training.seed)
    initial = ClassificationModel(model, 2).head.weight.detach().clone()
    result = train(cfg)
    loaded, payload = load_task(result.best_checkpoint, torch.device(device))
    assert loaded.model.head.margin == pytest.approx(0.15)
    assert not torch.equal(loaded.model.head.weight.cpu(), initial)
    assert not loaded.model.encoder.embedding_norm.training
    assert payload["task"]["model.encoder.embedding_norm.num_batches_tracked"] == 2
    row = result.history[0]
    assert row["arcface_margin"] == pytest.approx(0.15)
    assert row["val_loss"] == pytest.approx(
        row["val_classification_loss"] + 0.25 * row["val_prototype_mean_norm"]
    )
    assert all(torch.isfinite(value).all() for value in payload["task"].values())
