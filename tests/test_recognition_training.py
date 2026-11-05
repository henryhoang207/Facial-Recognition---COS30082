"""Fast integration tests for resume and artifact contracts using a tiny encoder."""

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.recognition.identification import notebook_utils, training
from src.recognition.identification.config import ExperimentConfig
from src.recognition.identification.data import prepare_manifests
from src.recognition.identification.embedding_evaluation import subject_bootstrap_ci
from src.recognition.identification.task import MetricLearningTask
from src.training import Trainer, TrainingConfig

training_engine = import_module("src.training.trainer")


def assert_state_equal(left, right):
    """Compare nested checkpoint state without weakening tensor equality."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key, value in left.items():
            assert_state_equal(value, right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_state_equal(first, second)
    else:
        assert left == right


def load_checkpoint(path):
    """Metric checkpoints must use the shared weights-only-safe schema."""
    return torch.load(path, map_location="cpu", weights_only=True)


class TinyEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(3, embedding_dim)

    def forward(self, images):
        return nn.functional.normalize(
            self.fc(self.dropout(images.mean((2, 3)))), dim=1
        )


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "faces"
    rng = np.random.default_rng(99)
    for split, identities, offset in [("train", 10, 0), ("val", 3, 20)]:
        for identity in range(identities):
            folder = root / split / f"id{identity+offset:03d}"
            folder.mkdir(parents=True)
            for image_index in range(4):
                pixels = rng.integers(0, 256, (16, 20, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{image_index}.png")
    prepared = prepare_manifests(
        root,
        tmp_path / "manifests",
        dev_identities=2,
        genuine_per_identity=3,
        impostor_per_identity_pair=4,
        probe_per_identity=2,
    )
    config = ExperimentConfig(
        data_root=root,
        artifacts_root=tmp_path / "baseline",
        device="cpu",
        amp=False,
        num_workers=0,
        epochs=2,
        warmup_epochs=0,
        minimum_epochs=2,
        confirmation_seeds=(),
        max_batches_per_epoch=1,
        evaluation_batch_size=8,
        identities_per_batch=2,
        images_per_identity=2,
        cpu_threads=2,
        gate_minimum_epochs=1,
        gate_consecutive_checks=1,
        gate_ranking_rate=0,
        gate_mean_gap=-4,
        gate_feature_spread=-1,
    )
    monkeypatch.setattr(training, "IResNet34", TinyEncoder)
    return config, prepared


@pytest.mark.parametrize("method", ["triplet", "arcface"])
def test_resume_replays_from_complete_epoch_with_matching_rng(
    experiment, monkeypatch, method
):
    config, prepared = experiment
    config = replace(config, method=method)
    complete = training.train_run(config, prepared, margin=0.3)
    baseline = load_checkpoint(Path(complete["best_checkpoint"]).with_name("last.pt"))
    resumed_config = replace(
        config, artifacts_root=Path(config.artifacts_root).parent / "resumed"
    )
    original_save = training_engine._atomic_save

    def interrupt_after_epoch(state, path):
        original_save(state, path)
        if path.name == "last.pt" and state["epoch"] == 1:
            raise InterruptedError("Simulated interruption after a complete epoch.")

    with monkeypatch.context() as context:
        context.setattr(training_engine, "_atomic_save", interrupt_after_epoch)
        with pytest.raises(InterruptedError):
            training.train_run(resumed_config, prepared, margin=0.3)
    result = training.train_run(resumed_config, prepared, margin=0.3)
    resumed = load_checkpoint(Path(result["best_checkpoint"]).with_name("last.pt"))
    assert_state_equal(baseline["task"], resumed["task"])
    assert_state_equal(baseline["optimizer"], resumed["optimizer"])
    assert_state_equal(baseline["rng"], resumed["rng"])
    assert_state_equal(baseline["loader_rng"], resumed["loader_rng"])
    assert (
        baseline["metadata"]["initial_encoder_sha256"]
        == resumed["metadata"]["initial_encoder_sha256"]
    )
    assert resumed["history"][1]["strategy"] == (
        "semi_hard" if method == "triplet" else "arcface"
    )
    if method == "triplet":
        assert resumed["task"]["_extra_state"]["strategy"] == "semi_hard"
    assert len(resumed["history"]) == 2
    for expected_row, actual_row in zip(
        baseline["history"], resumed["history"], strict=True
    ):
        assert {
            key: value for key, value in expected_row.items() if key != "epoch_seconds"
        } == {key: value for key, value in actual_row.items() if key != "epoch_seconds"}
    stamp = Path(result["best_checkpoint"]).stat().st_mtime_ns
    assert training.train_run(resumed_config, prepared, 0.3) == result
    assert Path(result["best_checkpoint"]).stat().st_mtime_ns == stamp
    with pytest.raises(ValueError, match="configuration changed"):
        training.train_run(replace(resumed_config, learning_rate=1e-4), prepared, 0.3)


def test_arcface_sweep_and_frozen_threshold_final_artifacts(experiment):
    config, prepared = experiment
    config = replace(config, method="arcface", epochs=1, minimum_epochs=1)
    sweep = training.run_sweep(config, prepared, margins=(0.3,), confirmation_seeds=())
    assert (config.output_dir / "sweep_val.csv").is_file()
    assert (config.output_dir / "graphs/sweep_val.png").is_file()
    run = sweep.iloc[0]
    export = load_checkpoint(run.embedding_checkpoint)
    assert export["format_version"] == 1
    assert export["checkpoint_kind"] == "inference"
    assert export["task"] and all(key.startswith("model.") for key in export["task"])
    assert export["metadata"]["embedding_dim"] == 256
    assert (
        export["metadata"]["environment_compatibility"]
        == training.environment_compatibility()
    )
    # Reload the saved table to exercise pandas 3 CSV types and runtime guards.
    results = training.evaluate_selected(
        config, prepared, pd.read_csv(config.output_dir / "sweep_val.csv")
    )
    assert results.iloc[0].threshold_from_dev == run.threshold
    assert (config.output_dir / "test/seed_42.json").is_file()
    assert (config.output_dir / "test_results.csv").is_file()
    assert (config.output_dir / "test_summary.csv").is_file()


@pytest.mark.parametrize("method", ["triplet", "arcface"])
def test_metric_runs_use_shared_training_and_inference_checkpoint_schema(
    experiment, method
):
    config, prepared = experiment
    config = replace(config, method=method, epochs=1, minimum_epochs=1)
    result = training.train_run(config, prepared, margin=0.3)
    best = load_checkpoint(result["best_checkpoint"])
    last = load_checkpoint(Path(result["best_checkpoint"]).with_name("last.pt"))
    export = load_checkpoint(result["embedding_checkpoint"])
    required = {
        "format_version",
        "checkpoint_kind",
        "task",
        "epoch",
        "best_epoch",
        "best",
        "training_config",
        "metadata",
    }
    assert required <= best.keys() and required <= last.keys()
    assert best["format_version"] == last["format_version"] == 1
    assert best["checkpoint_kind"] == export["checkpoint_kind"] == "inference"
    assert last["checkpoint_kind"] == "training"
    assert {
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "loader_rng",
        "best_task",
        "history",
    } <= last.keys()
    assert "optimizer" not in best and "optimizer" not in export
    assert "model" not in last and "head" not in last
    assert best["metadata"]["task"] == "recognition_metric_learning"
    assert best["metadata"]["experiment_config"] == {
        key: value for key, value in config.to_dict().items() if key != "resume"
    }
    assert (
        best["metadata"]["initial_encoder_sha256"] == result["initial_encoder_sha256"]
    )
    assert last["task"]["_extra_state"]["strategy"] in {
        "random",
        "semi_hard",
        "arcface",
    }
    assert any(key.startswith("head.") for key in last["task"]) == (method == "arcface")
    restored = TinyEncoder().eval()
    restored.load_state_dict(
        {key.removeprefix("model."): value for key, value in export["task"].items()}
    )
    embeddings = restored(torch.zeros(2, 3, 16, 20))
    assert embeddings.shape == (2, 256)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(2))


@pytest.mark.parametrize("strategy", ["random", "semi_hard"])
def test_zero_signal_triplets_never_update_adamw_parameters_or_state(
    experiment, strategy
):
    config, _ = experiment

    class SeparatedEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            values = torch.zeros(4, 256)
            values[:2, 0], values[2:, 1] = 1, 1
            self.features = nn.Parameter(values)

        def forward(self, images):
            return self.features

    task = MetricLearningTask(
        SeparatedEncoder(), config, class_count=2, margin=0.3, seed=42
    )
    task.strategy = strategy
    batch = {"image": torch.zeros(4, 3, 2, 2), "label": torch.tensor([0, 0, 1, 1])}
    step = task.step(batch)
    assert step.loss.item() == 0 and not step.optimize
    assert step.metrics["selected_triplets"].total == (4 if strategy == "random" else 0)
    before = task.model.features.detach().clone()
    optimizer = torch.optim.AdamW(task.parameters(), lr=0.1, weight_decay=0.5)
    shared_config = TrainingConfig(
        epochs=1,
        batch_size=4,
        lr=0.1,
        weight_decay=0.5,
        optimizer="adamw",
        scheduler="none",
        seed=42,
        device="cpu",
        amp=False,
        deterministic=True,
        num_workers=0,
        monitor="train_loss",
        mode="min",
        progress=False,
    )
    loader = DataLoader(
        [{key: value[index] for key, value in batch.items()} for index in range(4)],
        batch_size=4,
    )
    fit = Trainer(
        task,
        optimizer,
        shared_config,
        Path(config.artifacts_root) / strategy,
        evaluator=lambda task, device: {"dev_tar": 1.0},
    ).fit(loader, None)
    assert fit.history[0]["train_optimizer_steps"] == 0
    assert not optimizer.state
    torch.testing.assert_close(task.model.features, before, rtol=0, atol=0)


def test_legacy_metric_checkpoint_cannot_be_silently_resumed(experiment):
    config, prepared = experiment
    run_dir = config.output_dir / "runs" / "margin_0.3_seed_42"
    last = run_dir / "weights" / "last.pt"
    last.parent.mkdir(parents=True)
    torch.save(
        {
            "model": TinyEncoder().state_dict(),
            "epoch": 1,
            "signature": training._fingerprint(config, prepared, 0.3, 42),
            "environment_compatibility": training.environment_compatibility(),
        },
        last,
    )
    previous = last.read_bytes()
    with pytest.raises(ValueError, match="(?i)legacy|schema|format|supported|contract"):
        training.train_run(config, prepared, margin=0.3)
    assert last.read_bytes() == previous


def test_auc_tie_break_saves_best_without_resetting_tar_patience(
    experiment, monkeypatch
):
    config, prepared = experiment
    config = replace(config, method="arcface", epochs=5, minimum_epochs=1, patience=2)
    scores = iter([(0.5, 0.7), (0.5, 0.8), (0.5, 0.8)])

    def metrics(*args, **kwargs):
        tar, auc = next(scores)
        return {"tar_at_fmr": tar, "auc": auc, "eer": 0.2, "threshold": 0.5, "fmr": 0.0}

    monkeypatch.setattr(training, "verification_metrics", metrics)
    result = training.train_run(config, prepared, margin=0.3)
    last = load_checkpoint(Path(result["best_checkpoint"]).with_name("last.pt"))
    best = load_checkpoint(result["best_checkpoint"])
    assert result["best_epoch"] == best["epoch"] == 2
    assert result["epochs_completed"] == last["epoch"] == 3
    assert last["bad_epochs"] == 2
    assert best["val_metrics"]["dev_auc"] == 0.8


def test_resume_finalizes_an_interrupted_early_stop_without_another_epoch(
    experiment, monkeypatch
):
    config, prepared = experiment
    config = replace(config, method="arcface", epochs=5, minimum_epochs=1, patience=1)
    calls = []

    def metrics(*args, **kwargs):
        calls.append(1)
        return {"tar_at_fmr": 0.5, "auc": 0.7, "eer": 0.2, "threshold": 0.5, "fmr": 0.0}

    monkeypatch.setattr(training, "verification_metrics", metrics)
    original_save = training_engine._atomic_save

    def interrupt_after_stop(state, path):
        original_save(state, path)
        if path.name == "last.pt" and state["epoch"] == 2:
            raise InterruptedError(
                "Simulated interruption after the early-stop checkpoint."
            )

    with monkeypatch.context() as context:
        context.setattr(training_engine, "_atomic_save", interrupt_after_stop)
        with pytest.raises(InterruptedError):
            training.train_run(config, prepared, margin=0.3)
    assert len(calls) == 2
    # The saved last.pt must reconstruct best.pt as well as the stopping state.
    for previous_best in config.output_dir.glob("runs/*/trainer_runs/*/best.pt"):
        previous_best.unlink()
    result = training.train_run(config, prepared, margin=0.3)
    assert len(calls) == 2
    assert result["epochs_completed"] == 2 and result["best_epoch"] == 1
    assert Path(result["best_checkpoint"]).is_file()


def test_early_stopping_waits_for_minimum_semi_hard_epochs(experiment, monkeypatch):
    config, prepared = experiment
    config = replace(
        config, epochs=6, minimum_epochs=1, patience=1, minimum_semi_hard_epochs=3
    )
    monkeypatch.setattr(
        training,
        "verification_metrics",
        lambda *args, **kwargs: {
            "tar_at_fmr": 0.5,
            "auc": 0.7,
            "eer": 0.2,
            "threshold": 0.5,
            "fmr": 0.0,
        },
    )
    result = training.train_run(config, prepared, margin=0.3)
    last = load_checkpoint(Path(result["best_checkpoint"]).with_name("last.pt"))
    assert result["switch_epoch"] == 2
    assert result["epochs_completed"] == 4
    assert [row["strategy"] for row in last["history"]] == [
        "random",
        "semi_hard",
        "semi_hard",
        "semi_hard",
    ]
    assert result["best_checkpoint_strategy"] == "random"
    assert not result["best_checkpoint_has_semi_hard_updates"]


def test_run_with_no_optimizer_updates_is_never_accepted(experiment, monkeypatch):
    config, prepared = experiment
    config = replace(config, epochs=1, minimum_epochs=1)

    def empty_loss(self, embeddings, labels, **kwargs):
        return embeddings.sum() * 0, {
            "positive_pairs": 4,
            "selected_triplets": 0,
            "active_triplets": 0,
            "empty_batch": 1,
            "zero_loss_batch": 1,
            "mean_positive_distance": 0.0,
            "mean_negative_distance": 0.0,
            "candidate_coverage": 0.0,
        }

    monkeypatch.setattr(
        import_module("src.recognition.identification.task").TripletMetricLoss,
        "forward",
        empty_loss,
    )
    with pytest.raises(RuntimeError, match="No optimizer updates completed"):
        training.train_run(config, prepared, margin=0.3)
    run_dir = config.output_dir / "runs" / "margin_0.3_seed_42"
    assert not (run_dir / "result.json").exists()
    assert not (run_dir / "weights/embedding.pt").exists()
    session_last = next(run_dir.glob("trainer_runs/*/last.pt"))
    saved = load_checkpoint(session_last)
    assert not saved["optimizer"]["state"]
    assert saved["history"][0]["optimizer_updates"] == 0


@pytest.mark.parametrize("artifact_kind", ["completed", "checkpoint"])
def test_changed_environment_cannot_reuse_or_resume_runs(
    experiment, monkeypatch, artifact_kind
):
    config, prepared = experiment
    config = replace(config, method="arcface", epochs=1, minimum_epochs=1)
    result = training.train_run(config, prepared, margin=0.3)
    run_dir = Path(result["best_checkpoint"]).parent.parent
    last = run_dir / "weights" / "last.pt"
    original_checkpoint = last.read_bytes()
    if artifact_kind == "checkpoint":
        (run_dir / "result.json").unlink()
    changed = {**training.environment_compatibility(), "torch": "different-runtime"}
    monkeypatch.setattr(training, "environment_compatibility", lambda: changed)
    with pytest.raises(ValueError, match="Run environment changed"):
        training.train_run(config, prepared, margin=0.3)
    assert last.read_bytes() == original_checkpoint


def test_legacy_result_without_environment_metadata_is_rejected(experiment):
    config, prepared = experiment
    run_dir = config.output_dir / "runs" / "margin_0.3_seed_42"
    run_dir.mkdir(parents=True)
    training.write_json(
        run_dir / "result.json",
        {
            "signature": training._fingerprint(config, prepared, margin=0.3, seed=42),
        },
    )
    with pytest.raises(ValueError, match="compatibility metadata is missing"):
        training.train_run(config, prepared, margin=0.3)


def test_fingerprint_tracks_runtime_but_ignores_resume_toggle(experiment, monkeypatch):
    config, prepared = experiment
    expected = training._fingerprint(config, prepared, margin=0.3, seed=42)
    assert expected == training._fingerprint(
        replace(config, resume=False), prepared, 0.3, 42
    )
    changed = {**training.environment_compatibility(), "numpy": "different-runtime"}
    monkeypatch.setattr(training, "environment_compatibility", lambda: changed)
    assert expected != training._fingerprint(config, prepared, margin=0.3, seed=42)


def test_resume_toggle_is_not_an_incompatible_checkpoint_setting(experiment):
    config, prepared = experiment
    config = replace(config, method="arcface", epochs=1, minimum_epochs=1, resume=False)
    first = training.train_run(config, prepared, margin=0.3)
    last_path = Path(first["best_checkpoint"]).with_name("last.pt")
    original = load_checkpoint(last_path)
    (last_path.parent.parent / "result.json").unlink()
    resumed = training.train_run(replace(config, resume=True), prepared, margin=0.3)
    final = load_checkpoint(Path(resumed["best_checkpoint"]).with_name("last.pt"))
    assert resumed["epochs_completed"] == 1
    assert resumed["total_optimizer_updates"] == first["total_optimizer_updates"]
    assert_state_equal(original["task"], final["task"])
    assert_state_equal(original["optimizer"], final["optimizer"])


@pytest.mark.parametrize("legacy_profile", [False, True])
def test_runtime_pilot_does_not_overwrite_incompatible_profile(
    experiment, legacy_profile
):
    config, prepared = experiment
    profile = Path(config.artifacts_root) / "shared" / "runtime_pilot.json"
    previous = {"P": 8, "K": 4}
    if not legacy_profile:
        previous["environment_compatibility"] = {
            **training.environment_compatibility(),
            "python": "different-runtime",
        }
    training.write_json(profile, previous)
    with pytest.raises(ValueError, match="Runtime pilot environment changed"):
        notebook_utils.pilot_runtime(config, prepared)
    assert json.loads(profile.read_text(encoding="utf-8")) == previous


def test_subject_bootstrap_retains_fixed_threshold_and_both_identity_endpoints():
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "aa": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),
        "bb": np.array([0.0, 1.0]),
        "c": np.array([-1.0, 0.0]),
        "cc": np.array([-1.0, 0.0]),
    }
    pairs = pd.DataFrame(
        [
            ("a", "aa", True, "A", "A"),
            ("b", "bb", True, "B", "B"),
            ("c", "cc", True, "C", "C"),
            ("a", "b", False, "A", "B"),
            ("b", "c", False, "B", "C"),
            ("a", "c", False, "A", "C"),
        ],
        columns=["path_a", "path_b", "is_same", "identity_a", "identity_b"],
    )
    ci = subject_bootstrap_ci(embeddings, pairs, threshold=0.5, n_bootstrap=100)
    assert ci["fixed_threshold"] == 0.5 and ci["identity_count"] == 3
    assert ci["tar"] == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}
    assert ci["fmr"] == {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    assert ci == subject_bootstrap_ci(embeddings, pairs, threshold=0.5, n_bootstrap=100)
