"""Fast integration tests for resume and artifact contracts using a tiny encoder."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
from torch import nn

from src.recognition.config import ExperimentConfig
from src.recognition.data import prepare_manifests
from src.recognition.evaluation import subject_bootstrap_ci
from src.recognition import training
from src.recognition import notebook_utils


class TinyEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(3, embedding_dim)

    def forward(self, images):
        return nn.functional.normalize(self.fc(self.dropout(images.mean((2, 3)))), dim=1)


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
    prepared = prepare_manifests(root, tmp_path / "manifests", dev_identities=2,
                                 genuine_per_identity=3, impostor_per_identity_pair=4, probe_per_identity=2)
    config = ExperimentConfig(data_root=root, artifacts_root=tmp_path / "baseline", device="cpu",
                              amp=False, num_workers=0, epochs=2, warmup_epochs=0, minimum_epochs=2,
                              confirmation_seeds=(), max_batches_per_epoch=1, evaluation_batch_size=8,
                              identities_per_batch=2, images_per_identity=2, cpu_threads=2,
                              gate_minimum_epochs=1, gate_consecutive_checks=1,
                              gate_ranking_rate=0, gate_mean_gap=-4, gate_feature_spread=-1)
    monkeypatch.setattr(training, "IResNet34", TinyEncoder)
    return config, prepared


def test_resume_replays_from_complete_epoch_with_matching_rng(experiment, monkeypatch):
    config, prepared = experiment
    complete = training.train_run(config, prepared, margin=0.3)
    baseline = torch.load(Path(complete["best_checkpoint"]).with_name("last.pt"), weights_only=False)
    resumed_config = replace(config, artifacts_root=Path(config.artifacts_root).parent / "resumed")
    original_save = training._save_checkpoint

    def interrupt_after_epoch(path, state):
        original_save(path, state)
        if path.name == "last.pt" and state["epoch"] == 1:
            raise InterruptedError("Simulated interruption after a complete epoch.")

    with monkeypatch.context() as context:
        context.setattr(training, "_save_checkpoint", interrupt_after_epoch)
        with pytest.raises(InterruptedError):
            training.train_run(resumed_config, prepared, margin=0.3)
    result = training.train_run(resumed_config, prepared, margin=0.3)
    resumed = torch.load(Path(result["best_checkpoint"]).with_name("last.pt"), weights_only=False)
    for key, value in baseline["model"].items():
        torch.testing.assert_close(value, resumed["model"][key], rtol=0, atol=0)
    assert baseline["initial_encoder_sha256"] == resumed["initial_encoder_sha256"]
    assert resumed["history"][1]["strategy"] == "semi_hard"
    assert len(resumed["history"]) == 2
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
    export = torch.load(run.embedding_checkpoint, weights_only=False)
    assert "model" in export and "head" not in export
    assert export["embedding_dim"] == 256
    assert export["environment_compatibility"] == training.environment_compatibility()
    # Reload the saved table to exercise pandas 3 CSV types and runtime guards.
    results = training.evaluate_selected(config, prepared, pd.read_csv(config.output_dir / "sweep_val.csv"))
    assert results.iloc[0].threshold_from_dev == run.threshold
    assert (config.output_dir / "test/seed_42.json").is_file()
    assert (config.output_dir / "test_results.csv").is_file()
    assert (config.output_dir / "test_summary.csv").is_file()


@pytest.mark.parametrize("artifact_kind", ["completed", "checkpoint"])
def test_changed_environment_cannot_reuse_or_resume_runs(experiment, monkeypatch, artifact_kind):
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
    training.write_json(run_dir / "result.json", {
        "signature": training._fingerprint(config, prepared, margin=0.3, seed=42),
    })
    with pytest.raises(ValueError, match="compatibility metadata is missing"):
        training.train_run(config, prepared, margin=0.3)


def test_fingerprint_tracks_runtime_but_ignores_resume_toggle(experiment, monkeypatch):
    config, prepared = experiment
    expected = training._fingerprint(config, prepared, margin=0.3, seed=42)
    assert expected == training._fingerprint(replace(config, resume=False), prepared, 0.3, 42)
    changed = {**training.environment_compatibility(), "numpy": "different-runtime"}
    monkeypatch.setattr(training, "environment_compatibility", lambda: changed)
    assert expected != training._fingerprint(config, prepared, margin=0.3, seed=42)


@pytest.mark.parametrize("legacy_profile", [False, True])
def test_runtime_pilot_does_not_overwrite_incompatible_profile(experiment, legacy_profile):
    config, prepared = experiment
    profile = Path(config.artifacts_root) / "shared" / "runtime_pilot.json"
    previous = {"P": 8, "K": 4}
    if not legacy_profile:
        previous["environment_compatibility"] = {
            **training.environment_compatibility(), "python": "different-runtime",
        }
    training.write_json(profile, previous)
    with pytest.raises(ValueError, match="Runtime pilot environment changed"):
        notebook_utils.pilot_runtime(config, prepared)
    assert json.loads(profile.read_text(encoding="utf-8")) == previous


def test_subject_bootstrap_retains_fixed_threshold_and_both_identity_endpoints():
    embeddings = {"a": np.array([1.0, 0.0]), "aa": np.array([1.0, 0.0]),
                  "b": np.array([0.0, 1.0]), "bb": np.array([0.0, 1.0]),
                  "c": np.array([-1.0, 0.0]), "cc": np.array([-1.0, 0.0])}
    pairs = pd.DataFrame([
        ("a", "aa", True, "A", "A"), ("b", "bb", True, "B", "B"),
        ("c", "cc", True, "C", "C"), ("a", "b", False, "A", "B"),
        ("b", "c", False, "B", "C"), ("a", "c", False, "A", "C"),
    ], columns=["path_a", "path_b", "is_same", "identity_a", "identity_b"])
    ci = subject_bootstrap_ci(embeddings, pairs, threshold=0.5, n_bootstrap=100)
    assert ci["fixed_threshold"] == 0.5 and ci["identity_count"] == 3
    assert ci["tar"] == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}
    assert ci["fmr"] == {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    assert ci == subject_bootstrap_ci(embeddings, pairs, threshold=0.5, n_bootstrap=100)
