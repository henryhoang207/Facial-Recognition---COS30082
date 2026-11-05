"""Scientific invariants for verification calibration and open-set evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.recognition.identification.embedding_evaluation import (
    create_open_set_manifest,
    evaluate_identification,
    evaluate_probe,
    extract_embeddings,
    plot_sweep,
    save_training_graph,
    save_verification_graphs,
    verification_metrics,
)


def _vector(score):
    return np.array([score, np.sqrt(max(0, 1 - score**2))], dtype=np.float32)


def _pairs(genuine_scores, impostor_scores):
    embeddings = {"anchor": np.array([1.0, 0.0], dtype=np.float32)}
    rows = []
    for same, scores in ((True, genuine_scores), (False, impostor_scores)):
        for index, score in enumerate(scores):
            path = f"{same}_{index}"
            embeddings[path] = _vector(score)
            rows.append({"path_a": "anchor", "path_b": path, "is_same": same})
    return embeddings, pd.DataFrame(rows)


def test_tied_scores_never_split_to_meet_fmr():
    embeddings, pairs = _pairs([0.95, 0.8, 0.7], [0.8, 0.8, 0.1, 0.0])
    metrics = verification_metrics(embeddings, pairs, target_fmr=0.25)
    assert metrics["fmr"] == 0
    assert metrics["tar"] == pytest.approx(1 / 3)
    assert metrics["threshold"] > 0.8
    assert metrics["auc"] <= 1
    json.dumps(metrics, allow_nan=False)


def test_all_tied_can_reject_everything_with_finite_threshold():
    embeddings, pairs = _pairs([1, 1], [1, 1])
    metrics = verification_metrics(embeddings, pairs, target_fmr=0)
    assert np.isfinite(metrics["threshold"])
    assert metrics["threshold"] > 1
    assert metrics["fmr"] == metrics["tar"] == 0
    assert metrics["auc"] == pytest.approx(0.5)
    assert metrics["eer"] == pytest.approx(0.5)


def test_auc_does_not_require_removed_numpy_trapz(monkeypatch):
    monkeypatch.delattr(np, "trapz", raising=False)
    embeddings, pairs = _pairs([0.9, 0.8], [0.2, 0.1])
    assert verification_metrics(embeddings, pairs)["auc"] == pytest.approx(1.0)


def test_frozen_dev_threshold_is_not_recalibrated_on_test():
    dev_embeddings, dev_pairs = _pairs([0.9], [0.8])
    dev = verification_metrics(dev_embeddings, dev_pairs, target_fmr=0)
    embeddings, pairs = _pairs([0.7], [0.95])
    test = verification_metrics(
        embeddings, pairs, target_fmr=0, threshold=dev["threshold"]
    )
    assert test["threshold"] == dev["threshold"]
    assert test["threshold_source"] == "fixed"
    assert test["fmr"] == 1
    assert test["tar"] == 0


class _TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(3)

    def forward(self, images):
        return self.bn(images.mean(dim=(2, 3)))


def _image_frame(tmp_path):
    rows = []
    colors = [(230, 30, 20), (220, 40, 30), (20, 220, 30), (30, 230, 40)]
    for index, color in enumerate(colors):
        path = f"{index}.jpg"
        Image.new("RGB", (20, 30), color).save(tmp_path / path)
        rows.append(
            {"rel_path": path, "identity_id": str(index // 2), "label": index // 2}
        )
    return pd.DataFrame(rows)


def test_extract_normalizes_preserves_modes_and_does_not_update_bn(tmp_path):
    frame = _image_frame(tmp_path)
    model = _TinyEncoder().train()
    model.bn.eval()
    running = model.bn.running_mean.clone()
    tracked = model.bn.num_batches_tracked.clone()
    embeddings = extract_embeddings(model, frame, tmp_path, "cpu", batch_size=2)
    assert set(embeddings) == set(frame["rel_path"])
    assert model.training and not model.bn.training
    assert torch.equal(running, model.bn.running_mean)
    assert torch.equal(tracked, model.bn.num_batches_tracked)
    for vector in embeddings.values():
        assert vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1, abs=1e-6)


def test_probe_uses_unique_images_and_rejects_nontraining_paths(tmp_path):
    frame = _image_frame(tmp_path)
    model = _TinyEncoder().train()
    probe = pd.DataFrame(
        {
            "anchor": ["0.jpg", "0.jpg"],
            "positive": ["1.jpg", "1.jpg"],
            "negative": ["2.jpg", "3.jpg"],
        }
    )
    result = evaluate_probe(model, frame, probe, tmp_path, "cpu", batch_size=2)
    assert result["ranking_rate"] == 1
    assert result["mean_gap"] > 0
    embeddings = extract_embeddings(model, frame, tmp_path, "cpu", batch_size=2)
    expected = np.stack(list(embeddings.values())).std(axis=0, ddof=0).mean()
    assert result["feature_spread"] == pytest.approx(float(expected))
    assert result["unique_images"] == 4
    assert model.training
    probe.loc[0, "negative"] = "heldout.jpg"
    with pytest.raises(ValueError, match="training manifest"):
        evaluate_probe(model, frame, probe, tmp_path, "cpu")


def test_open_set_manifest_has_no_enrollment_or_duplicate_leakage():
    rows = [
        {
            "rel_path": f"{identity}/{image}.jpg",
            "identity_id": str(identity),
            "sha256": f"{identity}_{image // 2}",
        }
        for identity in range(4)
        for image in range(8)
    ]
    frame = pd.DataFrame(rows)
    protocol = create_open_set_manifest(frame, seed=42, enrolled=2, enrollment=2)
    assert protocol == create_open_set_manifest(
        frame, seed=42, enrolled=2, enrollment=2
    )
    gallery = {path for person in protocol["gallery"] for path in person["paths"]}
    probes = {probe["rel_path"] for probe in protocol["probes"]}
    assert not gallery & probes
    hashes = frame.set_index("rel_path")["sha256"]
    assert not set(hashes.loc[list(gallery)]) & set(hashes.loc[list(probes)])
    assert len(protocol["gallery"]) == 2
    assert {probe["is_enrolled"] for probe in protocol["probes"]} == {False, True}
    assert not (
        {
            probe["identity_id"]
            for probe in protocol["probes"]
            if not probe["is_enrolled"]
        }
        & {person["identity_id"] for person in protocol["gallery"]}
    )


def test_unknown_fpir_counts_searches_not_gallery_comparisons():
    embeddings = {
        "gA": np.array([1, 0]),
        "gB": np.array([0, 1]),
        "pA": np.array([1, 0]),
        "pB": np.array([0, 1]),
        "unknown": np.array([1, 1]),
    }
    protocol = {
        "gallery": [
            {"identity_id": "A", "paths": ["gA"]},
            {"identity_id": "B", "paths": ["gB"]},
        ],
        "probes": [
            {"rel_path": "pA", "identity_id": "A", "is_enrolled": True},
            {"rel_path": "pB", "identity_id": "B", "is_enrolled": True},
            {"rel_path": "unknown", "identity_id": "C", "is_enrolled": False},
        ],
    }
    result = evaluate_identification(embeddings, protocol=protocol, threshold=0.7)
    assert result["fpir"] == 1  # One false search, not two independent comparisons.
    assert result["tpir"] == result["rank1"] == 1
    assert result["unknown_probes"] == 1
    calibrated = evaluate_identification(embeddings, protocol=protocol, target_fpir=0)
    assert calibrated["fpir"] == 0
    fixed = evaluate_identification(
        embeddings, protocol=protocol, threshold=calibrated["threshold"]
    )
    assert fixed["threshold"] == calibrated["threshold"]
    json.dumps(result, allow_nan=False)


def test_identification_requires_correct_identity_and_threshold():
    embeddings = {
        "gA": np.array([1, 0]),
        "gB": np.array([0, 1]),
        "wrongA": np.array([0, 1]),
        "unknown": np.array([-1, -1]),
    }
    protocol = {
        "gallery": [
            {"identity_id": "A", "paths": ["gA"]},
            {"identity_id": "B", "paths": ["gB"]},
        ],
        "probes": [
            {"rel_path": "wrongA", "identity_id": "A", "is_enrolled": True},
            {"rel_path": "unknown", "identity_id": "C", "is_enrolled": False},
        ],
    }
    metrics = evaluate_identification(embeddings, protocol=protocol, threshold=0.5)
    assert metrics["rank1"] == metrics["tpir"] == 0
    assert metrics["fpir"] == 0


def test_plot_artifacts_keep_frozen_threshold_and_full_history(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.6],
            "dev_tar_at_fmr": [0.2, 0.4, 0.5],
            "dev_auc": [0.6, 0.7, 0.8],
            "stage": ["random", "random", "semihard"],
        }
    )
    graph = save_training_graph(history, tmp_path / "training.png")
    assert Path(graph).stat().st_size > 0
    embeddings, pairs = _pairs([0.9, 0.8], [0.2, 0.1])
    artifacts = save_verification_graphs(
        embeddings, pairs, tmp_path / "verification", threshold=0.85
    )
    assert artifacts["metrics"]["threshold"] == 0.85
    assert Path(artifacts["roc"]).stat().st_size > 0
    assert Path(artifacts["score_distribution"]).stat().st_size > 0
    sweep = pd.DataFrame(
        {
            "family": ["B0", "B0", "B1", "B1"],
            "margin": [0.2, 0.3, 0.3, 0.4],
            "dev_tar_at_fmr": [0.5, 0.6, 0.7, 0.65],
        }
    )
    assert Path(plot_sweep(sweep, tmp_path / "sweep.png")).stat().st_size > 0
