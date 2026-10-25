"""Anti-spoofing tests. The pipeline test trains both models for one epoch on tiny random images (CPU, ~1 min)."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.common.face_crop import crop_face
from src.antispoofing.metrics import eer_threshold, summarize

REPO = Path(__file__).resolve().parents[1]


def test_crop_face_size_and_padding():
    frame = np.full((100, 200, 3), 255, np.uint8)
    face = crop_face(frame, (150, 10, 190, 50), size=64, scale=2.0)  # crop runs off the top-right edge
    assert face.shape == (64, 64, 3)
    assert face[0, -1].sum() == 0 and face[32, 32].sum() == 255 * 3


def test_metrics_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.3])
    labels = np.array([1, 1, 0, 0, 0])
    attacks = np.array(["live", "live", "print", "replay", "replay"])
    r = summarize(scores, labels, attacks, threshold=0.5)
    assert r["acer"] == 0 and r["auc"] == 1 and r["accuracy"] == 1


def test_apcer_is_worst_attack():
    scores = np.array([0.9, 0.9, 0.9, 0.1, 0.6, 0.1])
    labels = np.array([1, 1, 0, 0, 0, 0])
    attacks = np.array(["live", "live", "print", "print", "replay", "replay"])
    r = summarize(scores, labels, attacks, threshold=0.5)
    assert r["apcer_per_attack"] == {"print": 0.5, "replay": 0.5}
    assert r["apcer"] == 0.5 and r["bpcer"] == 0 and r["acer"] == 0.25
    assert r["apcer_pooled"] == 0.5


def test_eer_threshold_balances_errors():
    rng = np.random.default_rng(0)
    labels = np.r_[np.ones(500), np.zeros(500)]
    scores = np.r_[rng.normal(0.7, 0.1, 500), rng.normal(0.4, 0.1, 500)]
    t, eer = eer_threshold(scores, labels)
    r = summarize(scores, labels, np.where(labels == 1, "live", "print"), t)
    assert 0.5 < t < 0.6 and abs(r["apcer"] - r["bpcer"]) < 0.02 and abs(eer - r["acer"]) < 0.02


def run(*args):
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


@pytest.mark.parametrize("model", ["mobilenetv3", "deeppixbis"])
def test_pipeline_end_to_end(tmp_path, model):
    rng = np.random.default_rng(0)
    root = tmp_path / "data" / "recorded"
    for person in ["p1", "p2", "p3", "p4", "p5"]:
        for cls in ["live", "print", "replay"]:
            (root / person / cls).mkdir(parents=True)
            for i in range(4):
                cv2.imwrite(str(root / person / cls / f"{i}.jpg"), rng.integers(0, 255, (80, 80, 3), np.uint8))

    manifest = tmp_path / "data" / "recorded.csv"
    run("-m", "src.antispoofing.manifest", "folder", "--root", root, "--data-root", tmp_path / "data",
        "--detect", "none", "--test-subjects", "p5", "--val-frac", "0.25", "--out", manifest)
    df = pd.read_csv(manifest)
    assert set(df[df.split == "test"].subject) == {"p5"}
    assert not set(df[df.split == "train"].subject) & set(df[df.split == "val"].subject)

    ckpt_dir, fig_dir = tmp_path / "ckpt", tmp_path / "fig"
    common = ["--manifest", manifest, "--data-root", tmp_path / "data", "--fig-dir", fig_dir, "--num-workers", "0"]
    run("-m", "src.antispoofing.train", "--model", model, "--ckpt-dir", ckpt_dir, "--epochs", "1",
        "--img-size", "64", "--batch-size", "8", "--no-pretrained", *common)
    ckpt = next(ckpt_dir.glob(f"antispoofing_{model}_*.pt"))
    run("-m", "src.antispoofing.evaluate", "--checkpoint", ckpt, *common)

    table = pd.read_csv(fig_dir / "antispoofing_results.csv")
    assert list(table.model) == [model] and table.acer.between(0, 1).all()
    for kind in ["training", "roc", "scores"]:
        assert (fig_dir / f"antispoofing_{model}_{kind}.png").exists()

    from src.antispoofing.predictor import AntiSpoofingPredictor
    antispoofing = AntiSpoofingPredictor(ckpt, device="cpu")
    frame = rng.integers(0, 255, (120, 160, 3), np.uint8)
    label, prob = antispoofing.predict_frame(frame, (40, 20, 100, 80))
    assert label in {"live", "spoof"} and 0 <= prob <= 1
