"""Mask pipeline on a tiny fake FMLD: prepare -> train -> evaluate -> predictor (CPU, ~20 s)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.mask import evaluate, prepare, train
from src.mask.model import build_model, load_checkpoint
from src.mask.predictor import MaskPredictor

XML = """<annotation><filename>{name}</filename><path>{path}</path>
<source><database>{db}</database></source>{objects}</annotation>"""
OBJECT = """<object><name>{label}</name><difficult>0</difficult><bndbox>
<xmin>40</xmin><ymin>30</ymin><xmax>140</xmax><ymax>150</ymax></bndbox></object>"""


def _face(masked: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60, (200, 200, 3), dtype=np.uint8)
    cv2.ellipse(image, (90, 90), (45, 60), 0, 0, 360, (150, 170, 200), -1)
    if masked:
        cv2.rectangle(image, (50, 95), (130, 150), (230, 220, 210), -1)
    return image


def _fake_fmld(root: Path) -> None:
    for split, count in (("train", 24), ("test", 8)):
        (root / "FMLD_annotations" / split).mkdir(parents=True)
        folder = f"{split}-images"
        for i in range(count):
            masked = i % 2 == 0
            name = f"{split}_{i:04d}.jpg"
            image_path = root / "MAFA" / folder / "images" / name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), _face(masked, i))
            label = "masked_face" if masked else "unmasked_face"
            xml = XML.format(
                name=name, path=f"{folder}/{name}", db="MAFA", objects=OBJECT.format(label=label)
            )
            (root / "FMLD_annotations" / split / f"{split}_{i:04d}.xml").write_text(xml)


def test_mask_pipeline(tmp_path, monkeypatch):
    data = tmp_path / "data"
    _fake_fmld(data)
    faces = data / "mask_faces"
    options = prepare.Options(data, faces, size=96, scale=1.3, min_face=40, val_fraction=0.25)
    counts = prepare.prepare(options, workers=1)
    assert counts["test"] == {"mask": 4, "no_mask": 4}
    assert sum(counts["train"].values()) + sum(counts["val"].values()) == 24

    ckpt_dir, figures = tmp_path / "ckpt", tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--faces",
            str(faces),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--workers",
            "0",
            "--no-pretrained",
            "--ckpt-dir",
            str(ckpt_dir),
            "--figures",
            str(figures),
        ],
    )
    train.main()
    checkpoint = next(ckpt_dir.glob("mask_mobilenetv3s_*.pt"))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--faces",
            str(faces),
            "--figures",
            str(figures),
            "--workers",
            "0",
        ],
    )
    evaluate.main()
    assert (figures / "mask_results.md").exists()

    label, prob = MaskPredictor(checkpoint).predict_frame(_face(True, 0), (40, 30, 140, 150))
    assert label in ("mask", "no_mask") and 0.0 <= prob <= 1.0


def test_mobilenetv4_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("timm")
    model = build_model("mobilenetv4m", pretrained=False).eval()
    path = tmp_path / "mask_mobilenetv4m.pt"
    config = {
        "task": "mask",
        "arch": "mobilenetv4m",
        "classes": ["mask", "no_mask"],
        "img_size": 224,
        "crop_scale": 1.3,
        "mean": [],
        "std": [],
        "threshold": 0.5,
    }
    torch.save({"model": model.state_dict(), "config": config}, path)
    loaded, _ = load_checkpoint(path, torch.device("cpu"))
    images = torch.rand(2, 3, 224, 224)
    with torch.no_grad():
        assert loaded(images).shape == (2, 2)
        assert torch.allclose(loaded(images), model(images))
