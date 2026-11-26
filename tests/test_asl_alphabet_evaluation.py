"""Score failed landmark extraction as a miss in the image-level report."""

from importlib import import_module

import numpy as np
import pytest

alphabet = import_module("src.hand-gestures.alphabet")


def test_report_uses_manifest_denominator(tmp_path, monkeypatch):
    data = alphabet.Dataset(
        np.zeros((1, 21, 3), dtype=np.float32),
        np.array([0]),
        np.array(["detected.jpg"]),
        np.array(["test"]),
        "extractor",
        "manifest",
        {"detected.jpg": ("test", 0), "missing.jpg": ("test", 0)},
    )
    metadata = {
        "fingerprint": "extractor",
        "manifest_sha256": "manifest",
        "model": "mlp",
    }
    scores = np.eye(len(alphabet.CLASSES))[:1]
    monkeypatch.setattr(
        alphabet, "load_predictor", lambda *_: (metadata, lambda _: scores)
    )
    report = alphabet.evaluate(data, tmp_path, "test")
    assert report["accuracy"] == 1
    assert report["accuracy_with_extraction_failures"] == 0.5
    assert report["macro_f1_with_extraction_failures"] == pytest.approx(2 / 3 / 26)
    assert report["per_class"]["A"]["recall_with_extraction_failures"] == 0.5
