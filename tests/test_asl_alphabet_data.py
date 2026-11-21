"""Checks for split isolation and the publisher holdout."""

from hashlib import sha256
from importlib import import_module
from string import ascii_uppercase

import numpy as np

alphabet = import_module("src.hand-gestures.alphabet_data")


def test_manifest_excludes_boundary_neighbors_and_cross_split_hashes(tmp_path):
    dataset = tmp_path / "ASL"
    train = dataset / "asl_alphabet_train" / "asl_alphabet_train"
    external = dataset / "asl_alphabet_test" / "asl_alphabet_test"
    external.mkdir(parents=True)
    for label in ascii_uppercase:
        directory = train / label
        directory.mkdir(parents=True)
        for index in (50, 100, 101, 150, 250, 350, 450, 550):
            (directory / f"{label}{index}.jpg").write_bytes(f"{label}:{index}".encode())
    (external / "A_test.jpg").write_bytes(b"A:50")
    (external / "space_test.jpg").write_bytes(b"space")
    (train / "A" / "A250.jpg").write_bytes(b"B:250")
    rows = alphabet.build_manifest(dataset, seed=42, workers=2)
    source = {row["path"].rsplit("/", 1)[-1]: row for row in rows}
    assert "space_test.jpg" not in source
    assert source["A_test.jpg"]["split"] == "external"
    assert source["A50.jpg"]["canonical_path"] == source["A_test.jpg"]["path"]
    assert source["A50.jpg"]["status"] == "duplicate"
    assert (
        source["A250.jpg"]["status"]
        == source["B250.jpg"]["status"]
        == "conflicting_labels"
    )
    ready = [row for row in rows if row["status"] == "ready"]
    assert len({row["sha256"] for row in ready}) == len(ready)
    assert set(row["split"] for row in ready) == set(alphabet.SPLITS)
    internal = [row for row in rows if row["split"] != "external"]
    assert all(
        len({row["split"] for row in internal if row["block"] == block}) == 1
        for block in {row["block"] for row in internal}
    )
    assignments = {row["block"]: row["split"] for row in internal}
    assert alphabet._guard(100, {0: "train", 1: "val"})
    assert alphabet._guard(101, {0: "train", 1: "val"})
    assert not alphabet._guard(100, {0: "train", 1: "train"})
    assert {row["number"] for row in internal if row["status"] == "boundary_guard"} == {
        number for number in (100, 101) if alphabet._guard(number, assignments)
    }


def test_extraction_resumes_completed_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(alphabet, "CHUNK_SIZE", 2)
    calls = []

    def detect(content):
        calls.append(content)
        if content == b"missing":
            return None, "no_hand"
        return np.ones((21, 3), np.float32), "Right"

    monkeypatch.setattr(
        alphabet, "_make_worker", lambda asset, device: (detect, lambda: None)
    )
    root = tmp_path / "images"
    chunks = tmp_path / "cache"
    root.mkdir()
    chunks.mkdir()
    rows = []
    for name, content in (("A1.jpg", b"ok"), ("A2.jpg", b"missing"), ("B1.jpg", b"ok")):
        (root / name).write_bytes(content)
        rows.append(
            {
                "path": name,
                "label": name[0],
                "split": "train",
                "sha256": sha256(content).hexdigest(),
            }
        )
    first, rejected = alphabet._extract(root, rows, b"task", 1, chunks, "cpu")
    assert len(calls) == 3
    assert rejected == [{"path": "A2.jpg", "reason": "no_hand"}]
    (chunks / "000002.npz").write_bytes(b"interrupted")
    resumed, rejected_again = alphabet._extract(root, rows, b"task", 1, chunks, "cpu")
    assert len(calls) == 4
    assert rejected == rejected_again
    assert np.array_equal(first["points"], resumed["points"])
    assert np.array_equal(first["paths"], resumed["paths"])
