"""Base recognition manifests support streamed training without face detection."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.recognition.identification.data import BBOX_COLUMNS, IMAGE_COLUMNS, FaceDataset
from src.recognition.identification.protocol import load_image_protocol


@pytest.fixture
def image_protocol(tmp_path):
    root, directory = tmp_path / "images", tmp_path / "base"
    directory.mkdir()
    rows = []
    for identity in range(7):
        for index in range(4):
            path = root / "train" / f"id_{identity}" / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((112, 112, 3), [1, 2, 240], dtype=np.uint8)
            pixels[28:84, 28:84] = [20 + identity * 20, 20 + index * 30, 10]
            Image.fromarray(pixels).save(path)
            rows.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    identity=f"id_{identity}",
                    split=(
                        "train" if identity < 3 else "val" if identity < 5 else "test"
                    ),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    frame = pd.DataFrame(rows)
    holdout = frame.iloc[[0]].copy()
    holdout["path"], holdout["split"] = "train/id_0/excluded.png", "val"
    pd.concat([frame[frame.split == "train"], holdout]).to_csv(
        directory / "vggface2_classification.csv", index=False
    )
    frame[frame.split != "train"].to_csv(
        directory / "vggface2_verification.csv", index=False
    )
    for split in ("val", "test"):
        groups = [
            list(group.path)
            for _, group in frame[frame.split == split].groupby("identity")
        ]
        pd.DataFrame(
            [(groups[0][0], groups[0][1], 1), (groups[0][0], groups[1][0], 0)],
            columns=["path1", "path2", "label"],
        ).to_csv(directory / f"verification_pairs_{split}.csv", index=False)
    summary = {
        "classification_classes": 3,
        "classification_images": 13,
        "classification_splits": {"train": 12, "val": 1},
        "verification": {
            "val": {"identities": 2, "pairs": 2},
            "test": {"identities": 2, "pairs": 2},
        },
    }
    _write_summaries(directory, summary)
    return root, directory


def _write_summaries(directory, summary):
    (directory / "recognition_summary.json").write_text(json.dumps(summary))
    (directory / "dataset_report.json").write_text(json.dumps({"recognition": summary}))


def test_full_image_protocol_is_metadata_only_and_keeps_unseen_splits(
    image_protocol, monkeypatch
):
    root, directory = image_protocol
    with monkeypatch.context() as context:
        context.setattr(
            Image, "open", lambda *args, **kwargs: pytest.fail("Eager image decoding")
        )
        prepared = load_image_protocol(root, directory)
    assert not (root / "val").exists()
    assert [len(frame) for frame in (prepared.train, prepared.dev, prepared.test)] == [
        12,
        8,
        8,
    ]
    for frame in (prepared.train, prepared.dev, prepared.test):
        assert list(frame.columns) == IMAGE_COLUMNS
        assert not set(BBOX_COLUMNS) & set(frame.columns)
    assert set(prepared.train_labels.values()) == {0, 1, 2}
    assert set(prepared.train.identity_id).isdisjoint(prepared.dev.identity_id)
    assert set(prepared.train.identity_id).isdisjoint(prepared.test.identity_id)
    assert set(prepared.dev.identity_id).isdisjoint(prepared.test.identity_id)
    assert len(prepared.dev_pairs) == len(prepared.test_pairs) == 2
    assert set(prepared.probe_triplets.anchor) <= set(prepared.train.rel_path)
    assert "train/id_0/excluded.png" not in set(prepared.train.rel_path)
    assert prepared.metadata["version"] == "full-image-identification-v1"
    assert "full image" in prepared.metadata["crop"]
    assert set(prepared.metadata["artifact_sha256"]) == {
        "vggface2_classification.csv",
        "vggface2_verification.csv",
        "verification_pairs_val.csv",
        "verification_pairs_test.csv",
        "dataset_report.json",
        "recognition_summary.json",
    }


def test_full_frame_pixels_and_optional_bbox_columns_are_not_used(image_protocol):
    root, directory = image_protocol
    for name in ("vggface2_classification.csv", "vggface2_verification.csv"):
        frame = pd.read_csv(directory / name)
        for column in BBOX_COLUMNS:
            frame[column] = -999
        frame.to_csv(directory / name, index=False)
    prepared = load_image_protocol(root, directory)
    sample = FaceDataset(prepared.train, root, train=True)[0]["image"]
    assert sample.shape == (3, 112, 112)
    assert sample.dtype == torch.uint8 and sample.device.type == "cpu"
    assert torch.equal(sample[:, 0, 0], torch.tensor([1, 2, 240], dtype=torch.uint8))
    assert torch.equal(sample[:, 50, 50], torch.tensor([20, 20, 10], dtype=torch.uint8))


def test_base_protocol_deduplicates_content_and_drops_short_train_identities(
    image_protocol,
):
    root, directory = image_protocol
    path = directory / "vggface2_classification.csv"
    frame = pd.read_csv(path)
    frame.loc[1, "sha256"] = frame.loc[0, "sha256"]
    frame.to_csv(path, index=False)
    prepared = load_image_protocol(root, directory)
    assert prepared.metadata["dropped_short_train_identities"] == ["id_0"]
    assert set(prepared.train.identity_id) == {"id_1", "id_2"}
    assert prepared.train_labels == {"id_1": 0, "id_2": 1}


def test_base_protocol_hashes_markers_and_is_reproducible(image_protocol):
    root, directory = image_protocol
    first = load_image_protocol(root, directory, seed=123)
    second = load_image_protocol(root, directory, seed=123)
    pd.testing.assert_frame_equal(first.probe_triplets, second.probe_triplets)
    summary = json.loads((directory / "recognition_summary.json").read_text())
    summary["note"] = "New generation metadata"
    _write_summaries(directory, summary)
    changed = load_image_protocol(root, directory, seed=123)
    assert changed.metadata["artifact_sha256"] != first.metadata["artifact_sha256"]


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing_marker", "Missing completed"),
        ("disagree", "incomplete"),
        ("train_count", "train count"),
        ("identity_count", "identity count"),
        ("pair_count", "pair count"),
    ],
)
def test_base_protocol_requires_matching_completion_markers(
    image_protocol, damage, match
):
    root, directory = image_protocol
    summary = json.loads((directory / "recognition_summary.json").read_text())
    if damage == "missing_marker":
        (directory / "dataset_report.json").unlink()
    elif damage == "disagree":
        (directory / "dataset_report.json").write_text("{}")
    else:
        if damage == "train_count":
            summary["classification_splits"]["train"] = 13
        elif damage == "identity_count":
            summary["verification"]["val"]["identities"] = 3
        else:
            summary["verification"]["test"]["pairs"] = 3
        _write_summaries(directory, summary)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_image_protocol(root, directory)


@pytest.mark.parametrize(
    "damage,match",
    [
        ("traversal", "Unsafe"),
        ("absolute", "Unsafe"),
        ("sha256", "Invalid SHA256"),
        ("duplicate_path", "Duplicate image paths"),
        ("duplicate_identity", "conflicting identity labels"),
        ("identity_leak", "leakage"),
        ("content_leak", "leakage"),
    ],
)
def test_full_image_protocol_preserves_manifest_integrity(
    image_protocol, damage, match
):
    root, directory = image_protocol
    name = (
        "vggface2_verification.csv"
        if damage in {"identity_leak", "content_leak"}
        else "vggface2_classification.csv"
    )
    path = directory / name
    frame = pd.read_csv(path)
    if damage == "traversal":
        frame.loc[0, "path"] = "train/../outside.png"
    elif damage == "absolute":
        frame.loc[0, "path"] = "C:/outside.png"
    elif damage == "sha256":
        frame.loc[0, "sha256"] = "not a sha256"
    elif damage == "duplicate_path":
        frame.loc[1, "path"] = frame.loc[0, "path"].replace("/", "\\")
    elif damage == "duplicate_identity":
        frame.loc[4, "sha256"] = frame.loc[0, "sha256"]
    elif damage == "identity_leak":
        frame.loc[frame.identity == "id_3", "identity"] = "id_0"
    else:
        train = pd.read_csv(directory / "vggface2_classification.csv")
        frame.loc[0, "sha256"] = train.loc[0, "sha256"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_image_protocol(root, directory)


@pytest.mark.parametrize("damage", ["labels", "unknown", "duplicate", "same_content"])
def test_full_image_protocol_validates_published_finite_pairs(image_protocol, damage):
    root, directory = image_protocol
    path = directory / "verification_pairs_val.csv"
    pairs = pd.read_csv(path)
    if damage == "labels":
        pairs.loc[0, "label"] = 0
    elif damage == "unknown":
        pairs.loc[0, "path1"] = "train/id_0/0.png"
    elif damage == "duplicate":
        pairs = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    else:
        pairs.loc[0, "path2"] = pairs.loc[0, "path1"]
    pairs.to_csv(path, index=False)
    with pytest.raises(ValueError, match="[Pp]air"):
        load_image_protocol(root, directory)


@pytest.mark.parametrize(
    "kwargs", [{"images_per_identity": 1}, {"probe_per_identity": 0}]
)
def test_full_image_protocol_validates_sampler_settings(image_protocol, kwargs):
    root, directory = image_protocol
    with pytest.raises(ValueError, match="required"):
        load_image_protocol(root, directory, **kwargs)
