"""Bbox protocol, bounded streaming, and split-integrity regression checks."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.recognition.identification.config import ExperimentConfig
from src.recognition.identification.data import FaceDataset, PKBatchSampler, make_loader
from src.recognition.identification.embedding_evaluation import extract_embeddings
from src.recognition.identification.protocol import load_face_protocol


@pytest.fixture
def face_protocol(tmp_path):
    root = tmp_path / "images"
    directory = tmp_path / "recognition_faces" / ("protocol_" + "a" * 12)
    directory.mkdir(parents=True)
    rows = []
    for identity in range(6):
        for index in range(4):
            path = root / "train" / f"id_{identity}" / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.zeros((24, 32, 3), dtype=np.uint8)
            pixels[:] = [1, 2, 240]
            pixels[4:20, 8:24] = [80 + identity * 20, 20 + index * 30, 10]
            Image.fromarray(pixels).save(path)
            rows.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    identity=f"id_{identity}",
                    split=(
                        "train" if identity < 2 else "val" if identity < 4 else "test"
                    ),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    x1=8,
                    y1=4,
                    x2=24,
                    y2=20,
                )
            )
    frame = pd.DataFrame(rows)
    classification = frame[frame.split == "train"]
    # Classification held-out images must never be folded back into training.
    holdout = classification.iloc[[0]].copy()
    holdout["path"] = "train/id_0/excluded.png"
    holdout["split"] = "val"
    pd.concat([classification, holdout]).to_csv(
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
    (directory / "summary.json").write_text(json.dumps({"complete": True}))
    pointer = directory.parent / "current.json"
    pointer.write_text(json.dumps({"complete": True, "protocol_sha256": "a" * 64}))
    return root, pointer, directory


def test_protocol_loads_metadata_without_decoding_and_preserves_unseen_splits(
    face_protocol, monkeypatch
):
    root, pointer, directory = face_protocol
    with monkeypatch.context() as context:
        context.setattr(
            Image, "open", lambda *a, **k: pytest.fail("Eager image decoding")
        )
        prepared = load_face_protocol(root, pointer)
    assert not (root / "val").exists()
    assert [len(frame) for frame in (prepared.train, prepared.dev, prepared.test)] == [
        8,
        8,
        8,
    ]
    assert set(prepared.train_labels.values()) == {0, 1}
    assert set(prepared.train.identity_id).isdisjoint(prepared.dev.identity_id)
    assert set(prepared.dev.identity_id).isdisjoint(prepared.test.identity_id)
    assert len(prepared.dev_pairs) == 2
    assert set(prepared.probe_triplets.anchor) <= set(prepared.train.rel_path)
    first = prepared.metadata["artifact_sha256"]
    csv = directory / "vggface2_classification.csv"
    frame = pd.read_csv(csv)
    frame.loc[0, "x1"] = 9
    frame.to_csv(csv, index=False)
    changed = load_face_protocol(root, pointer)
    assert changed.metadata["artifact_sha256"] != first


def test_bbox_crop_is_used_in_training_and_batched_evaluation(face_protocol):
    root, pointer, _ = face_protocol
    prepared = load_face_protocol(root, pointer)
    dataset = FaceDataset(prepared.train, root, train=True, require_bboxes=True)
    assert not hasattr(dataset, "frame")
    crop = dataset[0]["image"]
    assert crop.device.type == "cpu" and crop.dtype == torch.uint8
    assert crop.shape == (3, 112, 112)
    assert torch.equal(crop[:, 0, 0], torch.tensor([80, 20, 10], dtype=torch.uint8))

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, images):
            self.inputs.append(images.clone())
            return images.mean((2, 3))

    model = Capture()
    vectors = extract_embeddings(
        model, prepared.train, root, "cpu", batch_size=3, require_bboxes=True
    )
    assert len(vectors) == 8
    assert [len(batch) for batch in model.inputs] == [3, 3, 2]
    torch.testing.assert_close(
        model.inputs[0][0, :, 0, 0], crop[:, 0, 0].float() / 127.5 - 1
    )


@pytest.mark.parametrize(
    "damage,match",
    [
        ("missing_box", "positive-area"),
        ("pair", "pair labels"),
        ("leak", "leakage"),
        ("partial", "completed"),
    ],
)
def test_protocol_rejects_bad_boxes_pairs_leakage_and_incomplete_scans(
    face_protocol, damage, match
):
    root, pointer, directory = face_protocol
    if damage == "partial":
        pointer.write_text(json.dumps({"complete": False, "protocol_sha256": "a" * 64}))
    else:
        name = (
            "verification_pairs_val.csv"
            if damage == "pair"
            else "vggface2_verification.csv"
        )
        path = directory / name
        frame = pd.read_csv(path)
        if damage == "missing_box":
            frame.loc[0, "x2"] = 0
        elif damage == "pair":
            frame.loc[0, "label"] = 0
        else:
            frame.loc[0, "identity"] = "id_0"
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        load_face_protocol(root, pointer)


def test_missing_bboxes_never_silently_fall_back_when_required(face_protocol):
    root, pointer, _ = face_protocol
    prepared = load_face_protocol(root, pointer)
    missing = prepared.train.drop(columns=["x1", "y1", "x2", "y2"])
    with pytest.raises(ValueError, match="bboxes require"):
        FaceDataset(missing, root, require_bboxes=True)
    outside = prepared.train.copy()
    outside["x2"] = 100
    with pytest.raises(ValueError, match="outside"):
        FaceDataset(outside, root, require_bboxes=True)[0]


def test_spawn_streaming_matches_serial_bbox_batches(face_protocol):
    root, pointer, _ = face_protocol
    prepared = load_face_protocol(root, pointer)
    sampler = PKBatchSampler(prepared.train.label, P=2, K=2, batches_per_epoch=2)
    serial = make_loader(
        prepared.train,
        root,
        train=True,
        batch_sampler=sampler,
        num_workers=0,
        require_bboxes=True,
    )
    workers = make_loader(
        prepared.train,
        root,
        train=True,
        batch_sampler=sampler,
        num_workers=2,
        prefetch_factor=1,
        persistent_workers=False,
        require_bboxes=True,
    )
    for first, second in zip(serial, workers, strict=True):
        assert torch.equal(first["image"], second["image"])
        assert torch.equal(first["augmentation_params"], second["augmentation_params"])
        assert first["image"].device.type == "cpu"


def test_hardware_config_and_invalid_io_settings():
    config = ExperimentConfig()
    config.validate()
    assert config.embedding_dim == 128 and config.batch_size == 128
    assert config.num_workers == 4 and config.evaluation_num_workers == 2
    assert config.amp and config.pin_memory and not config.require_bboxes
    for settings in (
        {"prefetch_factor": 0},
        {"evaluation_num_workers": -1},
        {"cpu_threads": 0},
        {"evaluation_batch_size": 0},
    ):
        with pytest.raises(ValueError):
            replace(config, **settings).validate()
