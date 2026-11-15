"""Manifest exports exercise real miniature databases and image files."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pytest
from PIL import Image

from src.datasets.config import PreparationConfig
from src.datasets.emotion import EmotionPreparer
from src.datasets.liveness import LivenessPreparer
from src.datasets.recognition import RecognitionPreparer
from src.datasets.team import TeamPreparer, select_gallery_face
from src.datasets.validate import ManifestValidator
from src.recognition.classification.config import RecognitionConfig
from src.recognition.classification.data import make_loaders
from src.recognition.verification import PairEvaluator, VerificationConfig
from src.training.config import read_yaml

REPO = Path(__file__).resolve().parents[1]


def save_image(path: Path, seed: int, size=48):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(
        0, 256, (size, size, 3), dtype=np.uint8
    )
    Image.fromarray(image).save(path)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def preparation_config(tmp_path):
    raw = read_yaml(REPO / "configs/datasets.yaml")
    raw.update(output_dir=str(tmp_path / "manifests"), workers=2)
    for name in ("recognition", "liveness", "emotion", "team"):
        raw[name]["root"] = str(tmp_path / name)
        (tmp_path / name).mkdir()
    raw["recognition"].update(
        positive_pairs=4, negative_pairs=4, val_fraction=0.2, test_fraction=0.2
    )
    raw["emotion"]["expected_frames"] = 2
    config = PreparationConfig.model_validate(raw)
    config.output_dir.mkdir()
    return config


def build_vgg(config):
    root = config.recognition.root
    rows = []
    for identity_index in range(6):
        split = "train" if identity_index < 2 else "val"
        identity = f"person_{identity_index}"
        for index in range(6):
            relative = f"{split}/{identity}/{index:04d}_01.jpg"
            save_image(root / relative, 100 * identity_index + index)
            rows.append((relative, identity, split, identity_index))
    with duckdb.connect(str(root / config.recognition.database)) as con:
        con.execute(
            "CREATE TABLE vggface2_images (rel_path VARCHAR, identity_id VARCHAR, "
            "split VARCHAR, identity_idx INTEGER)"
        )
        con.executemany("INSERT INTO vggface2_images VALUES (?, ?, ?, ?)", rows)


def test_recognition_manifests_are_deterministic_and_loadable(preparation_config):
    config = preparation_config
    build_vgg(config)
    database = config.recognition.root / config.recognition.database
    original = hashlib.sha256(database.read_bytes()).hexdigest()
    first = RecognitionPreparer(config).prepare()
    before = {path.name: path.read_bytes() for path in config.output_dir.glob("*.csv")}
    assert first["classification_classes"] == 2
    assert first["classification_splits"] == {"train": 8, "val": 2, "test": 2}
    assert first["verified_image_dimensions"] == {"48x48": 36}
    assert first["verification_images"] == 24
    assert first["verification_duplicate_rows"] == 0
    assert RecognitionPreparer(config).prepare() == first
    assert before == {
        path.name: path.read_bytes() for path in config.output_dir.glob("*.csv")
    }
    assert hashlib.sha256(database.read_bytes()).hexdigest() == original
    raw = read_yaml(REPO / "configs/recognition/model_a.yaml")
    raw["data"].update(
        root=str(config.recognition.root),
        manifest=str(config.output_dir / "vggface2_classification.csv"),
    )
    raw["preprocess"]["size"] = 32
    raw["training"].update(batch_size=4, num_workers=0, device="cpu")
    training = RecognitionConfig.model_validate(raw)
    train_loader, val_loader, mapping = make_loaders(training)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32)
    assert labels.min() >= 0 and labels.max() < 2
    assert len(val_loader.dataset) == 2
    assert mapping == json.loads(
        (config.output_dir / "vggface2_class_to_idx.json").read_text()
    )
    val = PairEvaluator(
        VerificationConfig(
            root=config.recognition.root,
            pairs=config.output_dir / "verification_pairs_val.csv",
            pairs_format="csv",
            target_fpr=0.01,
            selection_metric="cosine",
        ),
        size=32,
        crop_scale=1,
        progress=False,
    )
    assert len(val.pairs) == 8
    val_ids = {
        path.parent.name for first, second, _ in val.pairs for path in (first, second)
    }
    test_rows = read_csv(config.output_dir / "verification_pairs_test.csv")
    test_ids = {
        Path(row[key]).parent.name for row in test_rows for key in ("path1", "path2")
    }
    assert not val_ids & test_ids and not val_ids & set(mapping)
    assert ManifestValidator(config).recognition()["classes"] == 2


def test_validation_rejects_wrong_verification_labels(preparation_config):
    config = preparation_config
    build_vgg(config)
    RecognitionPreparer(config).prepare()
    pairs_path = config.output_dir / "verification_pairs_val.csv"
    pairs = read_csv(pairs_path)
    pairs[0]["label"] = str(1 - int(pairs[0]["label"]))
    with pairs_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path1", "path2", "label"])
        writer.writeheader()
        writer.writerows(pairs)
    with pytest.raises(ValueError, match="label disagrees"):
        ManifestValidator(config).recognition()


def test_duplicate_bytes_and_filename_groups_cannot_cross_classification_splits(
    preparation_config,
):
    config = preparation_config
    build_vgg(config)
    root = config.recognition.root
    original = "train/person_0/0000_01.jpg"
    duplicate = "train/person_0/9000_01.jpg"
    shutil.copyfile(root / original, root / duplicate)
    with duckdb.connect(str(root / config.recognition.database)) as con:
        con.execute(
            "INSERT INTO vggface2_images VALUES (?, 'person_0', 'train', 0)",
            [duplicate],
        )
    RecognitionPreparer(config).prepare()
    rows = read_csv(config.output_dir / "vggface2_classification.csv")
    assert (
        len({row["split"] for row in rows if row["path"] in {original, duplicate}}) == 1
    )
    conflicting = "val/person_2/9000_01.jpg"
    shutil.copyfile(root / original, root / conflicting)
    with duckdb.connect(str(root / config.recognition.database)) as con:
        con.execute(
            "INSERT INTO vggface2_images VALUES (?, 'person_2', 'val', 2)",
            [conflicting],
        )
    summary = RecognitionPreparer(config).prepare()
    assert summary["conflicting_hashes"] == 1 and summary["rejected_images"] == 3
    assert {
        row["path"] for row in read_csv(config.output_dir / "vggface2_rejected.csv")
    } == {original, duplicate, conflicting}


def test_processed_liveness_labels_and_subject_overlap(preparation_config):
    config = preparation_config
    root, rows = config.liveness.root, []
    for split, subjects in (("train", [1, 2, 3]), ("test", [3, 4])):
        for subject in subjects:
            for attack, spoof_type, is_spoof in (("live", 0, 0), ("spoof", 1, 1)):
                relative = f"Data/{split}/{subject}/{attack}/face.jpg"
                save_image(root / relative, subject * 10 + is_spoof, size=224)
                (root / relative).with_name("face_BB.txt").write_text(
                    "16 16 128 128 0.99"
                )
                rows.append((relative, subject, split, spoof_type, is_spoof))
    with duckdb.connect(str(root / config.liveness.database)) as con:
        con.execute(
            "CREATE TABLE celeba_labels (path VARCHAR, subject_id INTEGER, "
            "split VARCHAR, spoof_type INTEGER, is_spoof INTEGER)"
        )
        con.executemany("INSERT INTO celeba_labels VALUES (?, ?, ?, ?, ?)", rows)
    summary = LivenessPreparer(config).prepare()
    assert summary["subject_overlap_in_source"] == 1
    assert summary["rejected_images"] == 2
    assert summary["splits"] == {"train": 2, "val": 2, "test": 4}
    rows = read_csv(config.output_dir / "celeba_liveness.csv")
    assert {int(row["label"]) for row in rows if row["attack"] == "live"} == {1}
    assert {int(row["label"]) for row in rows if row["attack"] == "print"} == {0}
    assert all(int(row["x2"]) == 144 for row in rows)
    from src.liveness.data import load_manifests, make_loader

    frame = load_manifests([config.output_dir / "celeba_liveness.csv"])
    loader = make_loader(frame[frame.split == "train"], root, 32, 2.0, 2, 0, False)
    images, labels = next(iter(loader))
    assert images.shape == (2, 3, 32, 32) and sorted(labels.tolist()) == [0, 1]


def test_dfew_folds_preserve_test_clips_and_keep_frames_together(preparation_config):
    config = preparation_config
    source, rows = config.emotion, []
    emotions = ["Happy", "Sad", "Neutral", "Angry", "Surprise", "Disgust", "Fear"]
    clip_id = 0
    for label, emotion in enumerate(emotions, 1):
        for split, count in (("train", 4), ("test", 2)):
            for _ in range(count):
                clip_id += 1
                path = f"Clip/{clip_id:05d}"
                for index in (1, 2):
                    save_image(source.root / path / f"{index}.jpg", clip_id * 2 + index)
                rows.append((1, split, clip_id, label, emotion, path))
    with duckdb.connect(str(source.root / source.database)) as con:
        con.execute(
            "CREATE TABLE dfew_splits (fold INTEGER, split VARCHAR, clip_id INTEGER, "
            "label INTEGER, emotion VARCHAR, rel_path_16f VARCHAR)"
        )
        con.executemany("INSERT INTO dfew_splits VALUES (?, ?, ?, ?, ?, ?)", rows)
        con.execute(
            "CREATE TABLE dfew_clips AS SELECT DISTINCT clip_id FROM dfew_splits"
        )
    for split in ("train", "test"):
        folder = source.root / source.splits_dir / f"{split}(single-labeled)"
        folder.mkdir(parents=True)
        (folder / "set_1.csv").write_text(
            "video_name,label\n"
            + "".join(f"{row[2]},{row[3]}\n" for row in rows if row[1] == split)
        )
    summary = EmotionPreparer(config).prepare()
    assert summary["benchmark_clips"] == 42 and summary["exported_frames"] == 84
    assert summary["folds"][1] == {"train": 21, "val": 7, "test": 14}
    manifest = read_csv(config.output_dir / "dfew_fold1.csv")
    assert {int(row["clip_id"]) for row in manifest if row["split"] == "test"} == {
        row[2] for row in rows if row[1] == "test"
    }
    assert {int(row["label"]) for row in manifest} == set(range(7))
    assert len({row["clip_id"] for row in manifest}) == 42


def test_team_photos_are_gallery_only_and_keep_unknown_sessions(preparation_config):
    config = preparation_config
    for person in ("person_a", "person_b"):
        folder = config.team.root / person
        folder.mkdir()
        Image.new("RGB", (64, 64), color="white").save(folder / "1.jpg")
    summary = TeamPreparer(config).prepare()
    assert summary["images"] == 2 and summary["people"] == 2
    assert summary["status"] == {"no_face": 2}
    assert summary["included_in_training"] is False
    rows = read_csv(config.output_dir / "team_gallery.csv")
    assert {row["role"] for row in rows} == {"gallery"}
    assert {row["session"] for row in rows} == {""}


def test_gallery_rejects_small_background_faces_and_multiple_candidates():
    assert select_gallery_face([(1, 1, 6, 6)], 100, 100, 0.05) == (
        "small_face_review",
        None,
    )
    assert select_gallery_face([(20, 20, 80, 80)], 100, 100, 0.05) == (
        "ready",
        (20, 20, 80, 80),
    )
    assert select_gallery_face([(20, 20, 80, 80), (1, 1, 6, 6)], 100, 100, 0.05) == (
        "multiple_faces",
        None,
    )
