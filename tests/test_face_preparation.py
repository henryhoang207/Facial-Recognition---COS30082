"""Conservative bbox selection, resumable scans and independent manifest outputs."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.datasets.catalog import write_csv
from src.datasets.config import PreparationConfig
from src.datasets.validate import ManifestValidator
from src.recognition.preprocessing.config import DetectorConfig, FacePreparationConfig
from src.recognition.preprocessing.detector import (
    FaceCandidate,
    YuNetDetector,
    inspect_face,
)
from src.recognition.preprocessing.prepare import FaceManifestPreparer, read_rows
from src.training.config import read_yaml

ROOT = Path(__file__).resolve().parents[1]


def detector_config(tmp_path):
    raw = read_yaml(ROOT / "configs/recognition/face-preparation.yaml")["detector"]
    model = tmp_path / "fake-model.onnx"
    model.write_bytes(b"fixture detector")
    raw.update(
        model=model,
        sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        min_face_pixels=3,
        max_center_distance=0.2,
    )
    return DetectorConfig.model_validate(raw)


def candidate(box):
    return FaceCandidate(tuple(box), 0.99, ((5.0, 5.0),) * 5)


def test_policy_never_selects_largest_face_from_multiple_people(tmp_path):
    cfg = detector_config(tmp_path)
    frame = np.zeros((100, 100, 3), np.uint8)
    assert inspect_face(frame, [], cfg)["status"] == "no_face"
    assert (
        inspect_face(frame, [candidate([20, 20, 80, 80])], cfg)["status"] == "accepted"
    )
    multiple = inspect_face(
        frame, [candidate([20, 20, 80, 80]), candidate([0, 0, 5, 5])], cfg
    )
    assert multiple["status"] == "multiple_faces" and multiple["box"] is None
    assert (
        inspect_face(frame, [candidate([49, 49, 51, 51])], cfg)["status"]
        == "small_face"
    )
    assert (
        inspect_face(frame, [candidate([0, 35, 30, 65])], cfg)["status"] == "off_center"
    )


def test_letterbox_boxes_are_mapped_back_to_original_pixels(tmp_path, monkeypatch):
    face = np.array([[80, 112, 160, 96, *([96, 128] * 5), 0.99]], dtype=np.float32)
    fake = SimpleNamespace(detect=lambda image: (None, face))
    monkeypatch.setattr(
        cv2, "FaceDetectorYN", SimpleNamespace(create=lambda *args: fake)
    )
    detector = YuNetDetector(detector_config(tmp_path))
    result = detector.candidates(np.zeros((100, 200, 3), np.uint8))[0]
    assert result.box == (50, 20, 150, 80)
    assert result.landmarks[0] == pytest.approx((60, 30))


def fixture_config(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    rng = np.random.default_rng(42)
    classification, verification = [], []
    for identity in ("a", "b"):
        for split in ("train", "val", "test"):
            for index in range(3):
                path = f"{identity}-{split}-{index}.png"
                image = rng.integers(0, 255, (20, 20, 3), np.uint8)
                image[0, 0, 0] = 60 if index == 2 else 0
                cv2.imwrite(str(root / path), image)
                classification.append(
                    {
                        "path": path,
                        "identity": identity,
                        "split": split,
                        "label": 0 if identity == "a" else 1,
                        "sha256": hashlib.sha256(
                            (root / path).read_bytes()
                        ).hexdigest(),
                        "group": path,
                    }
                )
    for identity, split in (("c", "val"), ("d", "val"), ("e", "test"), ("f", "test")):
        for index in range(3):
            path = f"{identity}-{index}.png"
            image = rng.integers(0, 255, (20, 20, 3), np.uint8)
            image[0, 0, 0] = 0
            cv2.imwrite(str(root / path), image)
            verification.append(
                {
                    "path": path,
                    "identity": identity,
                    "split": split,
                    "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                }
            )
    source_c, source_v = tmp_path / "classification.csv", tmp_path / "verification.csv"
    write_csv(source_c, classification, list(classification[0]))
    write_csv(source_v, verification, list(verification[0]))
    return FacePreparationConfig(
        root=root,
        classification=source_c,
        verification=source_v,
        output_dir=tmp_path / "output",
        detector=detector_config(tmp_path),
        seed=42,
        workers=2,
        chunk_size=4,
        positive_pairs=2,
        negative_pairs=2,
        progress=False,
    )


class FixtureDetector:
    calls = []

    def __init__(self, config):
        pass

    def candidates(self, frame):
        self.calls.append(1)
        return (
            [candidate([2, 2, 18, 18]), candidate([0, 0, 4, 4])]
            if frame[0, 0, 0] == 60
            else [candidate([2, 2, 18, 18])]
        )


def test_scan_resumes_and_publishes_bbox_manifests_without_changing_sources(tmp_path):
    cfg = fixture_config(tmp_path)
    before = {
        path: path.read_bytes() for path in (cfg.classification, cfg.verification)
    }
    partial = FaceManifestPreparer(cfg, detector_factory=FixtureDetector).run(
        scan_limit=3
    )
    assert partial["complete"] is False
    assert not (cfg.output_dir / "current.json").exists()
    report = FaceManifestPreparer(cfg, detector_factory=FixtureDetector).run()
    assert report["reused"] == 3 and report["source_images"] == 30
    assert report["classification_images"] == 12 and report["review_images"] == 6
    assert report["classification_splits"] == {"train": 4, "val": 4, "test": 4}
    directory = Path(report["output_dir"])
    rows = read_rows(directory / "vggface2_classification.csv")
    assert all(
        [row[k] for k in ("x1", "y1", "x2", "y2")] == ["2", "2", "18", "18"]
        for row in rows
    )
    assert all(row["detector_status"] == "accepted" for row in rows)
    for split in ("val", "test"):
        pairs = read_rows(directory / f"verification_pairs_{split}.csv")
        assert sorted(row["label"] for row in pairs) == ["0", "0", "1", "1"]
    original_exports = {p.name: p.read_bytes() for p in directory.glob("*.csv")}

    def no_detection(config):
        raise AssertionError("unchanged images must reuse detection cache")

    repeated = FaceManifestPreparer(cfg, detector_factory=no_detection).run()
    assert repeated["reused"] == 30 and repeated["detected_now"] == 0
    assert original_exports == {p.name: p.read_bytes() for p in directory.glob("*.csv")}
    assert before == {p: p.read_bytes() for p in before}


def test_manual_override_can_recover_a_reviewed_target_without_redetecting(tmp_path):
    cfg = fixture_config(tmp_path)
    initial = FaceManifestPreparer(cfg, detector_factory=FixtureDetector).run()
    overrides = tmp_path / "overrides.csv"
    write_csv(
        overrides,
        [
            {
                "path": "a-train-2.png",
                "decision": "accept",
                "x1": 3,
                "y1": 3,
                "x2": 17,
                "y2": 17,
            }
        ],
        ["path", "decision", "x1", "y1", "x2", "y2"],
    )
    cfg = cfg.model_copy(update={"overrides": overrides})
    report = FaceManifestPreparer(cfg, detector_factory=FixtureDetector).run()
    assert report["reused"] == 30 and report["classification_images"] == 13
    assert initial["output_dir"] != report["output_dir"]
    rows = read_rows(Path(report["output_dir"]) / "vggface2_classification.csv")
    selected = next(row for row in rows if row["path"] == "a-train-2.png")
    assert selected["detector_status"] == "manual_accept" and selected["x1"] == "3"


def test_independent_validation_rejects_an_out_of_bounds_generated_box(tmp_path):
    cfg = fixture_config(tmp_path)
    report = FaceManifestPreparer(cfg, detector_factory=FixtureDetector).run()
    directory = Path(report["output_dir"])
    raw = read_yaml(ROOT / "configs/datasets.yaml")
    raw["output_dir"] = str(directory)
    validator = ManifestValidator(PreparationConfig.model_validate(raw))
    assert validator.recognition()["bbox_rows"] == 24
    boxes = read_rows(directory / "bbox_manifest.csv")
    boxes[0]["x2"] = "21"
    write_csv(directory / "bbox_manifest.csv", boxes, list(boxes[0]))
    with pytest.raises(ValueError, match="bbox outside original image"):
        validator.recognition()
