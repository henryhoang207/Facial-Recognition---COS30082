"""Check attendance features, dataset separation, and webcam evaluation."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.recognition.attendance.config import (
    AttendanceConfig,
    VidTIMITConfig,
    YouTubeFacesConfig,
)
from src.recognition.attendance.evaluate_webcam import summarize
from src.recognition.attendance.features import attempt_features, features_from_scores
from src.recognition.attendance.prepare import AttemptPreparer
from src.recognition.attendance.prepare_vidtimit import VidTIMITPreparer
from src.recognition.attendance.prepare_youtube_faces import YouTubeFacesPreparer
from src.recognition.attendance.train import (
    FixedGalleryValidation,
    select_threshold,
    source_sampling_weights,
)
from src.recognition.classification.config import PreprocessConfig
from src.training.config import read_yaml


def test_window_features_follow_the_same_candidate_across_frames():
    winner, features = attempt_features(np.array([[1, 0], [1, 0], [0, 1]]), np.eye(2))
    assert winner == 0
    assert features == pytest.approx([2 / 3, 0, 2**0.5 / 3, 1 / 3, 2 / 3])


def test_top2_gap_uses_the_same_runner_up_across_the_whole_window():
    scores = np.array([[0.8, 0.7, 0.1], [0.8, 0.1, 0.7]])
    winner, original = features_from_scores(scores)
    new_winner, extended = features_from_scores(scores, include_top2_gap=True)
    assert winner == new_winner == 0
    assert np.array_equal(original, extended[:-1])
    assert extended[3] == pytest.approx(0.1)
    assert extended[-1] == pytest.approx(0.4)


def test_combined_threshold_respects_the_target_on_each_source():
    labels = np.array([1, 0, 1, 1, 0])
    scores = np.array([0.9, 0.2, 0.95, 0.7, 0.8])
    masks = {"first": np.array([True, True, False, False, False])}
    masks["second"] = ~masks["first"]
    threshold = select_threshold(labels, scores, 0.01, masks)
    assert (scores >= threshold).tolist() == [True, False, True, False, False]


def test_source_sampler_shares_weight_between_gallery_variants(tmp_path):
    path = tmp_path / "attempts.csv"
    path.write_text(
        "split,identity,clip,frame_paths\n"
        "train,alice,one,window-1\n"
        "train,alice,one,window-1\n"
        "train,alice,one,window-2\n"
        "train,alice,two,window-3\n"
        "train,bob,one,window-4\n"
        "dev,charlie,one,window-5\n",
        encoding="utf-8",
    )
    assert source_sampling_weights(path) == [1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2]


def test_fixed_gallery_validation_calibrates_on_each_dev_stratum(tmp_path):
    path = tmp_path / "attempts.csv"
    path.write_text(
        "split,source,gallery_mode,kind,score\n"
        "dev,video,random,correct,4\n"
        "dev,video,random,unknown,-2\n"
        "dev,video,hard,correct,0\n"
        "dev,video,hard,wrong_known,3\n"
        "dev,video,hard,unknown,2\n"
        "test,video,hard,unknown,5\n",
        encoding="utf-8",
    )
    device = torch.device("cpu")
    validate = FixedGalleryValidation(path, ("score",), ("video",), 0.01, device)
    metrics = validate(lambda features: features[:, 0], device)

    assert metrics["val_gallery_threshold"] > torch.sigmoid(torch.tensor(2.0)).item()
    assert metrics["val_gallery_correct_rate"] == 0.5
    assert metrics["val_gallery_wrong_rate"] == 0.25


def test_youtube_faces_crops_xy_boxes_in_wide_clips(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = YouTubeFacesConfig.model_validate(
        read_yaml(root / "configs/recognition/attendance-youtube-faces.yaml")
    ).model_copy(update={"root": tmp_path.resolve()})
    settings = PreprocessConfig.model_validate(
        read_yaml(root / "configs/recognition/model_a.yaml")["preprocess"]
    )
    frames = np.zeros((20, 60, 3, 1), dtype=np.uint8)
    frames[2:14, 34:46, 0, 0] = 255
    points = np.array([[34, 2], [34, 14], [46, 2], [46, 14]], dtype=float)
    np.savez(tmp_path / "clip.npz", colorImages=frames, boundingBox=points[..., None])

    [(key, crop)] = YouTubeFacesPreparer(cfg).clip_crops(("clip.npz", [0]), settings)

    assert key == "clip.npz#0"
    assert crop.shape == (settings.size, settings.size, 3)
    assert np.all(crop[..., 0] == 255)
    assert np.all(crop[..., 1:] == 0)


def test_sampling_keeps_query_images_separate_and_creates_unknowns():
    config = AttendanceConfig.model_validate(
        read_yaml(
            Path(__file__).resolve().parents[1] / "configs/recognition/attendance.yaml"
        )
    )
    pools = {
        split: {
            f"{split}-{identity}": [f"{split}/{identity}/{i}" for i in range(70)]
            for identity in range(7)
        }
        for split in ("train", "dev", "test")
    }
    enrollment, attempts = AttemptPreparer(config).sample(pools)
    enrolled_paths = {path for values in enrollment.values() for path in values}
    probe_paths = [path for row in attempts for path in row["frame_paths"]]
    assert len(attempts) == 3 * 7 * 12
    assert len(probe_paths) == len(set(probe_paths))
    assert not enrolled_paths.intersection(probe_paths)
    for row in attempts:
        assert len(set(row["gallery"])) == 6
        assert set(row["gallery"]) <= pools[row["split"]].keys()
        assert (row["identity"] in row["gallery"]) == bool(row["is_enrolled"])


def test_vidtimit_attempts_keep_recordings_and_identities_separate():
    config = VidTIMITConfig.model_validate(
        read_yaml(
            Path(__file__).resolve().parents[1]
            / "configs/recognition/attendance-vidtimit.yaml"
        )
    )
    people = {
        f"person-{person}": {
            (session, f"clip-{clip}"): [
                f"{person}/{session}/{clip}/{frame}" for frame in range(10)
            ]
            for session in (1, 2, 3)
            for clip in range(3)
        }
        for person in range(43)
    }
    vectors = {
        path for clips in people.values() for paths in clips.values() for path in paths
    }
    pools, enrollment, attempts, skipped = VidTIMITPreparer(config).sample_sessions(
        people, vectors
    )
    assert not skipped
    assert [len(pools[split]) for split in ("train", "dev", "test")] == [24, 8, 11]
    assert sum(map(len, pools.values())) == len(set().union(*pools.values()))
    assert all("/1/" in path for paths in enrollment.values() for path in paths)
    for attempt in attempts:
        gallery = set(attempt["gallery"])
        assert len(gallery) == 6
        assert gallery <= set(pools[attempt["split"]])
        assert (attempt["identity"] in gallery) == bool(attempt["is_enrolled"])
        frames = people[attempt["identity"]][attempt["session"], attempt["clip"]]
        offset = frames.index(attempt["frame_paths"][0])
        assert attempt["session"] in (2, 3)
        assert attempt["frame_paths"] == frames[offset : offset + 5]


def test_vidtimit_mining_uses_unused_windows_and_closest_distractors():
    config = VidTIMITConfig.model_validate(
        read_yaml(
            Path(__file__).resolve().parents[1]
            / "configs/recognition/attendance-vidtimit.yaml"
        )
    )
    preparer = VidTIMITPreparer(config)
    names = list("abcdefgh")
    enrollment = {name: [name] for name in names}
    vectors = {name: np.array([-1, 0]) for name in names}
    vectors["b"] = np.array([1, 0])
    already_used, probe = list("12345"), list("67890")
    vectors.update({path: vectors["b"] for path in already_used + probe})
    people = {name: {} for name in names}
    people["a"] = {(2, "head2"): already_used + probe}

    mined = list(
        preparer.mine_hard_attempts(
            people, names, enrollment, [{"frame_paths": already_used}], vectors, 0.55
        )
    )

    assert len(mined) == 2
    assert all(row["split"] == "train" and row["frame_paths"] == probe for row in mined)
    assert mined[0]["gallery"] == list("bcdefg")
    assert mined[1]["gallery"] == list("abcdef")
    assert [row["is_enrolled"] for row in mined] == [0, 1]


def test_webcam_report_counts_false_accepts_by_attempt_type(tmp_path):
    path = tmp_path / "attempts.csv"
    common = dict(
        encoder_sha256="encoder",
        mlp_sha256="mlp",
        gallery_sha256="gallery",
        frame_count=5,
        gallery_size=6,
        mlp_threshold=0.5,
        cosine_threshold=0.4,
        minimum_consensus=0.8,
        candidate="alice",
        session="session-01",
    )
    records = [
        dict(
            common,
            attempt_id=index,
            subject=subject,
            kind=kind,
            label=label,
            is_enrolled=enrolled,
            mlp_accepted=mlp,
            cosine_accepted=cosine,
        )
        for index, (subject, kind, label, enrolled, mlp, cosine) in enumerate(
            [
                ("alice", "correct", 1, 1, "true", "true"),
                ("bob", "wrong_known", 0, 1, "true", "false"),
                ("visitor-01", "unknown", 0, 0, "false", "true"),
            ]
        )
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    report = summarize([path])
    assert report["attempts"] == 3
    assert report["methods"]["mlp"]["correct_accept_rate_enrolled"] == 0.5
    assert report["methods"]["mlp"]["wrong_known_accept_rate_enrolled"] == 0.5
    assert report["methods"]["mlp"]["unknown_accept_rate"] == 0.0
    assert report["methods"]["cosine"]["unknown_accept_rate"] == 1.0
    with pytest.raises(ValueError, match="duplicate attempt"):
        summarize([path, path])
