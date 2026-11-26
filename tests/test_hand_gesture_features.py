"""Validate geometric invariants and shared clip-level data for ASL models."""

import csv
import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

data = importlib.import_module("src.hand-gestures.data")
features = importlib.import_module("src.hand-gestures.features")
camera = importlib.import_module("src.hand-gestures.test-cam")


def _clip():
    times = np.arange(12) / 30
    pose = np.random.default_rng(7).uniform(0.1, 0.3, (21, 3)).astype(np.float32)
    hand = np.broadcast_to(pose, (len(times), 21, 3)).copy()
    hand[:, :, 0] += times[:, None]
    anchors = np.broadcast_to(
        np.float32([[0.3, 0.2], [0.3, 0.3], [0.1, 0.5], [0.5, 0.5]]),
        (len(times), 4, 2),
    ).copy()
    return features.LandmarkClip(
        times, hand, np.full(len(times), "Right"), anchors, 640, 480
    )


def _sequence(clip, include_body=True):
    return features.sequence_features(
        clip, length=32, min_coverage=0.8, max_gap=0.1, include_body=include_body
    )


def test_camera_window_waits_for_motion_and_reuses_training_features():
    clip = _clip()
    settings = dict(
        sequence_length=32,
        include_body=False,
        max_hand_gap_sec=0.1,
        min_hand_coverage=0.8,
    )
    assert camera.camera_features(clip, "mlp_gru", settings, 2.0) is None
    sequence = _sequence(clip, include_body=False)
    np.testing.assert_array_equal(
        camera.camera_features(clip, "mlp_gru", settings, 0.5), sequence[None]
    )
    np.testing.assert_array_equal(
        camera.camera_features(clip, "motion_xgb", settings, 0.5),
        features.pool_motion(sequence, False)[None],
    )
    clip.hand[4:9] = np.nan
    assert camera.camera_features(clip, "mlp_gru", settings, 0.5) is None
    assert camera.camera_features(clip, "static_xgb", settings, 2.0).shape == (
        1,
        len(features.STATIC_COLUMNS),
    )


@pytest.mark.parametrize("model", ["mlp", "xgb"])
def test_camera_alphabet_passes_one_aspect_corrected_hand(model):
    clip = _clip()
    expected = clip.hand[-1:] * np.float32([1, 480 / 640, 1])
    np.testing.assert_array_equal(
        camera.camera_features(clip, model, {}, 2.0), expected
    )


def test_geometry_retains_motion_and_mirrors_full_sequence():
    clip = _clip()
    flipped = clip.mirrored()
    assert np.all(flipped.handedness == "Left")
    np.testing.assert_allclose(flipped.anchors[:, 2, 0], 1 - clip.anchors[:, 3, 0])
    np.testing.assert_allclose(flipped.mirrored().hand, clip.hand, atol=1e-7)
    np.testing.assert_allclose(flipped.mirrored().anchors, clip.anchors, atol=1e-7)

    sequence = _sequence(clip)
    mirrored = _sequence(flipped)
    columns = features.sequence_columns(True)
    np.testing.assert_allclose(sequence[0, :63], sequence[-1, :63], atol=1e-5)
    assert sequence[-1, columns.index("displacement_wrist_x")] > 0
    for name in ("displacement_wrist_x", "velocity_wrist_x", "relative_nose_x"):
        index = columns.index(name)
        np.testing.assert_allclose(
            sequence[:, index], -mirrored[:, index], atol=1e-5, rtol=1e-5
        )
    assert len(features.pool_motion(sequence, True)) == len(
        features.motion_columns(True)
    )


def test_quality_policy_handles_missing_anchors_gaps_and_a_hand_switch():
    clip = _clip()
    clip.hand[5] = np.nan
    clip.anchors[5, 2] = np.nan
    values = _sequence(clip)
    columns = features.sequence_columns(True)
    assert 0 in values[:, columns.index("hand_observed")]
    offset = columns.index("relative_nose_x")
    missing = values[:, columns.index("anchor_valid_nose")] == 0
    assert missing.any()
    assert (values[missing, offset] == 0).all()
    clip.hand[6:8] = np.nan
    with pytest.raises(ValueError, match="coverage|hand gap"):
        _sequence(clip)

    clip = _clip()
    clip.handedness[6:] = "Left"
    with pytest.raises(ValueError, match="hand changed"):
        _sequence(clip)

    clip = _clip()
    clip.hand[:, 9] = clip.hand[:, 0]
    with pytest.raises(ValueError, match="coverage"):
        _sequence(clip)


def test_motion_models_share_windows_and_augment_only_training(tmp_path):
    clip = _clip()
    rows = []
    for split in ("train", "val"):
        for label in ("I", "J"):
            digest = hex(len(rows) + 1)[2:] * 64
            rows.append(
                {
                    "clip_id": f"{split}/{label}",
                    "sha256": digest,
                    "split": split,
                    "label": label,
                    "status": "ready",
                    "orientation": 0,
                    "start_sec": "",
                    "end_sec": "",
                }
            )
            np.savez(
                tmp_path / f"{digest}.npz",
                timestamps=clip.timestamps,
                hand=clip.hand,
                handedness=clip.handedness,
                anchors=clip.anchors,
                width=clip.width,
                height=clip.height,
                sha256=digest,
                rotation=0,
                extractor="fixture",
            )
    manifest = tmp_path / "clips.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    config = SimpleNamespace(
        manifest=manifest,
        cache_dir=tmp_path,
        sequence_length=32,
        include_body=False,
        flip_train=True,
        static_frames=4,
        min_hand_coverage=0.8,
        max_hand_gap_sec=0.1,
    )
    gru = data.load_samples(config, "train", "mlp_gru")
    xgb = data.load_samples(config, "train", "motion_xgb")
    assert gru.window_ids == xgb.window_ids
    assert gru.inputs.shape == (4, 32, len(gru.columns))
    assert xgb.inputs.shape == (4, len(xgb.columns))
    assert gru.labels.tolist() == [gru.classes.index("OTHER")] * 2 + [0, 0]
    assert gru.weights.tolist() == [0.5] * 4
    assert gru.handedness == ("Right", "Left", "Right", "Left")
    val = data.load_samples(config, "val", "mlp_gru")
    assert val.clip_ids == ("val/I", "val/J")
    static = data.load_samples(config, "val", "static_xgb")
    assert static.clip_ids == ("val/I",) * 4
    assert static.weights.tolist() == [0.25] * 4

    alternate = replace(clip, timestamps=clip.timestamps * 2)
    normal, slow = _sequence(clip), _sequence(alternate)
    index = features.sequence_columns(True).index("velocity_wrist_x")
    np.testing.assert_allclose(normal[:, index], 2 * slow[:, index])
