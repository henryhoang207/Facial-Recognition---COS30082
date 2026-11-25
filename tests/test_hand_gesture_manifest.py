"""Verify recording groups and aliases stay together across splits."""

from importlib import import_module

import pytest

manifest = import_module("src.hand-gestures.manifest")


def _record(index, label="YES", signer=""):
    return {
        "clip_id": str(index),
        "canonical_id": str(index),
        "label": label,
        "sha256": str(index),
        "signer_id": signer,
        "session_id": "",
        "status": "ready",
    }


def test_recording_holdout_and_duplicate_validation():
    rows = [_record(index) for index in range(5)]
    manifest._assign_splits(rows, 42)
    alias = {**rows[0], "clip_id": "alias", "status": "duplicate"}
    report = manifest.validate_gesture_rows([*rows, alias])
    assert [report["support"][split]["YES"] for split in manifest.SPLITS] == [3, 1, 1]
    assert report["distinct_hashes"] == 5
    alias["split"] = "test" if rows[0]["split"] != "test" else "val"
    with pytest.raises(ValueError, match="duplicate association"):
        manifest.validate_gesture_rows([*rows, alias])


def test_known_signer_stays_together_across_labels():
    rows = [
        _record(1, "YES", "person"),
        _record(2, "NO", "person"),
        *[_record(index) for index in range(3, 7)],
    ]
    manifest._assign_splits(rows, 42)
    assert rows[0]["split"] == rows[1]["split"]
    manifest.validate_gesture_rows(rows)
