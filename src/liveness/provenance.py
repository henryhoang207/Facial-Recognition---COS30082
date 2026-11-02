"""Portable dataset identity and mandatory leakage checks."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .manifest import COLUMNS


def validate_manifest(frame: pd.DataFrame) -> None:
    if frame.empty or not set(COLUMNS).issubset(frame.columns):
        raise ValueError("manifest is empty or missing required columns")
    if frame[COLUMNS].isna().any().any():
        raise ValueError("manifest contains missing values")
    if frame.path.duplicated().any():
        raise ValueError("duplicate manifest path")
    if not set(frame.split).issubset({"train", "val", "test"}):
        raise ValueError("unknown split")
    if not set(frame.label).issubset({0, 1}):
        raise ValueError("binary labels must be 1=live, 0=spoof")
    if not np.array_equal(frame.label.to_numpy(), (frame.attack == "live").astype(int)):
        raise ValueError("live/spoof labels disagree with attack names")
    boxes = frame[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    if not np.isfinite(boxes).all() or (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("invalid face bbox")
    if (frame.groupby("subject").split.nunique() > 1).any():
        raise ValueError("subjects leak across splits")
    if "sha256" in frame and frame.sha256.notna().all():
        if (frame.groupby("sha256").split.nunique() > 1).any():
            raise ValueError("duplicate image bytes leak across splits")


def checked_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or Path(relative).is_absolute():
        raise ValueError(f"image path escapes data root: {relative}")
    return path


def fingerprint(frame: pd.DataFrame) -> str:
    columns = COLUMNS + ["sha256"]
    records = frame.sort_values("path")[columns].to_dict("records")
    encoded = json.dumps(
        records, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def audit_images(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Read actual bytes, fully decode, and check any previously recorded hash."""
    validate_manifest(frame)
    frame = frame.copy()
    hashes = []
    for row in frame.itertuples(index=False):
        path = checked_path(root, row.path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        previous = getattr(row, "sha256", None)
        if pd.notna(previous) and previous != digest:
            raise ValueError(f"image content changed: {row.path}")
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"image cannot be decoded: {row.path}")
        height, width = image.shape[:2]
        if row.x2 <= 0 or row.y2 <= 0 or row.x1 >= width or row.y1 >= height:
            raise ValueError(f"bbox does not intersect image: {row.path}")
        hashes.append(digest)
    frame["sha256"] = hashes
    validate_manifest(frame)
    return frame


def make_provenance(frame: pd.DataFrame) -> dict:
    validate_manifest(frame)
    if "sha256" not in frame or frame.sha256.isna().any():
        raise ValueError("provenance requires audited image hashes")
    return {
        "version": 1,
        "manifest_sha256": fingerprint(frame),
        "splits": {
            split: {
                "subjects": sorted(part.subject.unique().tolist()),
                "hashes": sorted(part.sha256.unique().tolist()),
                "rows": len(part),
            }
            for split, part in frame.groupby("split", sort=True)
        },
    }


def verify_provenance(frame: pd.DataFrame, provenance: dict | None, split: str) -> None:
    if not provenance or provenance.get("version") != 1:
        raise ValueError(
            "checkpoint lacks verifiable dataset provenance; retrain/export it"
        )
    validate_manifest(frame)
    if fingerprint(frame) != provenance["manifest_sha256"]:
        raise ValueError("manifest/content differs from checkpoint provenance")
    if split not in {"val", "test"} or split not in provenance["splits"]:
        raise ValueError("evaluation requires a recorded val or test split")
    held_out = frame[frame.split == split]
    training = provenance["splits"]["train"]
    if set(held_out.subject) & set(training["subjects"]):
        raise ValueError("evaluation subjects also used for training")
    if set(held_out.sha256) & set(training["hashes"]):
        raise ValueError("evaluation image bytes also used for training")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
