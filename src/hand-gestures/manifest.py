"""Index ASLDynamic recordings and split them before sampling landmarks."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from string import ascii_uppercase

import cv2

from src.datasets.catalog import stable_rng, write_csv, write_json
from src.datasets.config import PreparationConfig

STATIC_CLASSES = tuple(letter for letter in ascii_uppercase if letter not in "JZ")
DYNAMIC_CLASSES = ("J", "Z", "HELLO", "SORRY", "THANK_YOU", "YES", "NO", "OTHER")
SOURCE_LABELS = {label: label for label in (*STATIC_CLASSES, *DYNAMIC_CLASSES[:-1])}
SOURCE_LABELS["THANKYOU"] = SOURCE_LABELS.pop("THANK_YOU")
SPLITS = ("train", "val", "test")
COLUMNS = (
    "clip_id",
    "relative_path",
    "source_label",
    "label",
    "sha256",
    "group_id",
    "signer_id",
    "session_id",
    "orientation",
    "start_sec",
    "end_sec",
    "other_intervals",
    "split",
    "status",
    "canonical_id",
    "fps",
    "duration_sec",
    "notes",
)


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _review_intervals(review: dict, duration: float) -> tuple[object, object, str]:
    start, end = review.get("start_sec"), review.get("end_sec")
    if start or end:
        start, end = float(start or 0), float(end or duration)
        if not 0 <= start < end <= duration + 0.001:
            raise ValueError("invalid positive interval")
    negatives = json.loads(review.get("other_intervals") or "[]")
    if not isinstance(negatives, list):
        raise ValueError("OTHER intervals must be a list")
    if negatives and not (review.get("start_sec") or review.get("end_sec")):
        raise ValueError("OTHER intervals require an annotated positive interval")
    for left, right in negatives:
        if not 0 <= left < right <= duration + 0.001:
            raise ValueError("invalid OTHER interval")
        if left < end and right > start:
            raise ValueError("OTHER interval overlaps the positive interval")
    return start, end, json.dumps(negatives)


def _record(path: Path, root: Path, review: dict) -> dict:
    relative = path.relative_to(root)
    clip_id = relative.with_suffix("").as_posix()
    source_label = relative.parts[0]
    if source_label not in SOURCE_LABELS:
        raise ValueError(f"{clip_id}: unknown ASL class directory")
    with path.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        valid = (
            cap.read()[0]
            and math.isfinite(fps)
            and fps > 0
            and math.isfinite(frames)
            and frames > 0
        )
    finally:
        cap.release()
    duration = frames / fps if valid else 0.0
    orientation = int(review.get("orientation") or 0)
    if orientation not in (0, 90, 180, 270):
        raise ValueError(f"{clip_id}: unsupported clockwise orientation")
    try:
        start, end, negatives = _review_intervals(review, duration)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{clip_id}: {exc}") from exc
    excluded = not valid or (review.get("exclude") or "").lower() in {
        "1",
        "true",
        "yes",
    }
    reason = review.get("notes") or ""
    if not valid:
        reason = f"Unreadable AVI header or first frame. {reason}".strip()
    return dict(
        clip_id=clip_id,
        relative_path=relative.as_posix(),
        source_label=source_label,
        label=SOURCE_LABELS[source_label],
        sha256=sha256,
        signer_id=review.get("signer_id") or "",
        session_id=review.get("session_id") or "",
        orientation=orientation,
        start_sec=start,
        end_sec=end,
        other_intervals=negatives,
        split="",
        status="excluded" if excluded else "ready",
        canonical_id=clip_id,
        fps=fps if valid else "",
        duration_sec=duration,
        notes=reason,
    )


def _assign_splits(originals: list[dict], seed: int) -> None:
    groups = defaultdict(list)
    for row in originals:
        signer, session = row["signer_id"], row["session_id"]
        group = (
            f"signer:{signer}"
            if signer
            else f"session:{session}"
            if session
            else row["sha256"]
        )
        row["group_id"] = group
        if row["status"] == "ready":
            groups[group].append(row)
    totals = Counter(row["label"] for group in groups.values() for row in group)
    targets = {
        label: dict(zip(SPLITS, (count - 2 * holdout, holdout, holdout), strict=True))
        for label, count in totals.items()
        for holdout in (round(count * 0.2),)
    }
    order = sorted(groups)
    stable_rng(seed, "asl_dynamic").shuffle(order)
    for group in sorted(order, key=lambda key: -len(groups[key])):
        labels = Counter(row["label"] for row in groups[group])
        split = max(
            SPLITS,
            key=lambda name: sum(
                min(targets[label][name], count) for label, count in labels.items()
            ),
        )
        for row in groups[group]:
            row["split"] = split
            targets[row["label"]][split] -= 1


def _summary(records: list[dict]) -> dict:
    return {
        "files": len(records),
        "distinct_hashes": len({row["sha256"] for row in records}),
        "statuses": dict(Counter(row["status"] for row in records)),
        "support": {
            split: dict(
                Counter(
                    row["label"]
                    for row in records
                    if row["split"] == split and row["status"] == "ready"
                )
            )
            for split in SPLITS
        },
        "grouping": (
            "recording holdout"
            if all(row["group_id"] == row["sha256"] for row in records)
            else "known signer/session where provided, otherwise recording"
        ),
    }


class GesturePreparer:
    def __init__(self, config: PreparationConfig):
        self.config = config

    def prepare(self) -> dict:
        source = self.config.hand_gestures
        if source is None or not source.root.is_dir():
            raise ValueError("hand_gestures requires an ASLDynamic source directory")
        directory = self.config.output_dir / "hand_gestures"
        if directory.is_relative_to(source.root):
            raise ValueError("gesture manifests must live outside the source dataset")
        review_rows = read_rows(source.review) if source.review else []
        reviews = {row["clip_id"]: row for row in review_rows}
        if len(reviews) != len(review_rows):
            raise ValueError("duplicate clip ID in gesture review table")
        records = []
        by_hash = defaultdict(list)
        for path in sorted(source.root.glob("*/*.avi")):
            row = _record(
                path,
                source.root,
                reviews.get(
                    path.relative_to(source.root).with_suffix("").as_posix(), {}
                ),
            )
            records.append(row)
            by_hash[row["sha256"]].append(row)
        if not records:
            raise ValueError(f"no AVI recordings under {source.root}")
        unknown_reviews = reviews.keys() - {row["clip_id"] for row in records}
        if unknown_reviews:
            raise ValueError(
                f"gesture review names unknown clips: {sorted(unknown_reviews)}"
            )
        originals = []
        for group in by_hash.values():
            if len({row["label"] for row in group}) != 1:
                raise ValueError(
                    f"same AVI content has conflicting labels: {group[0]['clip_id']}"
                )
            original = min(
                group,
                key=lambda row: (
                    row["status"] != "ready",
                    row["clip_id"] not in reviews,
                ),
            )
            originals.append(original)
            for row in group:
                if row is not original:
                    row.update(status="duplicate", canonical_id=original["clip_id"])
        _assign_splits(originals, self.config.seed)
        for original in originals:
            for row in by_hash[original["sha256"]]:
                row.update(split=original["split"], group_id=original["group_id"])
        report = validate_gesture_rows(records)
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(directory / "clips.csv", records, list(COLUMNS))
        write_json(
            directory / "labels.json",
            {"static": STATIC_CLASSES, "dynamic": DYNAMIC_CLASSES},
        )
        write_json(directory / "inventory.json", report)
        return report


def validate_gesture_rows(records: list[dict]) -> dict:
    by_id = {row["clip_id"]: row for row in records}
    if not records or len(by_id) != len(records):
        raise ValueError("expected unique gesture clip IDs")
    assignments = {}
    originals = set()
    for row in records:
        clip_id, status = row["clip_id"], row["status"]
        original = by_id.get(row["canonical_id"])
        if original is None or any(
            row[key] != original[key]
            for key in ("split", "label", "sha256", "group_id")
        ):
            raise ValueError(f"{clip_id}: invalid duplicate association")
        if status == "duplicate":
            if original["status"] not in {"ready", "excluded"}:
                raise ValueError(f"{clip_id}: duplicate lacks a canonical recording")
            continue
        if original is not row or row["sha256"] in originals:
            raise ValueError(f"{clip_id}: duplicate canonical recording")
        originals.add(row["sha256"])
        if status == "excluded" and not row["split"]:
            continue
        if status != "ready" or row["split"] not in SPLITS:
            raise ValueError(f"{clip_id}: invalid gesture status or split")
        group, split = row["group_id"], row["split"]
        if assignments.setdefault(group, split) != split:
            raise ValueError(f"{clip_id}: group spans splits")
    return _summary(records)
