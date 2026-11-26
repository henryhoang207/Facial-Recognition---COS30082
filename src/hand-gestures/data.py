"""Build deterministic static frames and motion windows from frozen clip splits."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from string import hexdigits
from zipfile import BadZipFile

import numpy as np

from .config import GestureConfig
from .features import (
    STATIC_COLUMNS,
    LandmarkClip,
    local_pose,
    motion_columns,
    pool_motion,
    sequence_columns,
    sequence_features,
    track_spans,
)
from .manifest import DYNAMIC_CLASSES, SPLITS, STATIC_CLASSES, read_rows


@dataclass(frozen=True, slots=True)
class Samples:
    inputs: np.ndarray
    labels: np.ndarray
    clip_ids: tuple[str, ...]
    window_ids: tuple[str, ...]
    weights: np.ndarray
    classes: tuple[str, ...]
    columns: tuple[str, ...]
    rejected: dict[str, str]
    handedness: tuple[str, ...]
    anchor_coverage: np.ndarray
    segments: tuple[str, ...]
    extractor: str


def _read_cache(directory: Path, row: dict[str, str]) -> tuple[LandmarkClip, str]:
    digest = row["sha256"]
    if len(digest) != 64 or any(letter not in hexdigits for letter in digest):
        raise ValueError("invalid source hash")
    path = directory / f"{digest}.npz"
    with np.load(path, allow_pickle=False) as cache:
        if cache["sha256"].item() != digest or cache["rotation"].item() != int(
            row["orientation"]
        ):
            raise ValueError("landmark cache does not match source hash and rotation")
        clip = LandmarkClip(
            cache["timestamps"],
            cache["hand"],
            cache["handedness"],
            cache["anchors"],
            int(cache["width"].item()),
            int(cache["height"].item()),
        )
        return clip, str(cache["extractor"].item())


def _read_split(path: Path, split: str) -> list[dict[str, str]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown ASL split {split!r}")
    rows = [
        row
        for row in read_rows(path)
        if row["status"] == "ready" and row["split"] == split
    ]
    ids = {row["clip_id"] for row in rows}
    hashes = {row["sha256"] for row in rows}
    if len(ids) != len(rows) or len(hashes) != len(rows):
        raise ValueError(f"Duplicate ready clips in {split} split")
    return rows


def _interval(clip: LandmarkClip, start: float, end: float) -> LandmarkClip:
    if not (np.isfinite(start) and np.isfinite(end) and 0 <= start < end):
        raise ValueError("invalid review interval")
    if not len(clip.timestamps):
        raise ValueError("the landmark cache contains no frames")
    # A rounded review endpoint may fall between the last frame and container end.
    if end > clip.timestamps[-1] + 0.1:
        raise ValueError("review interval extends beyond the recording")
    mask = (clip.timestamps >= start - 1e-6) & (clip.timestamps <= end + 1e-6)
    return clip.select(mask)


def _longest_track(clip: LandmarkClip, max_gap: float) -> LandmarkClip:
    spans = track_spans(clip, max_gap)
    if not spans:
        raise ValueError("no valid hand observations")
    span = max(
        spans,
        key=lambda item: clip.timestamps[item.stop - 1] - clip.timestamps[item.start],
    )
    return clip.select(span)


def _static_frames(clip: LandmarkClip, count: int, max_gap: float) -> LandmarkClip:
    track = _longest_track(clip, max_gap)
    track = track.select(track.observed())
    if len(track.timestamps) < 2:
        raise ValueError("static hold needs at least two observed frames")
    hand = track.image_units()
    speeds = np.linalg.norm(np.diff(hand[..., :2], axis=0), axis=-1).mean(axis=-1)
    speeds /= np.diff(track.timestamps)
    # shortcut: assumes a visible hold, review uncertain intervals.
    ends = np.searchsorted(track.timestamps, track.timestamps + 0.45)
    starts = np.flatnonzero(ends < len(track.timestamps))
    if len(starts):
        prefix = np.r_[0, np.cumsum(speeds)]
        mean_speed = (prefix[ends[starts]] - prefix[starts]) / (ends[starts] - starts)
        start = starts[np.argmin(mean_speed)]
        track = track.select(slice(start, ends[start] + 1))
    indices = np.linspace(
        0, len(track.timestamps) - 1, min(count, len(track.timestamps)), dtype=int
    )
    return track.select(indices)


def _side(clip: LandmarkClip) -> str:
    sides = Counter(clip.handedness[clip.observed()])
    left, right = sides["Left"], sides["Right"]
    return "Unknown" if left == right else ("Left" if left > right else "Right")


def load_samples(config: GestureConfig, split: str, model: str) -> Samples:
    """Prepare one split. Motion backends share IDs, augmentation and masks."""
    if model not in {"static_xgb", "motion_xgb", "mlp_gru"}:
        raise ValueError(f"Unknown ASL model {model!r}")
    static = model == "static_xgb"
    classes = STATIC_CLASSES if static else DYNAMIC_CLASSES
    columns = STATIC_COLUMNS
    if model == "motion_xgb":
        columns = motion_columns(config.include_body)
    elif model == "mlp_gru":
        columns = sequence_columns(config.include_body)

    inputs, labels, clip_ids, window_ids, sides, anchors, segments = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rejected = {}
    extractors = set()
    for row in _read_split(config.manifest, split):
        clip_id, label = row["clip_id"], row["label"]
        if label not in STATIC_CLASSES + DYNAMIC_CLASSES[:-1]:
            raise ValueError(f"Unknown manifest label {label!r} in {clip_id}")
        if static and label not in STATIC_CLASSES:
            continue
        try:
            raw, extractor = _read_cache(config.cache_dir, row)
            extractors.add(extractor)
            reviewed = bool(row.get("start_sec") and row.get("end_sec"))
            positive = (
                _interval(raw, float(row["start_sec"]), float(row["end_sec"]))
                if reviewed
                else raw
            )
            segment = "reviewed" if reviewed else "inferred"
            if static:
                windows = [
                    (
                        _static_frames(
                            positive, config.static_frames, config.max_hand_gap_sec
                        ),
                        label,
                        segment,
                    )
                ]
            else:
                # shortcut: use longest track, replace with reviewed intervals.
                window = (
                    positive
                    if reviewed
                    else _longest_track(positive, config.max_hand_gap_sec)
                )
                positive_label = label
                if label in STATIC_CLASSES:
                    positive_label, segment = "OTHER", "static_negative"
                windows = [(window, positive_label, segment)]
                for start, end in json.loads(row.get("other_intervals") or "[]"):
                    windows.append(
                        (_interval(raw, start, end), "OTHER", "reviewed_negative")
                    )
        except (OSError, ValueError, KeyError, BadZipFile) as error:
            rejected[clip_id] = str(error)
            continue

        for index, (window, target, interval_source) in enumerate(windows):
            window_id = f"{clip_id}#{index}"
            variants = [window]
            if split == "train" and config.flip_train:
                variants.append(window.mirrored())
            for variant, current in enumerate(variants):
                try:
                    if static:
                        values = local_pose(current.image_units())
                    else:
                        sequence = sequence_features(
                            current,
                            length=config.sequence_length,
                            min_coverage=config.min_hand_coverage,
                            max_gap=config.max_hand_gap_sec,
                            include_body=config.include_body,
                        )
                        values = (
                            pool_motion(sequence, config.include_body)
                            if model == "motion_xgb"
                            else sequence
                        )[None]
                except ValueError as error:
                    rejected[window_id] = str(error)
                    continue
                size = len(values)
                inputs.extend(values)
                labels.extend([classes.index(target)] * size)
                clip_ids.extend([clip_id] * size)
                window_ids.extend(
                    f"{window_id}:{variant}:{frame}" for frame in range(size)
                )
                sides.extend([_side(current)] * size)
                anchors.extend(
                    [np.isfinite(current.anchors).all(axis=-1).mean()] * size
                )
                segments.extend([interval_source] * size)

    if not inputs:
        raise ValueError(
            f"No usable {split} examples for {model}. "
            f"Rejections: {list(rejected.items())[:5]}"
        )
    if len(extractors) != 1:
        raise ValueError(
            "Landmark caches use different extractors. Extract all clips again."
        )
    counts = Counter(clip_ids)
    weights = np.array([1 / counts[clip_id] for clip_id in clip_ids], dtype=np.float32)
    return Samples(
        np.stack(inputs),
        np.asarray(labels, dtype=np.int64),
        tuple(clip_ids),
        tuple(window_ids),
        weights,
        classes,
        columns,
        rejected,
        tuple(sides),
        np.asarray(anchors, dtype=np.float32),
        tuple(segments),
        extractors.pop(),
    )
