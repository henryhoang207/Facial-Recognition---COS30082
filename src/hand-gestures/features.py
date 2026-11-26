"""Landmark geometry and fixed-width motion features for isolated ASL signs."""

from dataclasses import dataclass
from typing import Self

import numpy as np

FEATURE_VERSION = 1
MIN_SCALE = 1e-6
TIPS = (("wrist", 0), ("index", 8), ("pinky", 20))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ANGLES = tuple(f"{finger}_{joint}" for finger in FINGERS for joint in (1, 2))
APERTURES = ("thumb_index", "thumb_middle", "index_middle")
STATIC_COLUMNS = (
    tuple(f"hand_{point}_{axis}" for point in range(21) for axis in "xyz")
    + APERTURES
    + tuple(f"flex_{joint}" for joint in ANGLES)
    + tuple(f"flex_valid_{joint}" for joint in ANGLES)
)
TRAJECTORY_COLUMNS = tuple(f"{point}_{axis}" for point, _ in TIPS for axis in "xy")
ANCHOR_NAMES = ("nose", "mouth", "torso")
BODY_COLUMNS = tuple(
    f"relative_{name}_{axis}" for name in ANCHOR_NAMES for axis in "xy"
) + tuple(f"anchor_valid_{name}" for name in ANCHOR_NAMES)


@dataclass(frozen=True, slots=True)
class LandmarkClip:
    timestamps: np.ndarray
    hand: np.ndarray
    handedness: np.ndarray
    anchors: np.ndarray
    width: int
    height: int

    def __post_init__(self) -> None:
        count = len(self.timestamps)
        if (
            self.timestamps.shape != (count,)
            or self.hand.shape != (count, 21, 3)
            or self.handedness.shape != (count,)
            or self.anchors.shape != (count, 4, 2)
            or not np.isfinite(self.timestamps).all()
            or np.any(np.diff(self.timestamps) <= 0)
            or self.width <= 0
            or self.height <= 0
        ):
            raise ValueError("invalid landmark cache shapes, dimensions or timestamps")

    def select(self, index: slice | np.ndarray) -> Self:
        return LandmarkClip(
            self.timestamps[index],
            self.hand[index],
            self.handedness[index],
            self.anchors[index],
            self.width,
            self.height,
        )

    def mirrored(self) -> Self:
        hand = self.hand.copy()
        hand[..., 0] = 1 - hand[..., 0]
        anchors = self.anchors[:, [0, 1, 3, 2]].copy()
        anchors[..., 0] = 1 - anchors[..., 0]
        side = np.where(
            self.handedness == "Left",
            "Right",
            np.where(self.handedness == "Right", "Left", ""),
        )
        return LandmarkClip(
            self.timestamps, hand, side, anchors, self.width, self.height
        )

    def image_units(self) -> np.ndarray:
        return self.hand * np.array([1, self.height / self.width, 1], dtype=np.float32)

    def observed(self) -> np.ndarray:
        hand = self.image_units()
        scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)
        return np.isfinite(hand).all(axis=(1, 2)) & (scale > MIN_SCALE)


def local_pose(hand: np.ndarray) -> np.ndarray:
    """Convert finite, aspect-corrected hands into wrist-relative frame features."""
    hand = hand - hand[:, :1]
    scale = np.linalg.norm(hand[:, 9], axis=1)
    if not np.isfinite(hand).all() or np.any(scale <= MIN_SCALE):
        raise ValueError("local pose requires a finite, nondegenerate hand")
    hand /= scale[:, None, None]
    apertures = np.linalg.norm(hand[:, [4, 4, 8]] - hand[:, [8, 12, 12]], axis=-1)
    digits = hand[:, 1:].reshape(-1, 5, 4, 3)
    bones = np.diff(digits, axis=2)
    left, right = -bones[:, :, :-1], bones[:, :, 1:]
    divisor = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    valid = divisor > MIN_SCALE
    cosine = np.divide(
        (left * right).sum(axis=-1), divisor, out=np.zeros_like(divisor), where=valid
    )
    angles = np.where(valid, np.arccos(cosine.clip(-1, 1)), 0)
    return np.column_stack(
        (hand.reshape(-1, 63), apertures, angles.reshape(-1, 10), valid.reshape(-1, 10))
    ).astype(np.float32, copy=False)


def track_spans(clip: LandmarkClip, max_gap: float) -> list[slice]:
    """Find continuous tracks allowing brief occlusions and handedness jitter."""
    indices = np.flatnonzero(clip.observed())
    if not len(indices):
        return []
    sides = clip.handedness[indices]
    # Require three consecutive matching labels to avoid a one-frame side flicker.
    confirmed = np.flatnonzero(
        (sides[:-2] != "")
        & (sides[:-2] == sides[1:-1])
        & (sides[:-2] == sides[2:])
        & (np.diff(indices)[:-1] == 1)
        & (np.diff(indices)[1:] == 1)
    )
    change = confirmed[1:][sides[confirmed[1:]] != sides[confirmed[:-1]]]
    gaps = np.flatnonzero(np.diff(clip.timestamps[indices]) > max_gap + 1e-8) + 1
    groups = np.split(indices, np.union1d(change, gaps))
    return [slice(int(group[0]), int(group[-1]) + 1) for group in groups if len(group)]


def sequence_columns(include_body: bool) -> tuple[str, ...]:
    return (
        STATIC_COLUMNS
        + tuple(f"displacement_{column}" for column in TRAJECTORY_COLUMNS)
        + tuple(f"velocity_{column}" for column in TRAJECTORY_COLUMNS)
        + ("hand_observed", "duration_s")
        + (BODY_COLUMNS if include_body else ())
    )


def sequence_features(
    clip: LandmarkClip,
    *,
    length: int,
    min_coverage: float,
    max_gap: float,
    include_body: bool,
) -> np.ndarray:
    observed = clip.observed()
    if len(observed) < 2 or observed.mean() < min_coverage:
        raise ValueError("insufficient observed hand coverage")
    if len(track_spans(clip, max_gap)) != 1:
        raise ValueError("hand gap exceeds limit or tracked hand changed")
    times = clip.timestamps[observed]
    if len(times) < 2:
        raise ValueError("motion interval needs at least two observed frames")
    if max(times[0] - clip.timestamps[0], clip.timestamps[-1] - times[-1]) > max_gap:
        raise ValueError("hand missing near an interval boundary")
    duration = times[-1] - times[0]
    grid = np.linspace(times[0], times[-1], length)
    raw = clip.image_units()[observed]
    resampled = np.stack(
        [np.interp(grid, times, column) for column in raw.reshape(-1, 63).T],
        axis=1,
    ).reshape(length, 21, 3)
    shape = local_pose(resampled)
    scale = np.median(np.linalg.norm(raw[:, 9] - raw[:, 0], axis=1))
    tracks = resampled[:, [point for _, point in TIPS], :2].reshape(length, 6)
    displacement = (tracks - tracks[0]) / scale
    velocity = np.gradient(displacement, grid, axis=0)
    nearest_right = np.searchsorted(clip.timestamps, grid).clip(1, len(observed) - 1)
    nearest_left = nearest_right - 1
    nearest = np.where(
        grid - clip.timestamps[nearest_left] <= clip.timestamps[nearest_right] - grid,
        nearest_left,
        nearest_right,
    )
    features = [
        shape,
        displacement,
        velocity,
        observed[nearest],
        np.full(length, duration),
    ]

    if include_body:
        anchors = clip.anchors[nearest].copy()
        anchors[..., 1] *= clip.height / clip.width
        shoulder_width = np.linalg.norm(anchors[:, 2] - anchors[:, 3], axis=-1)
        references = np.concatenate(
            (anchors[:, :2], anchors[:, 2:].mean(axis=1, keepdims=True)), axis=1
        )
        valid = np.isfinite(references).all(axis=-1) & (
            shoulder_width[:, None] > MIN_SCALE
        )
        offsets = np.divide(
            resampled[:, :1, :2] - references,
            shoulder_width[:, None, None],
            out=np.zeros_like(references),
            where=valid[..., None],
        )
        features += [offsets.reshape(length, 6), valid]
    result = np.column_stack(features).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite temporal features")
    return result


def motion_columns(include_body: bool) -> tuple[str, ...]:
    tips = tuple(name for name, _ in TIPS)
    return (
        tuple(f"net_{c}" for c in TRAJECTORY_COLUMNS)
        + tuple(f"range_{c}" for c in TRAJECTORY_COLUMNS)
        + tuple(f"path_{name}" for name in tips)
        + tuple(f"{stat}_speed_{name}" for stat in ("mean", "max") for name in tips)
        + tuple(f"bin_{i}_{c}" for i in range(3) for c in TRAJECTORY_COLUMNS)
        + tuple(
            f"aperture_{stat}_{name}" for stat in ("net", "range") for name in APERTURES
        )
        + tuple(f"{end}_{c}" for end in ("start", "end") for c in STATIC_COLUMNS)
        + ("observed_fraction", "duration_s")
        + (tuple(f"mean_{c}" for c in BODY_COLUMNS) if include_body else ())
    )


def pool_motion(sequence: np.ndarray, include_body: bool) -> np.ndarray:
    pose_width = len(STATIC_COLUMNS)
    trajectory = sequence[:, pose_width : pose_width + 6]
    velocity = sequence[:, pose_width + 6 : pose_width + 12]
    speed = np.linalg.norm(velocity.reshape(-1, 3, 2), axis=-1)
    step = np.diff(trajectory.reshape(-1, 3, 2), axis=0)
    endpoints = np.linspace(0, len(trajectory) - 1, 4, dtype=int)
    apertures = sequence[:, 63:66]
    values = [
        trajectory[-1] - trajectory[0],
        np.ptp(trajectory, axis=0),
        np.linalg.norm(step, axis=-1).sum(axis=0),
        speed.mean(axis=0),
        speed.max(axis=0),
        np.diff(trajectory[endpoints], axis=0).ravel(),
        apertures[-1] - apertures[0],
        np.ptp(apertures, axis=0),
        sequence[0, :pose_width],
        sequence[-1, :pose_width],
        np.array([sequence[:, pose_width + 12].mean(), sequence[0, pose_width + 13]]),
    ]
    if include_body:
        body = sequence[:, pose_width + 14 :]
        counts = np.repeat(body[:, 6:].sum(axis=0), 2)
        means = np.divide(
            body[:, :6].sum(axis=0),
            counts,
            out=np.zeros(6, dtype=np.float32),
            where=counts > 0,
        )
        values += [means, body[:, 6:].mean(axis=0)]
    return np.concatenate(values).astype(np.float32, copy=False)
