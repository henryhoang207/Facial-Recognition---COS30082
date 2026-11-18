"""IoU tracker: gives each face a stable track_id across frames, holds its smoothing history."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from src.common.interfaces import Box


def iou(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: Box
    window: int
    age: int = 0
    missed: int = 0
    live_probs: deque = field(init=False)
    identities: deque = field(init=False)  # (person_id | None, name | None, score)
    emotions: deque = field(init=False)  # dict label -> prob
    logged_person: int | None = None
    last_spoof_event: float = 0.0
    last_live: float | None = None
    top_matches: list = field(default_factory=list)
    mlp_frames: deque = field(init=False)
    mlp_gallery_revision: int | None = None

    def __post_init__(self) -> None:
        self.live_probs = deque(maxlen=self.window)
        self.identities = deque(maxlen=self.window)
        self.emotions = deque(maxlen=self.window)
        self.mlp_frames = deque(maxlen=5)

    def live_prob(self) -> float | None:
        return float(np.mean(self.live_probs)) if self.live_probs else None

    def identity(self) -> tuple[int | None, str | None, float]:
        """Bỏ phiếu theo đa số trong cửa sổ; điểm = trung bình các phiếu của người thắng."""
        if not self.identities:
            return None, None, 0.0
        votes = Counter(pid for pid, _, _ in self.identities)
        winner, _ = votes.most_common(1)[0]
        hits = [(name, score) for pid, name, score in self.identities if pid == winner]
        return winner, hits[-1][0], float(np.mean([s for _, s in hits]))

    def emotion(self) -> tuple[str | None, dict[str, float]]:
        if not self.emotions:
            return None, {}
        labels = self.emotions[-1].keys()
        mean = {k: float(np.mean([e.get(k, 0.0) for e in self.emotions])) for k in labels}
        return max(mean, key=mean.get), mean

    def stable_identity(self, frames: int, threshold: float) -> int | None:
        """Người được nhận ra ổn định: `frames` frame gần nhất cùng một người và đều live."""
        if len(self.identities) < frames or len(self.live_probs) < frames:
            return None
        recent_ids = list(self.identities)[-frames:]
        recent_live = list(self.live_probs)[-frames:]
        first = recent_ids[0][0]
        if first is None or any(pid != first for pid, _, _ in recent_ids):
            return None
        if any(p < threshold for p in recent_live):
            return None
        return first


class Tracker:
    def __init__(self, iou_threshold: float, ttl: int, window: int):
        self.iou_threshold = iou_threshold
        self.ttl = ttl
        self.window = window
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, boxes: list[Box]) -> tuple[list[Track], list[int]]:
        """Ghép box mới với track cũ (tham lam theo IoU). Trả về (track đang thấy, id track đã xoá)."""
        pairs = sorted(
            (
                (iou(track.box, box), track_id, index)
                for track_id, track in self.tracks.items()
                for index, box in enumerate(boxes)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        assigned: dict[int, int] = {}
        for score, track_id, index in pairs:
            if score < self.iou_threshold:
                break
            if track_id in used_tracks or index in assigned:
                continue
            used_tracks.add(track_id)
            assigned[index] = track_id

        visible: list[Track] = []
        for index, box in enumerate(boxes):
            if index in assigned:
                track = self.tracks[assigned[index]]
                track.box, track.missed = box, 0
            else:
                track = Track(self._next_id, box, self.window)
                self.tracks[track.track_id] = track
                self._next_id += 1
            track.age += 1
            visible.append(track)

        removed: list[int] = []
        seen = {t.track_id for t in visible}
        for track_id in list(self.tracks):
            if track_id not in seen:
                self.tracks[track_id].missed += 1
                self.tracks[track_id].mlp_frames.clear()
                if self.tracks[track_id].missed > self.ttl:
                    del self.tracks[track_id]
                    removed.append(track_id)
        return visible, removed

    def resize_window(self, window: int) -> None:
        if window == self.window:
            return
        self.window = window
        self.tracks.clear()
