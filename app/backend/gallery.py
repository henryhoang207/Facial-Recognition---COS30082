"""In-memory copy of registered embeddings. Matching never touches the database."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from itertools import count

import numpy as np


_revisions = count(1)


@dataclass(frozen=True)
class Match:
    person_id: int
    name: str
    score: float


class GalleryCache:
    def __init__(self, dim: int):
        self.dim = dim
        self._lock = threading.Lock()
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._person_ids: list[int] = []
        self._names: dict[int, str] = {}
        self._revision = next(_revisions)

    def load(self, rows: list[tuple[int, str, np.ndarray]]) -> None:
        rows = [r for r in rows if r[2].shape == (self.dim,)]
        with self._lock:
            self._matrix = (
                np.stack([r[2] for r in rows]).astype(np.float32)
                if rows
                else np.zeros((0, self.dim), dtype=np.float32)
            )
            self._person_ids = [r[0] for r in rows]
            self._names = {r[0]: r[1] for r in rows}
            self._revision = next(_revisions)

    def add(self, person_id: int, name: str, vector: np.ndarray) -> None:
        with self._lock:
            self._matrix = np.vstack([self._matrix, vector[None].astype(np.float32)])
            self._person_ids.append(person_id)
            self._names[person_id] = name
            self._revision = next(_revisions)

    def remove(self, person_id: int) -> None:
        with self._lock:
            keep = [i for i, pid in enumerate(self._person_ids) if pid != person_id]
            self._matrix = self._matrix[keep]
            self._person_ids = [self._person_ids[i] for i in keep]
            self._names.pop(person_id, None)
            self._revision = next(_revisions)

    def snapshot(self) -> tuple[int, dict[int, tuple[str, np.ndarray]]]:
        with self._lock:
            return self._revision, {
                pid: (self._names[pid], vector.copy())
                for pid, vector in zip(self._person_ids, self._matrix, strict=True)
            }

    def __len__(self) -> int:
        return len(self._names)

    def match(self, embedding: np.ndarray, k: int = 3) -> list[Match]:
        """Top-k người theo cosine (vector đã L2-normalize nên cosine = tích vô hướng)."""
        with self._lock:
            if not self._person_ids:
                return []
            scores = self._matrix @ embedding.astype(np.float32)
            best: dict[int, float] = {}
            for pid, score in zip(self._person_ids, scores, strict=True):
                best[pid] = max(best.get(pid, -1.0), float(score))
            ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:k]
            return [Match(pid, self._names[pid], score) for pid, score in ranked]
