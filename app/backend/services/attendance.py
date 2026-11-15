"""Attendance + spoof logging against the active session."""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from src.common.face_crop import crop_face

if TYPE_CHECKING:
    from ..context import AppContext


class AttendanceService:
    def __init__(self, ctx: AppContext):
        self.ctx = ctx
        self._lock = threading.Lock()
        self._logged: set[tuple[int, int]] = set()

    def session_id(self) -> int:
        session = self.ctx.repo.active_session()
        if session is None:
            session = self.ctx.repo.start_session(
                f"Session {datetime.now():%Y-%m-%d %H:%M}"
            )
        return int(session.id)

    def log(
        self, person_id: int, live_prob: float, match_score: float, emotion: str | None
    ) -> dict[str, Any] | None:
        """Ghi điểm danh; trả về event nếu đây là lần đầu trong session, None nếu đã có."""
        session_id = self.session_id()
        key = (session_id, person_id)
        with self._lock:
            if key in self._logged:
                return None
            inserted = self.ctx.repo.log_attendance(
                session_id, person_id, live_prob, match_score, emotion
            )
            self._logged.add(key)
        if not inserted:
            return None
        return {
            "session_id": session_id,
            "person_id": person_id,
            "time": datetime.now().strftime("%H:%M:%S"),
            "emotion": emotion,
            "match_score": round(match_score, 4),
        }

    def record_spoof(
        self, frame: np.ndarray, box, live_prob: float
    ) -> dict[str, Any] | None:
        session_id = self.session_id()
        ts = datetime.now()
        folder = self.ctx.config.path(self.ctx.config.storage.spoof_dir)
        folder.mkdir(parents=True, exist_ok=True)
        snapshot = folder / f"s{session_id}_{ts:%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(snapshot), crop_face(frame, box, 160, 1.5))
        relative = snapshot.relative_to(self.ctx.config.path("app/instance")).as_posix()
        event = self.ctx.repo.add_spoof_event(session_id, live_prob, relative)
        return {
            "session_id": session_id,
            "event_id": event.id,
            "time": ts.strftime("%H:%M:%S"),
            "live_prob": round(live_prob, 4),
            "snapshot": relative,
        }

    def forget_cache(self) -> None:
        with self._lock:
            self._logged.clear()


def snapshot_path(ctx: AppContext, relative: str) -> Path:
    base = ctx.config.path("app/instance").resolve()
    target = (base / relative).resolve()
    if base not in target.parents:
        raise ValueError("invalid snapshot path")
    return target
