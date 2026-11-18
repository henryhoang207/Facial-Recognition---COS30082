"""Per-frame pipeline: detect -> track -> liveness gate -> embed + match -> emotion -> attendance.

One FramePipeline per WebSocket connection (it owns the tracker); models, gallery
and database are shared through AppContext.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from src.common.face_crop import crop_face

from .tracker import Track, Tracker

if TYPE_CHECKING:
    from ..context import AppContext

MODES = ("pipeline", "liveness", "recognition", "emotion", "register_check")


class _Clock:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    def run(self, stage: str, fn, *args):
        started = time.perf_counter()
        result = fn(*args)
        self.ms[stage] = self.ms.get(stage, 0.0) + (time.perf_counter() - started) * 1000
        return result

    def rounded(self) -> dict[str, float]:
        out = {k: round(v, 1) for k, v in self.ms.items()}
        out["total"] = round(sum(self.ms.values()), 1)
        return out


class FramePipeline:
    def __init__(self, ctx: AppContext, mode: str = "pipeline"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.ctx = ctx
        self.mode = mode
        cfg = ctx.config.pipeline
        self.tracker = Tracker(cfg.track_iou, cfg.track_ttl_frames, cfg.smooth_window)

    # ------------------------------------------------------------------

    def close(self) -> None:
        for track_id in list(self.tracker.tracks):
            self._drop(track_id)
        self.tracker.tracks.clear()

    def _drop(self, track_id: int) -> None:
        drop = getattr(self.ctx.models.emotion, "drop_track", None)
        if drop:
            drop(track_id)

    def _due(self, track: Track, offset: int) -> bool:
        every = self.ctx.config.pipeline.model_every_n
        return every == 1 or track.age == 1 or (track.age + offset) % every == 0

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ctx, cfg = self.ctx, self.ctx.config.pipeline
        if self.tracker.window != cfg.smooth_window:
            self.close()
            self.tracker.resize_window(cfg.smooth_window)
        clock = _Clock()
        events: list[dict[str, Any]] = []
        with ctx.models.lock:
            boxes = clock.run("detect", ctx.models.detector.detect, frame)[: cfg.max_faces]
            tracks, removed = self.tracker.update(boxes)
            for track_id in removed:
                self._drop(track_id)
            faces = [self._face(frame, track, clock, events) for track in tracks]
        return {
            "type": "result",
            "mode": self.mode,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "latency_ms": clock.rounded(),
            "faces": faces,
        }, events

    def _face(self, frame, track: Track, clock: _Clock, events: list) -> dict[str, Any]:
        face: dict[str, Any] = {
            "track_id": track.track_id,
            "box": [int(v) for v in track.box],
            "liveness": None,
            "identity": None,
            "emotion": None,
            "attendance_mlp": None,
            "logged": track.logged_person is not None,
        }
        if self.mode in ("pipeline", "liveness", "register_check"):
            face["liveness"] = self._liveness(frame, track, clock)
        if self.mode == "register_check":
            face["quality"] = self._quality(track, face["liveness"])
            return face
        if self.mode == "pipeline" and face["liveness"]["label"] == "spoof":
            track.mlp_frames.clear()
            self._spoof_event(frame, track, face["liveness"]["prob"], events)
            return face  # spoof dừng ở đây: không nhận diện, không ghi điểm danh
        if self.mode in ("pipeline", "recognition"):
            face["identity"] = self._identity(frame, track, clock)
        if self.mode in ("pipeline", "emotion"):
            face["emotion"] = self._emotion(frame, track, clock)
        if self.mode == "pipeline":
            self._attendance(track, face, events, clock)
        return face

    # stages -------------------------------------------------------------

    def _liveness(self, frame, track: Track, clock: _Clock) -> dict[str, Any]:
        fresh = (
            (self.mode == "pipeline" and self.ctx.models.attendance is not None)
            or self._due(track, 0)
            or track.last_live is None
        )
        if fresh:
            _, prob = clock.run("liveness", self.ctx.models.liveness.predict_frame, frame, track.box)
            track.last_live = float(prob)
            track.live_probs.append(float(prob))
        smoothed = track.live_prob() or 0.0
        threshold = self.ctx.liveness_threshold()
        return {
            "label": "live" if smoothed >= threshold else "spoof",
            "prob": round(smoothed, 4),
            "raw": round(track.last_live, 4),
            "threshold": round(threshold, 4),
            "fresh": fresh,
        }

    def _identity(self, frame, track: Track, clock: _Clock) -> dict[str, Any]:
        threshold = self.ctx.recognition_threshold()
        mlp_active = self.mode == "pipeline" and self.ctx.models.attendance is not None
        if mlp_active or self._due(track, 1) or not track.identities:
            size, scale = self.ctx.models.embed_crop
            crop = crop_face(frame, track.box, size, scale)
            vector = clock.run("embed", self.ctx.models.embedder.embed, crop)
            if mlp_active:
                revision, _ = self.ctx.gallery.snapshot()
                if track.mlp_gallery_revision != revision:
                    track.mlp_frames.clear()
                    track.mlp_gallery_revision = revision
                track.mlp_frames.append(vector)
            matches = clock.run("match", self.ctx.gallery.match, vector, 3)
            track.top_matches = matches
            best = matches[0] if matches else None
            if best and best.score >= threshold:
                track.identities.append((best.person_id, best.name, best.score))
            else:
                track.identities.append((None, None, best.score if best else 0.0))
        person_id, name, score = track.identity()
        result = {
            "person_id": person_id,
            "name": name,
            "score": round(score, 4),
            "known": person_id is not None,
            "threshold": round(threshold, 4),
            # top-3 của lần so khớp gần nhất, để UI hiện cả những người gần giống
            "top": [
                {"person_id": m.person_id, "name": m.name, "score": round(m.score, 4)}
                for m in track.top_matches
            ],
        }
        return result

    def _emotion(self, frame, track: Track, clock: _Clock) -> dict[str, Any] | None:
        model = self.ctx.models.emotion
        if self._due(track, 2) or not track.emotions:
            crop = crop_face(frame, track.box, model.img_size, model.crop_scale)
            predict = getattr(model, "predict_track", None)
            if predict:
                _, probs = clock.run("emotion", predict, track.track_id, crop)
            else:
                _, probs = clock.run("emotion", model.predict, crop)
            track.emotions.append(probs)
        label, probs = track.emotion()
        if label is None:
            return None
        return {"label": label, "probs": {k: round(v, 4) for k, v in probs.items()}}

    def _quality(self, track: Track, liveness: dict) -> dict[str, Any]:
        x1, y1, x2, y2 = track.box
        min_side = min(x2 - x1, y2 - y1)
        reasons = []
        if len(self.tracker.tracks) > 1:
            reasons.append("more_than_one_face")
        if min_side < self.ctx.config.registration.min_face_pixels:
            reasons.append("face_too_small")
        if liveness["label"] != "live":
            reasons.append("not_live")
        return {"ok": not reasons, "reasons": reasons, "face_pixels": int(min_side)}

    # side effects -----------------------------------------------------

    def _spoof_event(self, frame, track: Track, live_prob: float, events: list) -> None:
        now = time.monotonic()
        cooldown = self.ctx.config.pipeline.spoof_cooldown_s
        if len(track.live_probs) < self.ctx.config.pipeline.stable_frames:
            return  # chờ đủ frame để chắc chắn mới ghi
        if now - track.last_spoof_event < cooldown:
            return
        track.last_spoof_event = now
        event = self.ctx.attendance.record_spoof(frame, track.box, live_prob)
        if event:
            events.append({"type": "spoof_blocked", "track_id": track.track_id, **event})

    def _attendance(self, track: Track, face: dict, events: list, clock: _Clock) -> None:
        if self.ctx.models.attendance is not None:
            self._attendance_mlp(track, face, events, clock)
            return
        cfg = self.ctx.config.pipeline
        person_id = track.stable_identity(cfg.stable_frames, self.ctx.liveness_threshold())
        if person_id is None or track.logged_person == person_id:
            return
        _, name, score = track.identity()
        emotion = face["emotion"]["label"] if face["emotion"] else None
        logged = self.ctx.attendance.log(
            person_id, track.live_prob() or 0.0, score, emotion
        )
        track.logged_person = person_id
        face["logged"] = True
        if logged:
            events.append({"type": "attendance_logged", **logged, "name": name})

    def _attendance_mlp(self, track: Track, face: dict, events: list, clock: _Clock) -> None:
        ctx, mlp = self.ctx, self.ctx.models.attendance
        revision, gallery = ctx.gallery.snapshot()
        if track.mlp_gallery_revision != revision:
            track.mlp_frames.clear()
        status = {
            "state": "collecting",
            "frames": len(track.mlp_frames),
            "required_frames": mlp.frame_count,
            "gallery_size": len(gallery),
            "required_gallery_size": mlp.gallery_size,
            "score": None,
            "threshold": round(mlp.threshold, 4),
            "candidate_name": None,
            "accepted": False,
        }
        face["attendance_mlp"] = status
        if len(gallery) != mlp.gallery_size:
            track.mlp_frames.clear()
            status.update(state="needs_gallery", frames=0)
            return
        if face["liveness"]["raw"] < ctx.liveness_threshold():
            track.mlp_frames.clear()
            status["frames"] = 0
            return
        if len(track.mlp_frames) < mlp.frame_count:
            return
        frames = np.stack(track.mlp_frames)
        decision = clock.run(
            "attendance_mlp", mlp.predict, frames,
            {str(pid): vector for pid, (_, vector) in gallery.items()},
        )
        person_id = int(decision.candidate)
        name, template = gallery[person_id]
        status.update(
            state="accepted" if decision.accepted else "rejected",
            score=round(decision.score, 4),
            candidate_name=name,
            accepted=decision.accepted,
        )
        if not decision.accepted or track.logged_person is not None:
            return
        if ctx.gallery.snapshot()[0] != revision:
            track.mlp_frames.clear()
            return
        cosine = float(np.mean(frames @ template))
        emotion = face["emotion"]["label"] if face["emotion"] else None
        logged = ctx.attendance.log(person_id, track.live_prob() or 0.0, cosine, emotion)
        track.logged_person = person_id
        face["logged"] = True
        if logged:
            events.append({"type": "attendance_logged", **logged, "name": name})
