"""Per-frame pipeline: detect -> track -> liveness gate -> embed + match -> emotion -> attendance.

Mode "mask" (individual feature) is separate: detect -> track -> mask / no mask, nothing is logged.

One FramePipeline per WebSocket connection (it owns the tracker); models, gallery
and database are shared through AppContext.
"""

from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import numpy as np

from src.common.face_crop import crop_face

from .tracker import Track, Tracker

if TYPE_CHECKING:
    from ..context import AppContext

MODES = ("pipeline", "liveness", "recognition", "emotion", "register_check", "mask", "attendance_mlp")
log = logging.getLogger("faceguard")


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
        if mode == "mask" and ctx.models.mask is None:
            raise ValueError("mask model is not configured (models.mask in app/config.yaml)")
        if mode == "attendance_mlp" and ctx.models.attendance is None:
            raise ValueError("attendance MLP is not configured with a compatible Model A")
        self.ctx = ctx
        self.mode = mode
        cfg = ctx.config.pipeline
        self.tracker = Tracker(cfg.track_iou, cfg.track_ttl_frames, cfg.smooth_window)
        self._gesture_stream = ExitStack()
        self._gesture_model = None
        self._gesture_predict = None

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._gesture_stream.close()
        self._gesture_model = None
        self._gesture_predict = None
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

    def _gesture(self, frame):
        model = self.ctx.models.gesture
        try:
            if model is not self._gesture_model:
                self._gesture_stream.close()
                self._gesture_model = model
                self._gesture_predict = None
                if model is not None:
                    self._gesture_predict = self._gesture_stream.enter_context(
                        model.open_stream()
                    )
            if self._gesture_predict is not None:
                return self._gesture_predict(frame)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            self._gesture_stream.close()
            self._gesture_predict = None
            log.warning("ASL stream unavailable: %s", exc)
        return {"status": "unavailable", "label": None, "confidence": None}

    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        ctx, cfg = self.ctx, self.ctx.config.pipeline
        if self.tracker.window != cfg.smooth_window:
            self.close()
            self.tracker.resize_window(cfg.smooth_window)
        clock = _Clock()
        events: list[dict[str, Any]] = []
        with ctx.models.lock:
            score = ctx.models.mask_detector_score if self.mode == "mask" else None
            boxes = clock.run("detect", ctx.models.detector.detect, frame, score)[: cfg.max_faces]
            tracks, removed = self.tracker.update(boxes)
            for track_id in removed:
                self._drop(track_id)
            faces = [self._face(frame, track, clock, events) for track in tracks]
            extra = {}
            if self.mode == "pipeline":
                extra["gesture"] = (
                    clock.run("gesture", self._gesture, frame)
                    if ctx.models.gesture is not None
                    else self._gesture(frame)
                )
        return {
            "type": "result",
            "mode": self.mode,
            "frame_size": [int(frame.shape[1]), int(frame.shape[0])],
            "latency_ms": clock.rounded(),
            "faces": faces,
            **extra,
        }, events

    def _face(self, frame, track: Track, clock: _Clock, events: list) -> dict[str, Any]:
        face: dict[str, Any] = {
            "track_id": track.track_id,
            "box": [int(v) for v in track.box],
            "liveness": None,
            "identity": None,
            "emotion": None,
            "logged": track.logged_person is not None,
        }
        if self.mode == "mask":
            face["mask"] = self._mask(frame, track, clock)
            return face
        if self.mode == "attendance_mlp":
            face["liveness"] = self._liveness(frame, track, clock)
            face["logged"] = False
            self._attendance_mlp(frame, track, face, events, clock)
            return face
        if self.mode in ("pipeline", "liveness", "register_check"):
            face["liveness"] = self._liveness(frame, track, clock)
        if self.mode == "register_check":
            face["quality"] = self._quality(track, face["liveness"])
            return face
        if self.mode == "pipeline" and face["liveness"]["label"] == "spoof":
            self._spoof_event(frame, track, face["liveness"]["prob"], events)
            return face  # spoof dừng ở đây: không nhận diện, không ghi điểm danh
        if self.mode in ("pipeline", "recognition"):
            face["identity"] = self._identity(frame, track, clock)
        if self.mode in ("pipeline", "emotion"):
            face["emotion"] = self._emotion(frame, track, clock)
        if self.mode == "pipeline":
            self._attendance(track, face, events)
        return face

    # stages -------------------------------------------------------------

    def _liveness(self, frame, track: Track, clock: _Clock) -> dict[str, Any]:
        fresh = self.mode == "attendance_mlp" or self._due(track, 0) or track.last_live is None
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

    def _mask(self, frame, track: Track, clock: _Clock) -> dict[str, Any]:
        if self._due(track, 0) or not track.mask_probs:
            _, prob = clock.run("mask", self.ctx.models.mask.predict_frame, frame, track.box)
            track.mask_probs.append(float(prob))
        smoothed = track.mask_prob() or 0.0
        threshold = self.ctx.mask_threshold()
        return {
            "label": "mask" if smoothed >= threshold else "no_mask",
            "prob": round(smoothed, 4),
            "raw": round(track.mask_probs[-1], 4),
            "threshold": round(threshold, 4),
        }

    def _identity(self, frame, track: Track, clock: _Clock) -> dict[str, Any]:
        threshold = self.ctx.recognition_threshold()
        if self._due(track, 1) or not track.identities:
            vector = self._embed(frame, track, clock)
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

    def _embed(self, frame, track: Track, clock: _Clock) -> np.ndarray:
        size, scale = self.ctx.models.embed_crop
        crop = crop_face(frame, track.box, size, scale)
        return clock.run("embed", self.ctx.models.embedder.embed, crop)

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

    def _attendance(self, track: Track, face: dict, events: list) -> None:
        cfg = self.ctx.config.pipeline
        person_id = track.stable_identity(cfg.stable_frames, self.ctx.liveness_threshold())
        if person_id is None or track.logged_person == person_id:
            return
        _, name, score = track.identity()
        self._log_attendance(track, face, events, person_id, name, score)

    def _attendance_mlp(self, frame, track: Track, face: dict, events: list, clock: _Clock) -> None:
        ctx = self.ctx
        mlp, gallery = ctx.models.attendance, ctx.gallery
        if mlp is None:
            track.mlp_frames.clear()
            return
        revision, size = gallery.state()
        if track.mlp_gallery_revision != revision:
            track.mlp_frames.clear()
            track.mlp_gallery_revision = revision

        status = face["attendance_mlp"] = {
            "state": "collecting",
            "frames": 0,
            "required_frames": mlp.frame_count,
            "gallery_size": size,
            "required_gallery_size": mlp.gallery_size,
            "score": None,
            "threshold": round(mlp.threshold, 4),
            "candidate_name": None,
        }
        if face["liveness"]["label"] == "spoof" or track.last_live < ctx.liveness_threshold():
            track.mlp_frames.clear()
            if face["liveness"]["label"] == "spoof":
                self._spoof_event(frame, track, face["liveness"]["prob"], events)
            return
        if not size or (mlp.gallery_size is not None and size != mlp.gallery_size):
            track.mlp_frames.clear()
            status["state"] = "needs_gallery"
            return

        vector = self._embed(frame, track, clock)
        track.mlp_frames.append(vector)
        status["frames"] = len(track.mlp_frames)
        if len(track.mlp_frames) < mlp.frame_count:
            return
        snapshot_revision, entries = gallery.snapshot()
        if snapshot_revision != revision:
            track.mlp_frames.clear()
            status["frames"] = 0
            return
        frames = np.stack(track.mlp_frames)
        decision = clock.run(
            "attendance_mlp", mlp.predict, frames,
            {str(pid): entry[1] for pid, entry in entries.items()},
        )
        if ctx.gallery is not gallery or gallery.state()[0] != revision:
            track.mlp_frames.clear()
            status["frames"] = 0
            return
        status["score"] = round(decision.score, 4)
        if not decision.accepted:
            status["state"] = "rejected"
            return
        person_id = int(decision.candidate)
        name, template = entries[person_id]
        status.update(state="accepted", candidate_name=name)
        if track.logged_person == person_id:
            face["logged"] = True
        else:
            face["emotion"] = self._emotion(frame, track, clock)
            score = float(np.mean(frames @ template))
            self._log_attendance(track, face, events, person_id, name, score)

    def _log_attendance(self, track: Track, face: dict, events: list, person_id: int, name: str, score: float) -> None:
        emotion = face["emotion"]["label"] if face["emotion"] else None
        logged = self.ctx.attendance.log(
            person_id, track.live_prob() or 0.0, score, emotion
        )
        track.logged_person = person_id
        face["logged"] = True
        if logged:
            events.append({"type": "attendance_logged", **logged, "name": name})
