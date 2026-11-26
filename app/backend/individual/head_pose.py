"""Thành: head pose (which way the face is turned).

The app loads the predictor named in app/config.yaml (`models.individual.head_pose.predictor`,
for example `src.head_pose.predictor:HeadPosePredictor`). Contract:

    predictor = HeadPosePredictor(checkpoint_path, device="auto")
    predictor.predict_frame(frame_bgr, box) -> {"yaw": float, "pitch": float, "roll": float,
                                                "landmarks": [[x, y], ...]}   # landmarks optional

Angles in degrees: yaw > 0 = the person turned to their own left, pitch > 0 = chin up.
The app's config uses src.head_pose.predictor (Thành's HRNet-W48 checkpoint, decoded in that file).
`type: geometry` estimates the angles from YuNet's 5 points instead.

The model is slow (HRNet-W48), so it runs on a background thread (BackgroundRunner): the stream
keeps its frame rate, the newest model result is reused until the next one lands, and its
landmarks are moved with the face box in between. Until the first result, YuNet's points are used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import BackgroundRunner, from_box, import_object, to_box
from . import Feature as BaseFeature
from .geometry import head_angles

CONTRACT = "predict_frame(frame_bgr, box) -> {'yaw', 'pitch', 'roll'} in degrees, optional 'landmarks'"
TURN_DEG = 15.0
NOD_DEG = 12.0
SMOOTHING = 0.5  # trung bình trượt để số đỡ nhảy
MAX_DISAGREEMENT = 0.35  # landmark lệch quá mức này so với 5 điểm YuNet (theo khoảng cách 2 mắt) thì bỏ


def direction(yaw: float, pitch: float) -> str:
    if abs(yaw) >= TURN_DEG and abs(yaw) >= abs(pitch):
        return "left" if yaw > 0 else "right"
    if abs(pitch) >= NOD_DEG:
        return "up" if pitch > 0 else "down"
    return "frontal"


def disagreement(landmarks, points) -> float:
    """68 300W points vs YuNet's 5 points (eyes, nose, mouth corners), in eye distances."""
    lm = np.asarray(landmarks, np.float32)
    if lm.shape != (68, 2):
        return 0.0
    ours = np.array([lm[36:42].mean(0), lm[42:48].mean(0), lm[30], lm[48], lm[54]])
    eye = float(np.linalg.norm(points[0] - points[1])) or 1.0
    return float(np.linalg.norm(ours - points, axis=1).mean()) / eye


class Feature(BaseFeature):
    needs_points = True

    @staticmethod
    def load(config, spec):
        if spec.type == "geometry":
            return "geometry", {"contract": CONTRACT, "source": "YuNet 5 points"}
        if spec.type != "external":
            raise ValueError(f"unknown head pose type {spec.type!r}")
        extra = spec.model_extra or {}
        target = extra.get("predictor", "")
        predictor = import_object(target)(
            config.path(spec.path or ""), device=extra.get("device", config.device)
        )
        return predictor, {
            "contract": CONTRACT,
            "source": target,
            "device": str(getattr(predictor, "device", config.device)),
            "precision": getattr(predictor, "precision", "fp32"),
            "gpu_error": getattr(predictor, "gpu_error", None),
        }

    def __init__(self, ctx, model, pipeline):
        super().__init__(ctx, model, pipeline)
        self.smoothed: dict[int, dict[str, float]] = {}
        self.visible: list[int] = []
        self.runner = None if model == "geometry" else BackgroundRunner(self._estimate)

    def _estimate(self, frame, box, points) -> dict[str, Any]:
        """Runs on the background thread. Result kept relative to `box`."""
        raw = dict(self.model.predict_frame(frame, box))
        landmarks = raw.get("landmarks")
        if points is not None and landmarks is not None and (
            disagreement(landmarks, points) > MAX_DISAGREEMENT or abs(raw["roll"]) > 75
        ):
            return {**head_angles(points), "landmarks": None, "source": "detector"}  # mặt bị che
        relative = None if landmarks is None else to_box(box, landmarks)
        return {**raw, "landmarks": relative, "source": "model"}

    def begin(self, track_ids: list[int]) -> None:
        self.visible = track_ids

    def frame(self, frame, clock) -> dict[str, Any] | None:
        if self.runner is None:
            return None
        return {"error": self.runner.error}

    def drop(self, track_id: int) -> None:
        self.smoothed.pop(track_id, None)
        if self.runner is not None:
            self.runner.forget(track_id)

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()

    def face(self, frame, track, points, clock) -> dict[str, Any] | None:
        raw: dict[str, Any] | None = None
        if self.runner is not None:
            self.runner.offer(track.track_id, self.visible, frame, track.box, points)
            result = self.runner.results.get(track.track_id)
            if self.runner.last_ms is not None:
                clock.ms["head_pose"] = self.runner.last_ms
            if result is not None:
                raw = dict(result)
                if raw.get("landmarks") is not None:
                    raw["landmarks"] = from_box(track.box, raw["landmarks"])
        if raw is None:
            if points is None:
                return None
            # chưa có kết quả model đầu tiên (hoặc mode geometry): dùng 5 điểm YuNet
            source = "starting" if self.runner is not None else "detector"
            raw = {**clock.run("geometry", head_angles, points), "source": source}
        previous = self.smoothed.get(track.track_id)
        angles = {
            key: float(raw[key]) if previous is None
            else SMOOTHING * float(raw[key]) + (1 - SMOOTHING) * previous[key]
            for key in ("yaw", "pitch", "roll")
        }
        self.smoothed[track.track_id] = angles
        landmarks = raw.get("landmarks")
        return {
            **{key: round(value, 1) for key, value in angles.items()},
            "direction": direction(angles["yaw"], angles["pitch"]),
            "source": raw.get("source", "model"),
            "landmarks": None
            if landmarks is None
            else [[round(float(x), 1), round(float(y), 1)] for x, y in landmarks],
        }
