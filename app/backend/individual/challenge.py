"""Hoan: adaptive liveness with a head-turn challenge.

The distilled MobileNetV3-Small liveness model scores each face. A face whose first frames
pass the checkpoint threshold is trusted at once. A doubtful face must turn to its own left
and right (random order) within a time limit; the turn is measured from YuNet's 5 points
(see geometry.py). Only a verified face is recognised. Nothing is written to attendance.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from src.common.face_crop import crop_face
from src.liveness.data import eval_transform

from . import Feature as BaseFeature
from .geometry import head_angles

CHECK_FRAMES = 5  # số frame liveness trước khi quyết định
TURN_DEG = 20.0  # góc quay tối thiểu cho mỗi bước
HOLD_FRAMES = 3  # giữ đủ góc trong bấy nhiêu frame liên tiếp
CHALLENGE_SECONDS = 10.0
RETRY_SECONDS = 4.0


class AdaptiveLiveness:
    """Checkpoint from Hoan's student training (config.model = mobilenetv3_small)."""

    def __init__(self, path: Path, device: str = "cpu"):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = ckpt["config"]
        if config.get("model") != "mobilenetv3_small":
            raise ValueError(f"expected a mobilenetv3_small checkpoint, got {config.get('model')!r}")
        net = mobilenet_v3_small(weights=None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
        net.load_state_dict({k.removeprefix("net."): v for k, v in ckpt["model"].items()})
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.threshold = float(ckpt["threshold"])
        self.img_size = int(config["img_size"])
        self.crop_scale = float(config["crop_scale"])
        self.tag = config.get("tag", "")
        self.val_metrics = {k: float(v) for k, v in (ckpt.get("val_metrics") or {}).items()}
        self.transform = eval_transform()

    @torch.inference_mode()
    def predict_frame(self, frame: np.ndarray, box) -> float:
        """P(live) for the face in `box` (label convention 1 = live)."""
        face = crop_face(frame, box, size=self.img_size, scale=self.crop_scale)
        tensor = self.transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.net(tensor))[0, 0])


@dataclass
class Session:
    probs: deque = field(default_factory=lambda: deque(maxlen=CHECK_FRAMES))
    state: str = "checking"  # checking | challenge | passed | failed
    reason: str | None = None  # live | challenge | timeout
    steps: list[str] = field(default_factory=list)
    step: int = 0
    hold: int = 0
    deadline: float = 0.0
    retry_at: float = 0.0


class Feature(BaseFeature):
    needs_points = True

    @staticmethod
    def load(config, spec):
        if spec.type != "adaptive_liveness":
            raise ValueError(f"unknown challenge model type {spec.type!r}")
        model = AdaptiveLiveness(config.path(spec.path or ""), device=config.device)
        return model, {
            "threshold": model.threshold,
            "tag": model.tag,
            "img_size": model.img_size,
            "crop_scale": model.crop_scale,
            "val_acer": model.val_metrics.get("val_acer"),
            "val_auc": model.val_metrics.get("val_auc"),
            "turn_deg": TURN_DEG,
            "challenge_seconds": CHALLENGE_SECONDS,
        }

    def __init__(self, ctx, model, pipeline):
        super().__init__(ctx, model, pipeline)
        self.sessions: dict[int, Session] = {}

    def drop(self, track_id: int) -> None:
        self.sessions.pop(track_id, None)

    def face(self, frame, track, points, clock) -> dict[str, Any]:
        now = time.monotonic()
        session = self.sessions.setdefault(track.track_id, Session())
        prob = clock.run("adaptive_liveness", self.model.predict_frame, frame, track.box)
        yaw = None
        if points is not None:
            try:
                yaw = head_angles(points)["yaw"]
            except ValueError:
                yaw = None

        if session.state == "failed" and now >= session.retry_at:
            self.sessions[track.track_id] = session = Session()
        if session.state == "checking":
            session.probs.append(prob)
            if len(session.probs) == CHECK_FRAMES:
                if float(np.mean(session.probs)) >= self.model.threshold:
                    session.state, session.reason = "passed", "live"
                else:
                    session.state = "challenge"
                    session.steps = random.sample(["left", "right"], 2)
                    session.deadline = now + CHALLENGE_SECONDS
        elif session.state == "challenge":
            if now > session.deadline:
                session.state, session.reason = "failed", "timeout"
                session.retry_at = now + RETRY_SECONDS
            else:
                target = session.steps[session.step]
                turned = yaw is not None and (yaw >= TURN_DEG if target == "left" else yaw <= -TURN_DEG)
                session.hold = session.hold + 1 if turned else 0
                if session.hold >= HOLD_FRAMES:
                    session.step, session.hold = session.step + 1, 0
                    if session.step == len(session.steps):
                        session.state, session.reason = "passed", "challenge"

        identity = None
        if session.state == "passed":
            identity = self.pipeline._identity(frame, track, clock)
        in_challenge = session.state == "challenge"
        return {
            "state": session.state,
            "reason": session.reason,
            "live_prob": round(float(np.mean(session.probs)) if session.probs else prob, 4),
            "raw": round(prob, 4),
            "threshold": round(self.model.threshold, 4),
            "frames": len(session.probs),
            "required_frames": CHECK_FRAMES,
            "yaw": None if yaw is None else round(yaw, 1),
            "turn_deg": TURN_DEG,
            "steps": session.steps,
            "step_index": session.step,
            "step": session.steps[session.step] if in_challenge else None,
            "hold": session.hold,
            "hold_frames": HOLD_FRAMES,
            "seconds_left": round(max(0.0, session.deadline - now), 1) if in_challenge else None,
            "retry_in": round(max(0.0, session.retry_at - now), 1) if session.state == "failed" else None,
            "identity": identity,
        }
