"""Đức: face parsing (which pixels are skin, brows, eyes, nose, lips, mouth, hair).

The app loads the predictor named in app/config.yaml (`models.individual.face_parsing.predictor`,
for example `src.face_parsing.predictor:FaceParsingPredictor`). Contract:

    predictor = FaceParsingPredictor(checkpoint_path, device="cpu")
    predictor.classes -> list[str]          # index = class id, 0 = background
    predictor.predict_frame(frame_bgr, box) -> (mask, region)
        mask:   (H, W) uint8 class ids
        region: (x1, y1, x2, y2) in frame pixels that the mask covers

The app colours the mask, sends it to the page as a small PNG and lists the area of each part.
`type: geometry` (mock config only) draws rough regions from YuNet's 5 points instead.
"""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np

from . import BackgroundRunner, from_box, import_object, to_box
from . import Feature as BaseFeature

CONTRACT = "predict_frame(frame_bgr, box) -> (mask HxW uint8 class ids, region (x1, y1, x2, y2))"
CLASSES = [
    "Background", "Skin", "Left brow", "Right brow", "Left eye", "Right eye",
    "Nose", "Upper lip", "Inner mouth", "Lower lip", "Hair",
]
# nhóm hiển thị: tên lớp -> (tên nhóm, màu RGB)
PARTS = {
    "Skin": ("Skin", (244, 196, 160)),
    "Left brow": ("Brows", (139, 90, 43)),
    "Right brow": ("Brows", (139, 90, 43)),
    "Left eye": ("Eyes", (47, 125, 246)),
    "Right eye": ("Eyes", (47, 125, 246)),
    "Nose": ("Nose", (245, 158, 11)),
    "Upper lip": ("Lips", (225, 29, 72)),
    "Lower lip": ("Lips", (225, 29, 72)),
    "Inner mouth": ("Mouth", (124, 45, 18)),
    "Hair": ("Hair", (109, 40, 217)),
}
OVERLAY_SIDE = 192  # cạnh dài của PNG gửi lên trang
SKIN_ALPHA, PART_ALPHA = 70, 170


def geometry_mask(frame: np.ndarray, box, points: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Rough parts from YuNet's 5 points, only to try the page without Đức's model."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    top = max(0, int(y1 - 0.35 * height))
    region = (int(x1), top, int(x2), int(y2))
    mask = np.zeros((region[3] - region[1], region[2] - region[0]), np.uint8)
    p = (points - [region[0], region[1]]).astype(int)
    eye_dist = max(4, int(np.linalg.norm(points[1] - points[0])))
    center = (width // 2, int(y1 - top + height * 0.5))
    cv2.ellipse(mask, (center[0], center[1] - height // 5), (width // 2, int(height * 0.55)), 0, 180, 360, 10, -1)
    cv2.ellipse(mask, center, (int(width * 0.42), int(height * 0.52)), 0, 0, 360, 1, -1)
    for eye, brow, eye_id, brow_id in ((p[0], p[0], 5, 3), (p[1], p[1], 4, 2)):
        cv2.ellipse(mask, (int(brow[0]), int(brow[1] - eye_dist * 0.32)), (eye_dist // 4, eye_dist // 14 + 1), 0, 0, 360, brow_id, -1)
        cv2.ellipse(mask, tuple(int(v) for v in eye), (eye_dist // 5, eye_dist // 10 + 1), 0, 0, 360, eye_id, -1)
    cv2.ellipse(mask, tuple(int(v) for v in p[2]), (eye_dist // 6, eye_dist // 4), 0, 0, 360, 6, -1)
    mouth = ((p[3] + p[4]) / 2).astype(int)
    half = max(3, int(np.linalg.norm(p[4] - p[3]) / 2))
    cv2.ellipse(mask, tuple(mouth), (half, half // 3 + 2), 0, 180, 360, 7, -1)
    cv2.ellipse(mask, tuple(mouth), (half, half // 3 + 2), 0, 0, 180, 9, -1)
    cv2.ellipse(mask, tuple(mouth), (half // 2, 2), 0, 0, 360, 8, -1)
    return mask, region


def colour(mask: np.ndarray, classes: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Mask -> (PNG data URL, parts with colour and share of the face area)."""
    scale = OVERLAY_SIDE / max(mask.shape)
    small = cv2.resize(
        mask, (max(1, round(mask.shape[1] * scale)), max(1, round(mask.shape[0] * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    rgba = np.zeros((*small.shape, 4), np.uint8)
    face_pixels = max(1, int((mask > 0).sum()))
    areas: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(classes):
        if name not in PARTS:
            continue
        group, rgb = PARTS[name]
        rgba[small == index] = (*rgb[::-1], SKIN_ALPHA if group == "Skin" else PART_ALPHA)  # BGRA
        entry = areas.setdefault(group, {"name": group, "color": "#{:02x}{:02x}{:02x}".format(*rgb), "share": 0.0})
        entry["share"] += float((mask == index).sum()) / face_pixels
    ok, png = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("could not encode the parsing overlay")
    parts = [{**entry, "share": round(entry["share"], 4)} for entry in areas.values()]
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(), parts


class Feature(BaseFeature):
    """The parsing model (512 px) runs on a background thread (BackgroundRunner), so the stream
    keeps its frame rate; the newest mask is reused and moved with the face box until the next
    one lands."""

    needs_points = True

    @staticmethod
    def load(config, spec):
        if spec.type == "geometry":
            return "geometry", {"contract": CONTRACT, "source": "YuNet 5 points", "classes": CLASSES}
        if spec.type != "external":
            raise ValueError(f"unknown face parsing type {spec.type!r}")
        extra = spec.model_extra or {}
        target = extra.get("predictor", "")
        predictor = import_object(target)(
            config.path(spec.path or ""), device=extra.get("device", config.device)
        )
        return predictor, {
            "contract": CONTRACT,
            "source": target,
            "classes": list(predictor.classes),
            "device": str(getattr(predictor, "device", config.device)),
            "precision": getattr(predictor, "precision", "fp32"),
            "gpu_error": getattr(predictor, "gpu_error", None),
        }

    def __init__(self, ctx, model, pipeline):
        super().__init__(ctx, model, pipeline)
        self.visible: list[int] = []
        self.runner = BackgroundRunner(self._parse)

    def _parse(self, frame, box, points) -> dict[str, Any] | None:
        """Runs on the background thread. Region kept relative to `box`."""
        if self.model == "geometry":
            if points is None:
                return None
            mask, region = geometry_mask(frame, box, points)
            classes = CLASSES
        else:
            mask, region = self.model.predict_frame(frame, box)
            classes = list(self.model.classes)
        overlay, parts = colour(np.asarray(mask, np.uint8), classes)
        corners = to_box(box, [region[:2], region[2:]])
        return {"overlay": overlay, "corners": corners, "parts": parts}

    def begin(self, track_ids: list[int]) -> None:
        self.visible = track_ids

    def frame(self, frame, clock) -> dict[str, Any] | None:
        if self.runner is None:
            return None
        return {"error": self.runner.error}

    def drop(self, track_id: int) -> None:
        self.runner.forget(track_id)

    def close(self) -> None:
        self.runner.close()

    def face(self, frame, track, points, clock) -> dict[str, Any] | None:
        self.runner.offer(track.track_id, self.visible, frame, track.box, points)
        if self.runner.last_ms is not None:
            clock.ms["face_parsing"] = self.runner.last_ms
        result = self.runner.results.get(track.track_id)
        if result is None:
            return None
        (x1, y1), (x2, y2) = from_box(track.box, result["corners"])
        return {
            "overlay": result["overlay"],
            "region": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "parts": result["parts"],
        }
