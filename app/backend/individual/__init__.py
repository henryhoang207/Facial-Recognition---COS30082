"""Individual features (one per team member) that run as their own stream mode.

Each feature module defines a `Feature` subclass with a `load(config, spec)` that builds its
model from `models.individual.<name>` in app/config.yaml, and per-stream hooks:

    face(frame, track, points, clock) -> dict    result for one tracked face (face[<name>])
    frame(frame, clock) -> dict                  result for the whole frame (result[<name>])

A feature whose model cannot load is reported in /api/health with the reason, and its tab
shows that reason instead of a camera. The core pipeline never depends on these models.
"""

from __future__ import annotations

import importlib
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..config import AppConfig, ModelSpec
    from ..context import AppContext
    from ..pipeline.pipeline import FramePipeline
    from ..pipeline.tracker import Track

log = logging.getLogger("faceguard")

# tên mode -> module trong package này
FEATURE_MODULES = {
    "challenge": "challenge",  # Hoan: liveness thích ứng + thử thách quay đầu
    "gesture": "gesture",  # Đức Anh: ký hiệu tay ASL động
    "head_pose": "head_pose",  # Thành: hướng quay khuôn mặt
    "face_parsing": "face_parsing",  # Đức: phân vùng các bộ phận trên mặt
}


class Feature:
    """Per-stream state for one individual feature. One instance per WebSocket."""

    uses_faces = True  # False: bỏ qua bước detect mặt (ví dụ nhận diện tay)
    needs_points = False  # True: cần 5 điểm mốc YuNet

    def __init__(self, ctx: AppContext, model: Any, pipeline: FramePipeline):
        self.ctx = ctx
        self.model = model
        self.pipeline = pipeline

    @staticmethod
    def load(config: AppConfig, spec: ModelSpec) -> tuple[Any, dict[str, Any]]:
        """Return (model, extra info for /api/health). Raise to mark the feature unavailable."""
        raise NotImplementedError

    def begin(self, track_ids: list[int]) -> None:
        """Called once per frame with the visible tracks, before face()."""

    def face(self, frame: np.ndarray, track: Track, points: np.ndarray | None, clock) -> dict | None:
        return None

    def frame(self, frame: np.ndarray, clock) -> dict | None:
        return None

    def drop(self, track_id: int) -> None:
        pass

    def close(self) -> None:
        pass


class BackgroundRunner:
    """Runs a slow model on its own thread so the camera stream never waits for it.

    The stream offers work every frame; while the thread is busy the offer is dropped, so the
    model always gets the newest frame. Visible faces take turns (the one waiting longest goes
    next). Results are kept per track until a newer one arrives.
    """

    def __init__(self, fn):
        self.fn = fn
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="individual-model")
        self._busy: Future | None = None
        self._sent: dict[int, float] = {}
        self.results: dict[int, Any] = {}
        self.last_ms: float | None = None
        self.error: str | None = None  # lỗi gần nhất của model, để hiện lên trang

    def poll(self) -> None:
        if self._busy is None or not self._busy.done():
            return
        future, self._busy = self._busy, None
        try:
            key, value, ms = future.result()
        except Exception as exc:  # noqa: BLE001 - một frame lỗi không được làm sập stream
            message = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            if message != self.error:
                log.warning("individual model failed on a frame: %s", message)
            self.error = message
            return
        self.results[key] = value
        self.last_ms = ms
        self.error = None

    def offer(self, key: int, visible: list[int], *args) -> None:
        self.poll()
        if self._busy is not None or not visible:
            return
        if key != min(visible, key=lambda k: self._sent.get(k, 0.0)):
            return  # nhường cho mặt đã đợi lâu hơn
        self._sent[key] = time.monotonic()
        self._busy = self._pool.submit(self._run, key, args)

    def _run(self, key: int, args: tuple):
        started = time.perf_counter()
        value = self.fn(*args)
        return key, value, (time.perf_counter() - started) * 1000

    def forget(self, key: int) -> None:
        self.results.pop(key, None)
        self._sent.pop(key, None)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def to_box(box, values: np.ndarray) -> np.ndarray:
    """Points in frame pixels -> coordinates relative to `box` (0..1 inside it)."""
    x1, y1, x2, y2 = box
    return (np.asarray(values, np.float32) - [x1, y1]) / [max(1, x2 - x1), max(1, y2 - y1)]


def from_box(box, values: np.ndarray) -> np.ndarray:
    """Inverse of to_box: lets an older result follow the face to where it is now."""
    x1, y1, x2, y2 = box
    return np.asarray(values, np.float32) * [x2 - x1, y2 - y1] + [x1, y1]


def feature_class(name: str) -> type[Feature]:
    module = importlib.import_module(f".{FEATURE_MODULES[name]}", __name__)
    return module.Feature


def import_object(target: str) -> Any:
    """'package.module:Name' -> object. Used for predictors that live in a member's own code."""
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"predictor must look like 'package.module:Class', got {target!r}")
    return getattr(importlib.import_module(module_name), attribute)


def build_individual(
    config: AppConfig, specs: dict[str, ModelSpec]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load every configured feature. A failure only disables that feature."""
    models: dict[str, Any] = {}
    info: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        if name not in FEATURE_MODULES:
            log.warning("unknown individual feature %r in config", name)
            continue
        entry: dict[str, Any] = {"type": spec.type, "path": spec.path, "available": False}
        extra_spec = spec.model_extra or {}
        entry.update({k: extra_spec[k] for k in ("predictor", "hand_model") if k in extra_spec})
        try:
            model, extra = feature_class(name).load(config, spec)
        except Exception as exc:  # noqa: BLE001 - mọi lỗi của model cá nhân chỉ tắt tab đó
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("individual feature %s unavailable: %s", name, entry["error"])
        else:
            models[name] = model
            entry.update(available=True, **extra)
        info[name] = entry
    return models, info
