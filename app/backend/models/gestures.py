"""Serve a trained ASL Alphabet MLP with one hand tracker per stream."""

from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from time import monotonic_ns

import numpy as np

from src.training.config import resolve_device


class GestureRecognizer:
    def __init__(self, path: Path, hand_model: Path, device: str):
        alphabet = import_module("src.hand-gestures.alphabet")
        self._video_landmarker = import_module(
            "src.hand-gestures.landmarks"
        ).video_landmarker
        self._min_scale = import_module("src.hand-gestures.features").MIN_SCALE
        import_module("mediapipe")

        run = path
        if not (run / "metadata.json").is_file():
            candidates = sorted(
                item.parent
                for item in path.glob("mlp_*/metadata.json")
                if item.with_name("best.pt").is_file()
            )
            if not candidates:
                raise FileNotFoundError(f"No completed Alphabet MLP runs in {path}")
            run = candidates[-1]
        self.device = str(resolve_device(device))
        metadata, self._predict = alphabet.load_predictor(run, self.device)
        if metadata["model"] != "mlp":
            raise ValueError(f"Expected an Alphabet MLP run: {run}")
        self.classes = metadata["classes"]
        self.checkpoint = run / "best.pt"
        self._hand_model = hand_model.read_bytes()

    @contextmanager
    def open_stream(self):
        with self._video_landmarker(self._hand_model) as detect:
            started = monotonic_ns()
            previous_ms = -1

            def predict(frame):
                nonlocal previous_ms
                elapsed_ms = (monotonic_ns() - started) // 1_000_000
                previous_ms = max(previous_ms + 1, elapsed_ms)
                points, _, _, _ = detect(frame, previous_ms)
                height, width = frame.shape[:2]
                hand = points * np.array([1, height / width, 1], dtype=np.float32)
                if (
                    not np.isfinite(hand).all()
                    or np.linalg.norm(hand[9] - hand[0]) <= self._min_scale
                ):
                    return {"status": "no_hand", "label": None, "confidence": None}
                scores = self._predict(hand[None])[0]
                best = int(scores.argmax())
                confidence = float(scores[best])
                if not 0 <= confidence <= 1:
                    raise ValueError("Invalid ASL classifier confidence")
                return {
                    "status": "detected",
                    "label": self.classes[best],
                    "confidence": confidence,
                }

            yield predict
