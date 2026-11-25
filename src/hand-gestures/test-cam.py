"""Preview one trained ASL classifier on DroidCam Video."""

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch

from src.training.config import resolve_device

from .alphabet import MODELS as ALPHABET_MODELS
from .alphabet import load_predictor as load_alphabet_predictor
from .alphabet_config import AlphabetConfig
from .alphabet_config import load_config as load_alphabet_config
from .config import load_config as load_dynamic_config
from .features import (
    STATIC_COLUMNS,
    LandmarkClip,
    local_pose,
    motion_columns,
    pool_motion,
    sequence_columns,
    sequence_features,
    track_spans,
)
from .landmarks import video_landmarker
from .models import MODEL_NAMES, load_predictor

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "hand_gestures"
TITLE = "DroidCam | ASL recognition"


def camera_features(clip: LandmarkClip, model: str, settings: dict, window: float):
    if model in ALPHABET_MODELS:
        return clip.image_units()[-1:]
    if model == "static_xgb":
        return local_pose(clip.image_units()[-1:])

    max_gap = settings["max_hand_gap_sec"]
    track = clip.select(track_spans(clip, max_gap)[-1])
    if track.timestamps[-1] - track.timestamps[0] < window / 2:
        return None
    sequence = sequence_features(
        track,
        length=settings["sequence_length"],
        min_coverage=settings["min_hand_coverage"],
        max_gap=max_gap,
        include_body=settings["include_body"],
    )
    if model == "motion_xgb":
        sequence = pool_motion(sequence, settings["include_body"])
    return sequence[None]


def open_camera(camera: int):
    if sys.platform == "win32":
        backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)

    capture = None
    for backend in backends:
        candidate = cv2.VideoCapture(camera, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()

    if capture is None:
        capture = cv2.VideoCapture(camera)
    return capture


def preview(config, run: Path, camera: int, window: float) -> None:
    torch.set_num_threads(1)
    alphabet = isinstance(config, AlphabetConfig)
    if alphabet:
        device = str(resolve_device(config.training.device))
        metadata, predict = load_alphabet_predictor(run, device)
        settings = {"sample_fps": 20}
        pose_asset = None
    else:
        device = "cpu"
        metadata, predict = load_predictor(run)
        settings = metadata["preprocess"]
        body = metadata["model"] != "static_xgb" and settings["include_body"]
        columns = {
            "static_xgb": STATIC_COLUMNS,
            "motion_xgb": motion_columns(body),
            "mlp_gru": sequence_columns(body),
        }
        if list(columns[metadata["model"]]) != metadata["columns"]:
            raise ValueError(f"camera features do not match model artifact: {run}")
        if body and not config.pose_model:
            raise ValueError("this run requires a pose_model path in the config")
        pose_asset = config.pose_model.read_bytes() if body else None

    model = metadata["model"]
    hand_asset = config.hand_model.read_bytes()
    history_window = 0.0 if alphabet else window

    capture = open_camera(camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera {camera}. Connect the phone in the DroidCam "
                "client, activate DroidCam Video, and check the --camera index."
            )
        with video_landmarker(hand_asset, pose_asset) as detect:
            history = deque()
            last_tick = -1
            message = "Show one hand"
            timing = ""
            started = monotonic()
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("DroidCam stopped returning frames.")
                elapsed = monotonic() - started
                height, width = frame.shape[:2]
                tick = int(elapsed * settings["sample_fps"])
                if tick != last_tick:
                    last_tick = tick
                    detection_start = monotonic()
                    points, side, _, anchors = detect(frame, round(elapsed * 1000))
                    detection_ms = (monotonic() - detection_start) * 1000
                    timing = f"hand {detection_ms:.1f} ms"
                    history.append((elapsed, points, side, anchors))
                    while history[0][0] < elapsed - history_window:
                        history.popleft()
                    times, hands, sides, references = zip(*history, strict=True)
                    clip = LandmarkClip(
                        np.array(times),
                        np.stack(hands),
                        np.array(sides),
                        np.stack(references),
                        width,
                        height,
                    )
                    if clip.observed()[-1]:
                        if not alphabet:
                            message = "Collecting motion..."
                        try:
                            inference_start = monotonic()
                            inputs = camera_features(clip, model, settings, window)
                        except ValueError:
                            message = "Hold one hand in view"
                        else:
                            if inputs is not None:
                                scores = predict(inputs)[0]
                                inference_ms = (monotonic() - inference_start) * 1000
                                timing += f" | model {inference_ms:.1f} ms"
                                best = int(scores.argmax())
                                label = metadata["classes"][best].replace("_", " ")
                                message = f"{label}  {scores[best]:.0%}"
                    else:
                        message = "Show one hand"
                preview_frame = cv2.flip(frame, 1)
                cv2.rectangle(preview_frame, (0, 0), (width, 60), (0, 0, 0), -1)
                detail = f"{model} ({device}) | {timing} | Q / Esc"
                for y, line in ((23, message), (46, detail)):
                    cv2.putText(
                        preview_frame,
                        line,
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(TITLE, preview_frame)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    return
                if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    return
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML for the selected classifier.")
    parser.add_argument("--camera", type=int, default=0, help="DroidCam Video index.")
    parser.add_argument(
        "--window", type=float, default=2.0, help="Motion window in seconds."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        choices=(*ALPHABET_MODELS, *MODEL_NAMES),
        help="Use the latest run for this model (default: mlp).",
    )
    selection.add_argument(
        "--run", type=Path, help="Use a specific training run directory."
    )
    args = parser.parse_args()
    if args.camera < 0 or not math.isfinite(args.window) or args.window < 0.25:
        parser.error("--camera must be nonnegative and --window at least 0.25 seconds")
    try:
        run = args.run
        model = args.model or "mlp"
        if run is not None:
            metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
            model = metadata["model"]
        if model in ALPHABET_MODELS:
            load_config, filename = load_alphabet_config, "asl_alphabet.yaml"
        else:
            load_config, filename = load_dynamic_config, "asl_dynamic.yaml"
        config = load_config(args.config or CONFIG_DIR / filename)
        if run is None:
            matches = sorted(config.artifacts_dir.glob(f"{model}_*/metadata.json"))
            if not matches:
                raise ValueError(f"No {model} run found in {config.artifacts_dir}")
            run = matches[-1].parent
        preview(config, run, args.camera, args.window)
    except (OSError, RuntimeError, ValueError, cv2.error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
