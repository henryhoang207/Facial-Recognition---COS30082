"""Extract MediaPipe video landmarks and cache canonical ASLDynamic clips."""

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from uuid import uuid4
from zipfile import BadZipFile

import cv2
import numpy as np

from src.datasets.catalog import write_json

from .config import GestureConfig, load_config
from .manifest import read_rows

ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}
CACHE_VERSION = 1
ANCHOR_NAMES = ["nose", "mouth", "left_shoulder", "right_shoulder"]
REPORT_NAME = "extraction_report.json"
SMOKE_REPORT_NAME = "extraction_smoke_report.json"


def _body_anchors(result) -> np.ndarray:
    anchors = np.full((4, 2), np.nan, dtype=np.float32)
    if not result.pose_landmarks:
        return anchors
    pose = result.pose_landmarks[0]
    for index, ids in enumerate(((0,), (9, 10), (11,), (12,))):
        points = [pose[i] for i in ids]
        if all(
            (p.visibility is None or p.visibility >= 0.5)
            and (p.presence is None or p.presence >= 0.5)
            and 0 <= p.x <= 1
            and 0 <= p.y <= 1
            for p in points
        ):
            anchors[index] = np.mean([(p.x, p.y) for p in points], axis=0)
    return anchors


@contextmanager
def video_landmarker(hand_model: bytes, pose_model: bytes | None = None):
    """Open a tracker for one stream and yield its timestamped frame reader."""
    import mediapipe as mp

    vision = mp.tasks.vision
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=hand_model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    with ExitStack() as stack:
        hand = stack.enter_context(
            vision.HandLandmarker.create_from_options(hand_options)
        )
        pose = None
        if pose_model is not None:
            pose = stack.enter_context(
                vision.PoseLandmarker.create_from_options(
                    vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_buffer=pose_model
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_poses=1,
                    )
                )
            )

        def detect(frame: np.ndarray, milliseconds: int):
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            result = hand.detect_for_video(image, milliseconds)
            points = np.full((21, 3), np.nan, dtype=np.float32)
            body = np.full((4, 2), np.nan, dtype=np.float32)
            side, score = "", 0.0
            if result.hand_landmarks:
                points[:] = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
                if result.handedness and result.handedness[0]:
                    category = result.handedness[0][0]
                    side, score = category.category_name or "", category.score
                if pose is not None:
                    body = _body_anchors(pose.detect_for_video(image, milliseconds))
            return points, side, score, body

        yield detect


def extract_clip(
    path: Path,
    rotation: int,
    sample_fps: float,
    hand_model: bytes,
    pose_model: bytes | None,
) -> dict:
    """Return landmarks in the rotated frame and timestamps in seconds."""
    timestamps, hands, sides, scores, anchors = [], [], [], [], []
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError("invalid AVI frame rate")
        sample_rate = min(sample_fps, fps)
        with video_landmarker(hand_model, pose_model) as detect:
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = frame_index
                frame_index += 1
                if index < round(len(timestamps) * fps / sample_rate):
                    continue
                if rotation:
                    frame = cv2.rotate(frame, ROTATIONS[rotation])
                height, width = frame.shape[:2]
                timestamp = index / fps
                points, side, score, body = detect(frame, round(1000 * timestamp))
                timestamps.append(timestamp)
                hands.append(points)
                sides.append(side)
                scores.append(score)
                anchors.append(body)
        if not timestamps:
            raise ValueError("AVI produced no readable video frames")
        if expected_frames > frame_index:
            raise ValueError(
                f"decoded only {frame_index} of {expected_frames} AVI frames"
            )
    finally:
        capture.release()
    hands = np.stack(hands)
    anchors = np.stack(anchors)
    return {
        "timestamps": np.array(timestamps, dtype=np.float64),
        "hand": hands,
        "handedness": np.array(sides, dtype="U5"),
        "handedness_score": np.array(scores, dtype=np.float32),
        "anchors": anchors,
        "observed": np.isfinite(hands).all(axis=(1, 2)),
        "anchor_valid": np.isfinite(anchors).all(axis=2),
        "width": width,
        "height": height,
    }


def _check_cache(path: Path, metadata: dict) -> dict | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if any(data[key].item() != value for key, value in metadata.items()):
                return None
            timestamps = data["timestamps"]
            count = timestamps.size
            required = {
                "hand": (count, 21, 3),
                "handedness": (count,),
                "handedness_score": (count,),
                "anchors": (count, 4, 2),
                "observed": (count,),
                "anchor_valid": (count, 4),
                "width": (),
                "height": (),
            }
            if (
                timestamps.shape != (count,)
                or count == 0
                or not np.isfinite(timestamps).all()
                or not (np.diff(timestamps) > 0).all()
                or any(data[key].shape != shape for key, shape in required.items())
            ):
                return None
            return {"timestamps": timestamps, **{key: data[key] for key in required}}
    except (BadZipFile, EOFError, KeyError, OSError, ValueError):
        return None


def extract_manifest(config: GestureConfig, workers: int, limit: int | None) -> dict:
    if config.cache_dir.is_relative_to(config.dataset_root):
        raise ValueError("Cache directory must be outside the source dataset")
    hand_model = config.hand_model.read_bytes()
    pose_model = config.pose_model.read_bytes() if config.pose_model else None
    identity = {
        "schema": CACHE_VERSION,
        "python": python_version(),
        "opencv": cv2.__version__,
        "mediapipe": version("mediapipe"),
        "sample_fps": config.sample_fps,
        "hand_asset": hashlib.sha256(hand_model).hexdigest(),
        "pose_asset": hashlib.sha256(pose_model).hexdigest() if pose_model else None,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    rows = [row for row in read_rows(config.manifest) if row["status"] == "ready"]
    if not rows:
        raise ValueError(f"No ready recordings in {config.manifest}")
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_one(row: dict) -> dict:
        summary = {key: row[key] for key in ("clip_id", "label", "split")}
        sha256, rotation = row["sha256"], int(row["orientation"])
        source = (config.dataset_root / row["relative_path"]).resolve()
        try:
            if not source.is_relative_to(config.dataset_root):
                raise ValueError(f"Source path escapes dataset root: {source}")
            if rotation not in (0, *ROTATIONS):
                raise ValueError(f"Unsupported clockwise orientation: {rotation}")
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
                    raise ValueError("AVI hash changed. Rebuild the manifest.")
            cache = config.cache_dir / f"{sha256}.npz"
            metadata = {
                "extractor": fingerprint,
                "sha256": sha256,
                "rotation": rotation,
            }
            data = _check_cache(cache, metadata)
            if data is None:
                data = extract_clip(
                    source, rotation, config.sample_fps, hand_model, pose_model
                )
                temp = cache.with_name(f"{cache.stem}.{uuid4().hex}.tmp")
                try:
                    with temp.open("wb") as stream:
                        np.savez_compressed(stream, **data, **metadata)
                    os.replace(temp, cache)
                finally:
                    temp.unlink(missing_ok=True)
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            return {**summary, "error": str(exc)}
        observed = data["observed"]
        return {
            **summary,
            "frames": len(observed),
            "observed": int(observed.sum()),
            "duration_sec": float(data["timestamps"][-1] - data["timestamps"][0]),
            "anchor_observed": data["anchor_valid"][observed].sum(axis=0).tolist(),
        }

    reports = []
    rows = rows[:limit]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for count, result in enumerate(executor.map(cache_one, rows), 1):
            reports.append(result)
            if count % 20 == 0 or count == len(rows):
                print(f"Landmarks: {count}/{len(rows)}", flush=True)
    report = {
        "extractor": fingerprint,
        "identity": identity,
        "anchors": ANCHOR_NAMES,
        "failures": sum("error" in entry for entry in reports),
        "clips": reports,
    }
    filename = REPORT_NAME if limit is None else SMOKE_REPORT_NAME
    write_json(config.cache_dir / filename, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument(
        "--limit", type=int, help="Process only the first N ready clips."
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    config = load_config(args.config)
    report = extract_manifest(config, args.workers, args.limit)
    failures = report["failures"]
    filename = REPORT_NAME if args.limit is None else SMOKE_REPORT_NAME
    print(f"Extraction report: {config.cache_dir / filename}")
    if failures:
        raise SystemExit(
            f"{failures} recording(s) failed landmark extraction. See the report."
        )


if __name__ == "__main__":
    main()
