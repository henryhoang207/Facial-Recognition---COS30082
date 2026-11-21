"""Build a file-disjoint ASL Alphabet manifest and extract image landmarks."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from string import ascii_uppercase
from threading import local
from uuid import uuid4
from zipfile import BadZipFile

import cv2
import numpy as np

from src.datasets.catalog import (
    holdout,
    ordered_parallel,
    stable_rng,
    write_csv,
    write_json,
)

CLASSES = tuple(ascii_uppercase)
SPLITS = ("train", "val", "test", "external")
BLOCK_SIZE = 100
GUARD_SIZE = 10
MIN_PALM_SCALE = 1e-5
MIN_DETECTION_CONFIDENCE = 0.5
PAD_FRACTION = 0.1
CHUNK_SIZE = 512
MANIFEST_COLUMNS = [
    "path",
    "label",
    "number",
    "block",
    "split",
    "sha256",
    "status",
    "canonical_path",
]


def _dataset_directory(root: Path, name: str) -> Path:
    directory = root / name
    nested = directory / name
    if nested.is_dir():
        directory = nested
    if not directory.is_dir():
        raise ValueError(f"Expected the ASL Alphabet directory at {directory}")
    return directory


def _block_splits(blocks: set[int], seed: int) -> dict[int, str]:
    rng = stable_rng(seed, "asl-alphabet-blocks")
    remaining, test = holdout(list(blocks), 0.1, rng)
    train, validation = holdout(remaining, 1 / 9, rng)
    return {
        block: split
        for split, group in zip(SPLITS[:3], (train, validation, test), strict=True)
        for block in group
    }


def _guard(number: int, assignments: dict[int, str]) -> bool:
    block, offset = divmod(number - 1, BLOCK_SIZE)
    split = assignments[block]
    return (offset < GUARD_SIZE and assignments.get(block - 1, split) != split) or (
        offset >= BLOCK_SIZE - GUARD_SIZE and assignments.get(block + 1, split) != split
    )


def _unique_rows(rows: list[dict]) -> None:
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    for group in by_hash.values():
        if len({row["label"] for row in group}) != 1:
            for row in group:
                row["status"] = "conflicting_labels"
            continue
        active = [row for row in group if row["status"] == "ready"]
        if not active:
            continue
        canonical = max(active, key=lambda row: SPLITS.index(row["split"]))
        for row in active:
            row["canonical_path"] = canonical["path"]
            if row is not canonical:
                row["status"] = "duplicate"


def build_manifest(root: Path, seed: int, workers: int) -> list[dict]:
    train = _dataset_directory(root, "asl_alphabet_train")
    external = _dataset_directory(root, "asl_alphabet_test")
    examples = []
    for index, label in enumerate(CLASSES):
        paths = sorted((train / label).glob("*.jpg"))
        if not paths:
            raise ValueError(f"No numbered JPEGs for ASL class {label}")
        for path in paths:
            match = re.fullmatch(
                rf"{label}([1-9][0-9]*)\.jpg", path.name, re.IGNORECASE
            )
            if match is None:
                raise ValueError(f"Expected a numbered ASL image: {path}")
            examples.append((path, index, int(match[1])))
        external_path = external / f"{label}_test.jpg"
        if external_path.is_file():
            examples.append((external_path, index, 0))
    assignments = _block_splits(
        {(number - 1) // BLOCK_SIZE for _, _, number in examples if number}, seed
    )

    def row_for(item: tuple[Path, int, int]) -> dict:
        path, label, number = item
        block = (number - 1) // BLOCK_SIZE if number else -1
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        relative = path.relative_to(root).as_posix()
        return {
            "path": relative,
            "label": CLASSES[label],
            "number": number or "",
            "block": block if number else "",
            "split": assignments[block] if number else "external",
            "sha256": digest,
            "status": "boundary_guard"
            if number and _guard(number, assignments)
            else "ready",
            "canonical_path": relative,
        }

    rows = list(ordered_parallel(row_for, examples, workers))
    _unique_rows(rows)
    return rows


def _make_worker(asset: bytes, device: str) -> tuple[Callable, Callable]:
    import mediapipe as mp

    vision = mp.tasks.vision
    delegate = mp.tasks.BaseOptions.Delegate[device.upper()]
    base = mp.tasks.BaseOptions(model_asset_buffer=asset, delegate=delegate)
    if device == "gpu" and getattr(base.to_ctypes(), "delegate", None) != delegate:
        raise RuntimeError(
            "Installed MediaPipe ignores GPU selection. Use 0.10.32 or newer."
        )
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    state = local()
    landmarker_instances = []

    def detect(frame: np.ndarray):
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        return state.detector.detect(image)

    def extract(content: bytes) -> tuple[np.ndarray | None, str]:
        frame = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None, "unreadable"
        if not hasattr(state, "detector"):
            state.detector = vision.HandLandmarker.create_from_options(options)
            landmarker_instances.append(state.detector)
        height, width = frame.shape[:2]
        result = detect(frame)
        pad = 0
        if not result.hand_landmarks:
            pad = max(1, round(min(height, width) * PAD_FRACTION))
            frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            result = detect(frame)
        if not result.hand_landmarks:
            return None, "no_hand"
        points = np.asarray(
            [(point.x, point.y, point.z) for point in result.hand_landmarks[0]],
            np.float32,
        )
        points *= np.array([frame.shape[1], frame.shape[0], frame.shape[1]]) / width
        points[:, :2] -= pad / width
        if (
            not np.isfinite(points).all()
            or np.linalg.norm(points[9] - points[0]) < MIN_PALM_SCALE
        ):
            return None, "bad_landmarks"
        side = (
            result.handedness[0][0].category_name
            if result.handedness and result.handedness[0]
            else ""
        )
        return points, side or ""

    def close() -> None:
        for detector in landmarker_instances:
            detector.close()

    return extract, close


def _write_cache(path: Path, **arrays) -> None:
    temp = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _extract(
    root: Path, rows: list[dict], asset: bytes, workers: int, chunks: Path, device: str
) -> tuple[dict, list[dict]]:
    extract, close = _make_worker(asset, device)
    cv2.setNumThreads(1)

    def extract_row(row: dict) -> tuple[np.ndarray | None, str]:
        content = (root / row["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError(f"Image changed since manifest creation: {row['path']}")
        return extract(content)

    arrays = defaultdict(list)
    rejected = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, len(rows), CHUNK_SIZE):
                batch = rows[start : start + CHUNK_SIZE]
                checkpoint = chunks / f"{start:06d}.npz"
                cached = _read_chunk(checkpoint, len(batch))
                if cached is None:
                    points = np.full((len(batch), 21, 3), np.nan, np.float32)
                    info = np.empty(len(batch), dtype="U16")
                    for index, (landmarks, side) in enumerate(
                        pool.map(extract_row, batch)
                    ):
                        if landmarks is not None:
                            points[index] = landmarks
                        info[index] = side
                    _write_cache(checkpoint, points=points, info=info)
                else:
                    points, info = cached
                for row, landmarks, side in zip(batch, points, info, strict=True):
                    if not np.isfinite(landmarks).all():
                        rejected.append({"path": row["path"], "reason": str(side)})
                    else:
                        arrays["points"].append(landmarks)
                        arrays["labels"].append(CLASSES.index(row["label"]))
                        arrays["paths"].append(row["path"])
                        arrays["splits"].append(row["split"])
                        arrays["handedness"].append(side)
                count = start + len(batch)
                if count % 4096 == 0 or count == len(rows):
                    usable = len(arrays["points"])
                    print(
                        f"Image landmarks {count}/{len(rows)}. Usable {usable}.",
                        flush=True,
                    )
    finally:
        close()
    if not arrays["points"]:
        raise ValueError("MediaPipe did not produce usable landmarks for any image")
    return {key: np.asarray(values) for key, values in arrays.items()}, rejected


def _read_chunk(path: Path, count: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as chunk:
            points, info = chunk["points"], chunk["info"]
    except (OSError, KeyError, ValueError, BadZipFile):
        return None
    if points.shape == (count, 21, 3) and info.shape == (count,):
        return points, info
    return None


def prepare(config) -> dict:
    device = config.landmark_device
    if device == "gpu" and sys.platform != "linux":
        raise ValueError("MediaPipe GPU extraction requires Ubuntu. Use Ubuntu in WSL.")
    if config.cache_dir.is_relative_to(config.dataset_root):
        raise ValueError("Cache directory must be outside the ASL source dataset")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(config.dataset_root, config.seed, config.extraction_workers)
    manifest = config.cache_dir / "manifest.csv"
    write_csv(manifest, rows, MANIFEST_COLUMNS)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    asset = config.hand_model.read_bytes()
    identity = {
        "schema": 2,
        "opencv": cv2.__version__,
        "mediapipe": version("mediapipe"),
        "hand_sha256": hashlib.sha256(asset).hexdigest(),
        "min_detection_confidence": MIN_DETECTION_CONFIDENCE,
        "padding_retry_fraction": PAD_FRACTION,
        "chunk_size": CHUNK_SIZE,
    }
    if device == "gpu":
        identity.update(
            landmark_device=device, platform=sys.platform, numpy=np.__version__
        )
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cache = config.cache_dir / "landmarks.npz"
    report_path = config.cache_dir / "extraction_report.json"
    if cache.is_file() and report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            with np.load(cache, allow_pickle=False) as data:
                if (
                    data["fingerprint"].item() == report["fingerprint"] == fingerprint
                    and data["manifest_sha256"].item()
                    == report["manifest_sha256"]
                    == manifest_hash
                    and data["points"].shape == (report["usable"], 21, 3)
                ):
                    return report
        except (OSError, KeyError, ValueError, BadZipFile):
            pass
    ready = [row for row in rows if row["status"] == "ready"]
    print(
        f"Manifest {len(rows)} images. Extracting {len(ready)} on {device.upper()}.",
        flush=True,
    )
    chunk_key = hashlib.sha256((manifest_hash + fingerprint).encode()).hexdigest()
    chunks = config.cache_dir / "chunks" / chunk_key
    chunks.mkdir(parents=True, exist_ok=True)
    arrays, rejected = _extract(
        config.dataset_root, ready, asset, config.extraction_workers, chunks, device
    )
    report = {
        "identity": identity,
        "fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
        "usable": len(arrays["paths"]),
        "excluded_or_duplicate": dict(Counter(row["status"] for row in rows)),
        "support": {
            split: dict(
                Counter(
                    CLASSES[index]
                    for index in arrays["labels"][arrays["splits"] == split]
                )
            )
            for split in SPLITS
        },
        "rejected": rejected,
    }
    _write_cache(
        cache, **arrays, fingerprint=fingerprint, manifest_sha256=manifest_hash
    )
    write_json(report_path, report)
    return report


def main() -> None:
    from .alphabet_config import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    report = prepare(config)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rejected"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
