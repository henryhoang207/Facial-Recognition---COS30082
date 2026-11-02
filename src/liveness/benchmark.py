"""Measure crop-to-score latency on the machine actually running this command."""

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import load_manifests
from .predictor import LivenessPredictor
from .provenance import checked_path, file_sha256


def benchmark(
    checkpoint,
    manifest,
    data_root,
    *,
    repeats=30,
    warmup=5,
    threads=10,
    machine_role="demo_laptop",
):
    torch.set_num_threads(threads)
    frame = load_manifests(manifest)
    examples = frame[frame.split == "val"].groupby("attack").head(3)
    if examples.empty:
        raise ValueError("benchmark needs real validation images")
    inputs = [
        (
            cv2.imdecode(
                np.frombuffer(checked_path(data_root, row.path).read_bytes(), np.uint8),
                cv2.IMREAD_COLOR,
            ),
            (row.x1, row.y1, row.x2, row.y2),
        )
        for row in examples.itertuples()
    ]
    if any(image is None for image, _ in inputs):
        raise ValueError("benchmark image cannot be decoded")
    predictor = LivenessPredictor(checkpoint, device="cpu")
    times = []
    for index in range(warmup + repeats):
        image, box = inputs[index % len(inputs)]
        started = time.perf_counter()
        predictor.predict_frame(image, box)
        elapsed = (time.perf_counter() - started) * 1000
        if index >= warmup:
            times.append(elapsed)
    return {
        "checkpoint_sha256": file_sha256(checkpoint),
        "machine_role": machine_role,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "threads": threads,
        "torch": str(torch.__version__),
        "repeats": repeats,
        "warmup": warmup,
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "scope": "crop + preprocessing + CPU forward",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    measurements = [
        benchmark(path, args.manifest, args.data_root) for path in args.checkpoint
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    print(json.dumps(measurements, indent=2))


if __name__ == "__main__":
    main()
