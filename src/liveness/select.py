"""Freeze candidates and choose using validation ACER and demo-laptop latency only."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from .provenance import file_sha256


def select_candidates(checkpoints, benchmarks=()):
    candidates = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config.get("max_train") is not None or config.get("max_val") is not None:
            raise ValueError("pilot checkpoints cannot become final candidates")
        if not config.get("provenance"):
            raise ValueError("candidate lacks audited provenance")
        candidates.append(
            {
                "checkpoint": str(path),
                "sha256": file_sha256(path),
                "model": config["model"],
                "val_acer": checkpoint["val_metrics"]["val_acer"],
                "threshold": checkpoint["threshold"],
                "manifest_sha256": config["provenance"]["manifest_sha256"],
            }
        )
    if len({item["manifest_sha256"] for item in candidates}) != 1:
        raise ValueError("candidates must use the same data protocol")
    by_model = {}
    for item in sorted(candidates, key=lambda item: item["val_acer"]):
        by_model.setdefault(item["model"], item)
    if set(by_model) != {"mobilenetv3", "deeppixbis"}:
        raise ValueError("freeze requires both architectures")
    finalists = sorted(by_model.values(), key=lambda item: item["val_acer"])
    timings = {
        item["checkpoint_sha256"]: item["median_ms"]
        for item in benchmarks
        if item["machine_role"] == "demo_laptop"
    }
    winner = finalists[0]
    status = "selected_by_validation_acer"
    if finalists[1]["val_acer"] - finalists[0]["val_acer"] <= 0.02:
        if all(item["sha256"] in timings for item in finalists):
            winner = min(finalists, key=lambda item: timings[item["sha256"]])
            status = "selected_by_demo_laptop_latency_within_2pp"
        else:
            winner, status = None, "pending_demo_laptop_latency"
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "selected_sha256": winner["sha256"] if winner else None,
        "candidates": finalists,
        "rule": (
            "validation ACER; within 2 percentage points "
            "use measured demo-laptop CPU median"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmarks", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    measurements = json.loads(args.benchmarks.read_text()) if args.benchmarks else []
    result = select_candidates(args.checkpoint, measurements)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
