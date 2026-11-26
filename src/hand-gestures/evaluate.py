"""Score isolated ASL clips and compare independently trained classifiers."""

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.datasets.catalog import write_csv, write_json

from .config import GestureConfig, load_config
from .data import Samples, load_samples
from .features import FEATURE_VERSION
from .models import load_predictor


def json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def manifest_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def schema_digest(samples: Samples) -> str:
    return json_digest(
        {
            "feature_version": FEATURE_VERSION,
            "columns": samples.columns,
            "classes": samples.classes,
            "shape": samples.inputs.shape[1:],
        }
    )


def preprocess_settings(config: GestureConfig) -> dict:
    return config.model_dump(
        mode="json",
        include={
            "sample_fps",
            "sequence_length",
            "include_body",
            "static_frames",
            "min_hand_coverage",
            "max_hand_gap_sec",
            "flip_train",
        },
    )


def sample_metadata(samples: Samples) -> dict:
    digest = hashlib.sha256(samples.inputs.tobytes())
    digest.update(samples.weights.tobytes())
    pairs = [
        (window, int(label))
        for window, label in zip(samples.window_ids, samples.labels, strict=True)
    ]
    return {
        "window_sha256": json_digest(pairs),
        "feature_sha256": digest.hexdigest(),
        "window_ids": list(samples.window_ids),
        "rejected": samples.rejected,
    }


def clip_groups(samples: Samples) -> dict[tuple[str, int], list[int]]:
    groups = defaultdict(list)
    for index, (clip_id, label) in enumerate(
        zip(samples.clip_ids, samples.labels, strict=True)
    ):
        groups[clip_id, int(label)].append(index)
    return dict(groups)


def aggregate(groups, probabilities: np.ndarray):
    labels = np.fromiter((label for _, label in groups), dtype=np.int64)
    averaged = np.stack([probabilities[rows].mean(axis=0) for rows in groups.values()])
    return labels, averaged


def clip_macro_f1(groups, probabilities: np.ndarray) -> float:
    labels, averaged = aggregate(groups, probabilities)
    return float(
        f1_score(
            labels,
            averaged.argmax(axis=1),
            labels=range(averaged.shape[1]),
            average="macro",
            zero_division=0,
        )
    )


def classification_report(
    samples: Samples, probabilities: np.ndarray
) -> tuple[dict, list]:
    if probabilities.shape != (len(samples.labels), len(samples.classes)) or (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("invalid classifier probabilities")
    groups = clip_groups(samples)
    labels, averaged = aggregate(groups, probabilities)
    predicted = averaged.argmax(axis=1)
    matrix = confusion_matrix(labels, predicted, labels=range(len(samples.classes)))
    support = matrix.sum(axis=1)
    recall = np.divide(
        matrix.diagonal(), support, out=np.zeros(len(support)), where=support != 0
    )
    windows = np.bincount(samples.labels, minlength=len(samples.classes))
    report = {
        "classes": list(samples.classes),
        "clip_label_pairs": len(labels),
        "parent_clips": len(set(samples.clip_ids)),
        "window_count": len(probabilities),
        "accuracy": float((labels == predicted).mean()),
        "macro_f1": clip_macro_f1(groups, probabilities),
        "per_class": {
            label: {
                "clips": int(support[i]),
                "windows": int(windows[i]),
                "recall": float(recall[i]),
            }
            for i, label in enumerate(samples.classes)
        },
        "confusion_matrix": matrix.tolist(),
        "rejected_clips": samples.rejected,
        "handedness": {},
    }
    predictions = []
    for (clip, target), indices, code, confidence in zip(
        groups, groups.values(), predicted, averaged.max(axis=1), strict=True
    ):
        sides = {samples.handedness[i] for i in indices}
        side = next(iter(sides)) if len(sides) == 1 else "mixed"
        predictions.append(
            {
                "clip_id": clip,
                "actual": samples.classes[target],
                "predicted": samples.classes[code],
                "confidence": float(confidence),
                "windows": len(indices),
                "handedness": side,
                "anchor_coverage": float(samples.anchor_coverage[indices].mean()),
                "segments": ",".join(sorted({samples.segments[i] for i in indices})),
            }
        )
        counts = report["handedness"].setdefault(side, {"clips": 0, "correct": 0})
        counts["clips"] += 1
        counts["correct"] += int(code == target)
    return report, predictions


def evaluate(config: GestureConfig, run: Path, split: str) -> dict:
    torch.set_num_threads(1)
    metadata, predict = load_predictor(run)
    samples = load_samples(config, split, metadata["model"])
    sample_info = sample_metadata(samples)
    if (
        metadata["schema"] != schema_digest(samples)
        or metadata["manifest_sha256"] != manifest_digest(config.manifest)
        or metadata["extractor"] != samples.extractor
        or metadata["preprocess"] != preprocess_settings(config)
        or (split == "val" and metadata["val"] != sample_info)
    ):
        raise ValueError(f"gesture data/schema differs from the training run: {run}")
    predict(samples.inputs[:1])
    outputs, timings_ms = [], []
    for sample in samples.inputs:
        started = perf_counter_ns()
        output = predict(sample[None, ...])
        timings_ms.append((perf_counter_ns() - started) / 1e6)
        outputs.append(output)
    report, predictions = classification_report(samples, np.concatenate(outputs))
    report.update(
        {
            "model": metadata["model"],
            "run": str(run.resolve()),
            "split": split,
            "manifest_sha256": metadata["manifest_sha256"],
            "extractor": metadata["extractor"],
            "sample_set": sample_info,
            "latency": {
                "scope": "CPU model only, one sample per call, one warmup",
                "cpu": platform.processor() or platform.machine(),
                "p50_ms": float(np.percentile(timings_ms, 50)),
                "p95_ms": float(np.percentile(timings_ms, 95)),
            },
        }
    )
    extraction_report = config.cache_dir / "extraction_report.json"
    if extraction_report.is_file():
        report["extraction_report"] = str(extraction_report)
    write_json(run / f"evaluation_{split}.json", report)
    write_csv(run / f"predictions_{split}.csv", predictions, list(predictions[0]))
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    reports = [evaluate(config, run, args.split) for run in args.run]
    dynamic = [report for report in reports if report["model"] != "static_xgb"]
    if (
        len(
            {
                (report["sample_set"]["window_sha256"], report["extractor"])
                for report in dynamic
            }
        )
        > 1
    ):
        raise ValueError(
            "dynamic runs were evaluated on different windows or extractors"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, reports)
    print(
        json.dumps(
            [
                {
                    key: report[key]
                    for key in (
                        "model",
                        "run",
                        "split",
                        "accuracy",
                        "macro_f1",
                        "parent_clips",
                        "latency",
                    )
                }
                for report in reports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
