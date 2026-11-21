"""Train and score static landmark models on the ASL Alphabet image splits."""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src.datasets.catalog import write_csv, write_json
from src.training import new_run_dir
from src.training.config import resolve_device

from .alphabet_config import AlphabetConfig, load_config
from .alphabet_data import CLASSES, SPLITS
from .evaluate import json_digest, manifest_digest
from .features import MIN_SCALE, STATIC_COLUMNS
from .manifest import read_rows

MODELS = ("mlp", "xgb")
SCHEMA = json_digest(
    {
        "version": 1,
        "classes": CLASSES,
        "columns": STATIC_COLUMNS,
        "input": "image_width",
    }
)


@dataclass(frozen=True)
class Dataset:
    points: np.ndarray
    labels: np.ndarray
    paths: np.ndarray
    splits: np.ndarray
    fingerprint: str
    manifest_hash: str
    ready: dict


def load_dataset(config: AlphabetConfig) -> Dataset:
    manifest = config.cache_dir / "manifest.csv"
    digest = manifest_digest(manifest)
    report = json.loads(
        (config.cache_dir / "extraction_report.json").read_text(encoding="utf-8")
    )
    if report["identity"].get("landmark_device", "cpu") != config.landmark_device:
        raise ValueError("landmark cache was extracted with a different device")
    ready = {}
    for row in read_rows(manifest):
        if row["status"] != "ready":
            continue
        if row["path"] in ready or row["split"] not in SPLITS:
            raise ValueError("invalid path or split in the ASL Alphabet manifest")
        ready[row["path"]] = (row["split"], CLASSES.index(row["label"]))
    with np.load(config.cache_dir / "landmarks.npz", allow_pickle=False) as cache:
        if cache["manifest_sha256"].item() != digest:
            raise ValueError(
                "ASL Alphabet manifest has changed. Run preprocessing again."
            )
        points, labels, paths, splits = (
            cache[name] for name in ("points", "labels", "paths", "splits")
        )
        fingerprint = str(cache["fingerprint"].item())
        if report["fingerprint"] != fingerprint or report["manifest_sha256"] != digest:
            raise ValueError("landmark extraction report does not match the cache")
    count = len(paths)
    if (
        not count
        or points.shape != (count, 21, 3)
        or points.dtype != np.float32
        or labels.shape != (count,)
        or not np.issubdtype(labels.dtype, np.integer)
        or paths.shape != (count,)
        or splits.shape != (count,)
        or len(set(paths)) != count
        or not np.isfinite(points).all()
        or np.any(np.linalg.norm(points[:, 9] - points[:, 0], axis=1) <= MIN_SCALE)
    ):
        raise ValueError("invalid landmark cache")
    if any(
        ready.get(path) != (split, label)
        for path, split, label in zip(paths, splits, labels, strict=True)
    ):
        raise ValueError("landmark cache labels or splits differ from the manifest")
    return Dataset(points, labels, paths, splits, fingerprint, digest, ready)


def train(
    config: AlphabetConfig, data: Dataset, model: str, resume: Path | None
) -> Path:
    if resume is not None and model != "mlp":
        raise ValueError("resume is available for the MLP checkpoint")
    metadata = {
        "schema": SCHEMA,
        "model": model,
        "classes": list(CLASSES),
        "columns": list(STATIC_COLUMNS),
        "fingerprint": data.fingerprint,
        "manifest_sha256": data.manifest_hash,
        "parameters": config.model_dump(
            mode="json",
            exclude={
                "dataset_root",
                "cache_dir",
                "artifacts_dir",
                "hand_model",
                "extraction_workers",
                "training",
            },
        ),
    }
    run = new_run_dir(config.artifacts_dir, model)
    print(f"ASL Alphabet {model} run: {run}", flush=True)
    if model == "mlp":
        from .alphabet_mlp import train as fit

        fit(config, data.points, data.labels, data.splits, metadata, run, resume=resume)
    else:
        from .alphabet_xgb import train as fit

        fit(config, data.points, data.labels, data.splits, metadata, run)
    write_json(run / "resolved_config.json", config.model_dump(mode="json"))
    write_json(run / "metadata.json", metadata)
    return run


def load_predictor(run: Path, device: str = "cpu"):
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != SCHEMA
        or metadata.get("classes") != list(CLASSES)
        or metadata.get("columns") != list(STATIC_COLUMNS)
    ):
        raise ValueError(f"incompatible ASL Alphabet artifact: {run}")
    if metadata["model"] == "mlp":
        from .alphabet_mlp import load_alphabet_predictor

        checkpoint_metadata, predict = load_alphabet_predictor(run, device)
        if checkpoint_metadata != metadata:
            raise ValueError(f"ASL Alphabet checkpoint metadata mismatch: {run}")
    elif metadata["model"] == "xgb":
        from .alphabet_xgb import load_alphabet_xgb

        predict = load_alphabet_xgb(run, metadata, device)
    else:
        raise ValueError(f"unknown ASL Alphabet model: {metadata['model']}")
    return metadata, predict


def evaluate(data: Dataset, run: Path, split: str, device: str = "cpu") -> dict:
    device = str(resolve_device(device))
    if device == "cpu":
        torch.set_num_threads(1)
    metadata, predict = load_predictor(run, device)
    if (
        metadata["fingerprint"] != data.fingerprint
        or metadata["manifest_sha256"] != data.manifest_hash
    ):
        raise ValueError(
            "evaluation data differs from the training manifest or extractor"
        )
    mask = data.splits == split
    points, targets, paths = data.points[mask], data.labels[mask], data.paths[mask]
    if not len(targets):
        raise ValueError(f"no usable ASL Alphabet images in {split}")
    probabilities = np.concatenate(
        [predict(points[i : i + 512]) for i in range(0, len(points), 512)]
    )
    if (
        probabilities.shape != (len(targets), len(CLASSES))
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("invalid ASL Alphabet model probabilities")
    predicted = probabilities.argmax(axis=1)
    correct = int((targets == predicted).sum())
    indices = range(len(CLASSES))
    matrix = confusion_matrix(targets, predicted, labels=indices)
    eligible = Counter(
        label for source, label in data.ready.values() if source == split
    )
    detected = matrix.sum(axis=1)
    eligible_per_class = np.array([eligible[i] for i in indices])
    f1_denominator = eligible_per_class + matrix.sum(axis=0)
    f1_with_detection = np.divide(
        2 * matrix.diagonal(),
        f1_denominator,
        out=np.zeros(len(CLASSES)),
        where=f1_denominator > 0,
    )
    report = {
        "run": str(run.resolve()),
        "model": metadata["model"],
        "split": split,
        "device": device,
        "eligible_images": eligible.total(),
        "detected_images": len(targets),
        "accuracy": correct / len(targets),
        "accuracy_with_extraction_failures": correct / eligible.total(),
        "macro_f1": float(
            f1_score(
                targets, predicted, labels=indices, average="macro", zero_division=0
            )
        ),
        "macro_f1_with_extraction_failures": float(f1_with_detection.mean()),
        "per_class": {
            label: {
                "eligible": eligible[i],
                "detected": int(detected[i]),
                "correct": int(matrix[i, i]),
                "recall": float(matrix[i, i] / detected[i]) if detected[i] else None,
                "recall_with_extraction_failures": (
                    float(matrix[i, i] / eligible[i]) if eligible[i] else None
                ),
            }
            for i, label in enumerate(CLASSES)
        },
        "classes": list(CLASSES),
        "confusion_matrix": matrix.tolist(),
    }
    write_json(run / f"evaluation_{split}.json", report)
    predictions = [
        {
            "path": path,
            "actual": CLASSES[actual],
            "predicted": CLASSES[guess],
            "confidence": float(scores[guess]),
            **{
                f"p_{label}": float(score)
                for label, score in zip(CLASSES, scores, strict=True)
            },
        }
        for path, actual, guess, scores in zip(
            paths, targets, predicted, probabilities, strict=True
        )
    ]
    write_csv(run / f"predictions_{split}.csv", predictions, list(predictions[0]))
    return report


def summary(report: dict) -> dict:
    return {
        key: value
        for key, value in report.items()
        if key not in {"per_class", "classes", "confusion_matrix"}
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--model", choices=MODELS, required=True)
    training.add_argument("--resume", type=Path)
    scoring = commands.add_parser("evaluate")
    scoring.add_argument("--run", type=Path, action="append", required=True)
    scoring.add_argument("--split", choices=SPLITS[1:], required=True)
    scoring.add_argument("--output", type=Path)
    for command in (training, scoring):
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_dataset(config)
    device = config.training.device
    if args.command == "train":
        run = train(config, data, args.model, args.resume)
        reports = [evaluate(data, run, "val", device)]
    else:
        reports = [evaluate(data, run, args.split, device) for run in args.run]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, reports)
    print(json.dumps([summary(report) for report in reports], indent=2))


if __name__ == "__main__":
    main()
