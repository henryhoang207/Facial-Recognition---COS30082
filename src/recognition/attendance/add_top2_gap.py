"""Extend existing attempts with the gap between the two highest mean scores."""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.datasets.catalog import write_csv, write_json
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.prepare import digest
from src.training.config import resolve_device

from .config import read_config
from .features import FEATURES, TOP2_FEATURES, attempt_features, gallery_template
from .fixed_gallery import FixedGalleryPreparer
from .prepare import rows
from .train import load_dataset


def extend_attempts(records, vectors, templates):
    largest_error = 0.0
    for row in records:
        gallery = json.loads(row["gallery"])
        frames = np.stack([vectors[ref] for ref in json.loads(row["frame_paths"])])
        winner, features = attempt_features(
            frames,
            np.stack([templates[name] for name in gallery]),
            include_top2_gap=True,
        )
        old = [float(row[key]) for key in FEATURES]
        error = float(np.max(np.abs(np.asarray(old) - features[:-1])))
        largest_error = max(largest_error, error)
        # Different embedding batch shapes can slightly change CUDA rounding.
        if gallery[winner] != row["candidate"] or error > 5e-4:
            raise ValueError(
                f"original attempt could not be reproduced: "
                f"{row.get('attempt_id', row.get('gallery_id'))} / {row['identity']}, "
                f"expected {row['candidate']}, got {gallery[winner]}, "
                f"expected {old}, got {features[:-1].tolist()}"
            )
        row["top2_gap"] = float(features[-1])
    return largest_error


def extend_source(config, fixed, output):
    source = config.dataset_dir
    encoder_hash = digest(config.encoder_checkpoint)
    metadata, _ = load_dataset(source, config, encoder_hash)
    attempts = list(rows(source / "attempts.csv"))
    combined = [*attempts, *fixed[metadata["source"]]]
    enrollment = defaultdict(list)
    for name in ("enrollment.csv", "identity_to_idx.json"):
        if digest(source / name) != metadata["sha256"][name]:
            raise ValueError(f"checksum mismatch: {source / name}")
    for row in rows(source / "enrollment.csv"):
        enrollment[row["identity"]].append(row["path"])
    paths = {ref for refs in enrollment.values() for ref in refs} | {
        ref for row in combined for ref in json.loads(row["frame_paths"])
    }
    device = resolve_device(config.embedding_device)
    task, saved = load_task(config.encoder_checkpoint, device)
    preprocess = PreprocessConfig.model_validate(saved["metadata"]["preprocess_config"])
    vectors = FixedGalleryPreparer(config).embed_references(
        paths, task.model.encoder, preprocess, device
    )
    if vectors.keys() != paths:
        raise ValueError("attempt embeddings are incomplete")
    templates = {
        name: gallery_template(np.stack([vectors[ref] for ref in refs]))
        for name, refs in enrollment.items()
    }
    error = extend_attempts(combined, vectors, templates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        staged.mkdir()
        write_csv(staged / "attempts.csv", attempts, list(attempts[0]))
        for name in ("enrollment.csv", "identity_to_idx.json"):
            shutil.copyfile(source / name, staged / name)
        metadata["feature_names"] = list(TOP2_FEATURES)
        metadata["base_attempts_sha256"] = metadata["sha256"]["attempts.csv"]
        metadata["max_feature_recompute_error"] = error
        metadata["sha256"]["attempts.csv"] = digest(staged / "attempts.csv")
        write_json(staged / "dataset.json", metadata)
        staged.rename(output)
    print(output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("CONFIG", "DATASET", "OUTPUT"),
    )
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--fixed-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = [Path(output).resolve() for _, _, output in args.source]
    if args.fixed_output.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("top2 output already exists")
    fixed_rows = list(rows(args.fixed))
    fixed = defaultdict(list)
    for row in fixed_rows:
        fixed[row["source"]].append(row)
    for (config, dataset, _), output in zip(args.source, outputs, strict=True):
        resolved = read_config(Path(config)).model_copy(
            update={"dataset_dir": Path(dataset).resolve()}
        )
        extend_source(resolved, fixed, output)
    if any("top2_gap" not in row for row in fixed_rows):
        raise ValueError("every fixed-gallery source must be provided")
    write_csv(args.fixed_output, fixed_rows, list(fixed_rows[0]))
    print(args.fixed_output)


if __name__ == "__main__":
    main()
