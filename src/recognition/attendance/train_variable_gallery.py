"""Fit a Model A attendance MLP on simulated galleries of different sizes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.datasets.catalog import stable_rng, write_json
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.prepare import digest
from src.training.config import resolve_device

from .config import read_config
from .features import VARIABLE_GALLERY_FEATURES, features_from_scores, gallery_template, unit_rows
from .fixed_gallery import FixedGalleryPreparer, outcome_counts
from .model import AttendanceMLP
from .prepare import rows
from .train import select_gallery_threshold

WIDTHS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128)
PHASES = ("train", "dev", "test")


def simulate(config, seed):
    metadata = json.loads((config.dataset_dir / "dataset.json").read_text())
    encoder_hash = digest(config.encoder_checkpoint)
    if encoder_hash != metadata["encoder_sha256"] or metadata["frame_count"] != 5:
        raise ValueError("Use five-frame manifests made by the configured encoder")
    source = metadata["source"]
    enrollment = {split: defaultdict(list) for split in PHASES}
    probes = {split: set() for split in PHASES}
    for filename in ("enrollment.csv", "attempts.csv"):
        if digest(config.dataset_dir / filename) != metadata["sha256"][filename]:
            raise ValueError(f"Manifest checksum mismatch: {filename}")
    for row in rows(config.dataset_dir / "enrollment.csv"):
        enrollment[row["split"]][row["identity"]].append(row["path"])
    for row in rows(config.dataset_dir / "attempts.csv"):
        probes[row["split"]].add((row["identity"], tuple(json.loads(row["frame_paths"]))))
    labels = [set(enrollment[split]) | {name for name, _ in probes[split]} for split in PHASES]
    if sum(map(len, labels)) != len(set.union(*labels)):
        raise ValueError("Identities overlap across splits")
    enrollment_refs = {ref for split in PHASES for refs in enrollment[split].values() for ref in refs}
    probe_refs = {ref for split in PHASES for _, window in probes[split] for ref in window}
    if enrollment_refs & probe_refs:
        raise ValueError("Enrollment frames overlap with probe frames")
    references = enrollment_refs | probe_refs
    device = resolve_device(config.embedding_device)
    task, saved = load_task(config.encoder_checkpoint, device)
    preprocess = PreprocessConfig.model_validate(saved["metadata"]["preprocess_config"])
    vectors = FixedGalleryPreparer(config).embed_references(
        references, task.model.encoder, preprocess, device
    )
    if vectors.keys() != references:
        raise ValueError("Missing embeddings for the selected manifests")

    for split in PHASES:
        identities = sorted(enrollment[split])
        indices = {name: index for index, name in enumerate(identities)}
        gallery = np.stack([
            gallery_template(np.stack([vectors[ref] for ref in enrollment[split][name]]))
            for name in identities
        ])
        rng = stable_rng(seed, f"variable-gallery:{source}:{split}")
        for identity, window in sorted(probes[split]):
            true_index = indices.get(identity)
            frames = unit_rows([vectors[ref] for ref in window])
            scores = np.clip(frames @ gallery.T, -1, 1)
            other = [index for index in range(len(identities)) if index != true_index]
            hard = sorted(other, key=scores.mean(axis=0).__getitem__, reverse=True)
            random = rng.sample(other, len(other))
            for width in WIDTHS:
                if width > len(other):
                    break
                for mode, candidates in (("random", random), ("hard", hard)):
                    for enrolled in (False, True) if true_index is not None else (False,):
                        roster = candidates[:width - 1] + [true_index] if enrolled else candidates[:width]
                        roster = sorted(roster)
                        winner, features = features_from_scores(scores[:, roster], include_gallery_size=True)
                        yield split, (source, width, mode), features, enrolled, roster[winner] == true_index


def phase_data(records):
    others, features, enrolled, correct = zip(*records, strict=True)
    groups = defaultdict(list)
    for index, group in enumerate(others):
        groups[group].append(index)
    masks = {}
    for group, indices in groups.items():
        mask = np.zeros(len(records), dtype=bool)
        mask[indices] = True
        masks[group] = mask
    return torch.from_numpy(np.stack(features)), np.array(enrolled), np.array(correct), masks


def train_and_evaluate(configs, output, epochs, seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    encoder_hashes = {digest(config.encoder_checkpoint) for config in configs}
    if len(encoder_hashes) != 1:
        raise ValueError("Every source must use the same encoder")
    records = defaultdict(list)
    for cfg in configs:
        for split, *record in simulate(cfg, seed):
            records[split].append(record)
        print(f"Prepared {cfg.dataset_dir.name}", flush=True)

    train_x, _, train_correct, train_groups = phase_data(records.pop("train"))
    dev_x, dev_enrolled, dev_correct, dev_groups = phase_data(records.pop("dev"))
    labels = torch.from_numpy(train_correct).float()
    weights = torch.zeros(len(train_x))
    for mask in train_groups.values():
        weights[mask] = 1 / mask.sum()
    weights /= weights.sum()
    model = AttendanceMLP(16, len(VARIABLE_GALLERY_FEATURES))
    model.mean.copy_(weights @ train_x)
    model.scale.copy_((weights @ (train_x - model.mean).square()).sqrt().clamp_min(1e-6))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0001)
    best_rank = (-1.0, float("-inf"))
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.multinomial(weights, len(train_x), replacement=True)
        for batch in order.split(512):
            optimizer.zero_grad()
            loss = model.step((train_x[batch], labels[batch])).loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            scores = model(dev_x).sigmoid().numpy().astype(np.float64)
        threshold = select_gallery_threshold(scores, dev_correct, dev_enrolled, 0.01, dev_groups.values())
        accepts = scores >= threshold
        correct_rate = np.mean([
            (accepts & mask & dev_correct).sum() / (mask & dev_enrolled).sum()
            for mask in dev_groups.values()
        ])
        rank = (correct_rate, -int((accepts & ~dev_correct).sum()))
        if rank > best_rank:
            best_rank = rank
            best_epoch, best_threshold = epoch, threshold
            best_dev_scores = scores
            best_state = {name: value.clone() for name, value in model.state_dict().items()}
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, best dev correct rate {best_rank[0]:.3f}", flush=True)

    model.load_state_dict(best_state)
    test_x, test_enrolled, test_correct, test_groups = phase_data(records.pop("test"))
    with torch.inference_mode():
        test_scores = model(test_x).sigmoid().numpy().astype(np.float64)
    checkpoint = {
        "metadata": {
            "task": "attendance_confidence", "feature_names": VARIABLE_GALLERY_FEATURES,
            "encoder_sha256": encoder_hashes.pop(), "frame_count": 5, "gallery_size": None,
            "hidden_dim": 16, "simulated_gallery_sizes": sorted({key[1] for key in train_groups}),
        },
        "task": best_state, "threshold": best_threshold,
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "attendance.pt")
    metrics = {}
    for name, scores, enrolled, correct, groups in (
        ("dev", best_dev_scores, dev_enrolled, dev_correct, dev_groups),
        ("test", test_scores, test_enrolled, test_correct, test_groups),
    ):
        accepts = scores >= best_threshold
        metrics[name] = {
            f"{source}/{width}/{mode}": outcome_counts(accepts, mask, correct, enrolled)
            for (source, width, mode), mask in sorted(groups.items())
        }
    assert all(
        item["unknown_accepted"] <= int(0.01 * item["unknown"])
        and item["unknown_accepted"] + item["wrong_enrolled_accepted"] <= int(0.01 * (item["enrolled"] + item["unknown"]))
        for item in metrics["dev"].values()
    )
    write_json(output / "metrics.json", {
        "epoch": best_epoch, "epochs": epochs, "threshold": best_threshold, "seed": seed,
        "checkpoint_sha256": digest(output / "attendance.pt"),
        "input_sha256": {str(cfg.dataset_dir): digest(cfg.dataset_dir / "dataset.json") for cfg in configs},
        **metrics,
    })
    print(output / "attendance.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args()
    if args.epochs < 1 or args.output.exists():
        parser.error("Use positive epochs and a new output directory")
    train_and_evaluate([read_config(path) for path in args.configs], args.output, args.epochs, args.seed)
