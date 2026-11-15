"""Train and evaluate the attendance MLP against two simple baselines."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.datasets.catalog import write_json
from src.recognition.preprocessing.prepare import digest
from src.recognition.verification import score_metrics
from src.training import (
    Trainer,
    make_optimizer,
    make_scheduler,
    new_run_dir,
    seed_everything,
)
from src.training.config import resolve_device, seed_worker

from .config import AttendanceConfig, load_config
from .features import FEATURES, TOP2_FEATURES
from .model import AttendanceMLP
from .prepare import rows


def read_attempts(path: Path, feature_names=FEATURES):
    splits = {name: [] for name in ("train", "dev", "test")}
    for row in rows(path):
        splits[row["split"]].append(row)
    data = {}
    for name, records in splits.items():
        x = np.asarray(
            [[float(r[key]) for key in feature_names] for r in records],
            dtype=np.float32,
        )
        y = np.asarray([int(r["label"]) for r in records])
        kinds = np.asarray([r["kind"] for r in records])
        if (
            set(y) != {0, 1}
            or not np.isfinite(x).all()
            or set(kinds) - {"correct", "wrong_known", "unknown"}
            or not (kinds == "unknown").any()
            or not np.array_equal(y == 1, kinds == "correct")
        ):
            raise ValueError(f"{name} has invalid features, labels, or attempt types")
        data[name] = x, y, kinds
    return data


def evaluate(labels, scores, kinds, target, threshold=None):
    scores = np.asarray(scores, dtype=np.float64)
    report, _ = score_metrics(labels, scores, target_fpr=target, threshold=threshold)
    accepted = scores >= report["score_threshold"]
    report["average_precision"] = float(average_precision_score(labels, scores))
    report["coverage"] = float(accepted.mean())
    report["wrong_share_of_accepted"] = (
        float((labels[accepted] == 0).mean()) if accepted.any() else None
    )
    enrolled = kinds != "unknown"
    report["correct_accept_rate_enrolled"] = float(
        (accepted & (labels == 1)).sum() / enrolled.sum()
    )
    report["wrong_known_accept_rate_enrolled"] = float(
        (accepted & (kinds == "wrong_known")).sum() / enrolled.sum()
    )
    report["unknown_accept_rate"] = float(accepted[kinds == "unknown"].mean())
    return report


def load_dataset(directory, cfg, encoder_hash):
    path = directory / "attempts.csv"
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    feature_names = tuple(metadata["feature_names"])
    if (
        feature_names not in (FEATURES, TOP2_FEATURES)
        or metadata["sha256"]["attempts.csv"] != digest(path)
        or metadata["encoder_sha256"] != encoder_hash
        or any(
            metadata[key] != getattr(cfg, key)
            for key in ("frame_count", "gallery_size", "enrollment_count")
        )
    ):
        raise ValueError(
            f"attendance dataset does not match its metadata or config: {directory}"
        )
    return metadata, read_attempts(path, feature_names)


def select_threshold(labels, scores, target, source_masks):
    if len(source_masks) == 1:
        return score_metrics(labels, scores, target_fpr=target)[0]["score_threshold"]
    scores = np.asarray(scores, dtype=np.float64)
    excluded = []
    for mask in source_masks.values():
        negatives = scores[mask & (labels == 0)]
        allowed = int(target * len(negatives))
        index = len(negatives) - 1 - allowed
        excluded.append(np.partition(negatives, index)[index])
    return float(np.nextafter(max(excluded), np.inf))


def source_sampling_weights(path: Path):
    keys = [
        (row["identity"], row["clip"], row["frame_paths"])
        for row in rows(path)
        if row["split"] == "train"
    ]
    windows = Counter(keys)
    clips = Counter((identity, clip) for identity, clip, _ in windows)
    identities = Counter(identity for identity, _ in clips)
    return [
        1
        / (
            len(identities)
            * identities[identity]
            * clips[identity, clip]
            * windows[key]
        )
        for key in keys
        for identity, clip, _ in (key,)
    ]


class FixedGalleryValidation:
    def __init__(self, path, feature_names, expected_sources, target, device):
        records = [row for row in rows(path) if row["split"] == "dev"]
        sources = {row["source"] for row in records}
        if sources != set(expected_sources):
            raise ValueError("fixed-gallery dev sources must match training sources")
        values = np.asarray(
            [[float(row[key]) for key in feature_names] for row in records],
            dtype=np.float32,
        )
        source, mode, kinds = (
            np.asarray([row[key] for row in records])
            for key in ("source", "gallery_mode", "kind")
        )
        if not np.isfinite(values).all() or not set(kinds) <= {
            "correct",
            "wrong_known",
            "unknown",
        }:
            raise ValueError("fixed-gallery dev has invalid features or labels")
        self.correct = kinds == "correct"
        self.enrolled = kinds != "unknown"
        self.strata = {
            (name, scenario): (source == name) & (mode == scenario)
            for name in sorted(sources)
            for scenario in ("random", "hard")
        }
        self.enrolled_masks = [mask & self.enrolled for mask in self.strata.values()]
        if set(mode) != {"random", "hard"} or any(
            not (mask & self.correct).any() or not (mask & ~self.enrolled).any()
            for mask in self.strata.values()
        ):
            raise ValueError(
                "each fixed-gallery dev stratum needs correct and guest trials"
            )
        self.inputs = torch.from_numpy(values).to(device)
        self.target = target

    def __call__(self, task, _device):
        scores = task(self.inputs).sigmoid().cpu().numpy().astype(np.float64)
        threshold = select_threshold(self.enrolled, scores, self.target, self.strata)
        accepted = scores >= threshold
        return {
            "val_gallery_correct_rate": float(
                np.mean(
                    [
                        (accepted & self.correct)[mask].mean()
                        for mask in self.enrolled_masks
                    ]
                )
            ),
            "val_gallery_wrong_rate": float(
                np.mean(
                    [
                        (accepted & ~self.correct)[mask].mean()
                        for mask in self.enrolled_masks
                    ]
                )
            ),
            "val_gallery_auc": float(roc_auc_score(self.correct, scores)),
            "val_gallery_threshold": threshold,
        }


def train(cfg: AttendanceConfig) -> Path:
    encoder_hash = digest(cfg.encoder_checkpoint)
    sources = {}
    for directory in (cfg.dataset_dir, *cfg.additional_datasets):
        metadata, data = load_dataset(directory, cfg, encoder_hash)
        name = metadata["source"]
        if name in sources:
            raise ValueError(f"duplicate attendance source: {name}")
        sources[name] = (metadata, data)
    metadata = sources[next(iter(sources))][0]
    feature_names = tuple(metadata["feature_names"])
    if any(
        tuple(info["feature_names"]) != feature_names for info, _ in sources.values()
    ):
        raise ValueError("all attendance datasets must use the same feature schema")
    combined = len(sources) > 1
    if combined:
        metadata = {
            **{
                key: metadata[key]
                for key in (
                    "encoder_sha256",
                    "encoder_epoch",
                    "feature_names",
                    "frame_count",
                    "gallery_size",
                    "enrollment_count",
                )
            },
            "source": "combined",
            "datasets": {name: info for name, (info, _) in sources.items()},
        }
    data, masks = {}, {}
    for split in ("train", "dev", "test"):
        records = [dataset[split] for _, dataset in sources.values()]
        data[split] = tuple(
            np.concatenate(column) for column in zip(*records, strict=True)
        )
        names = np.repeat(list(sources), [len(values) for values, _, _ in records])
        masks[split] = {name: names == name for name in sources}
    settings = cfg.training
    seed_everything(settings.seed, settings.deterministic)
    device = resolve_device(settings.device)
    fixed_validation = None
    if cfg.fixed_gallery_dev is not None:
        fixed_validation = FixedGalleryValidation(
            cfg.fixed_gallery_dev,
            feature_names,
            sources,
            cfg.target_false_accept,
            device,
        )
        metadata = {
            **metadata,
            "fixed_gallery_dev_sha256": digest(cfg.fixed_gallery_dev),
        }
    model = AttendanceMLP(cfg.hidden_dim, len(feature_names)).to(device)
    x, y, _ = data["train"]
    model.mean.copy_(torch.from_numpy(x.mean(axis=0)))
    scale = x.std(axis=0)
    model.scale.copy_(torch.from_numpy(np.where(scale > 0, scale, 1.0)))
    sampler = None
    if cfg.balance_training_sources:
        weights = [
            weight
            for directory in (cfg.dataset_dir, *cfg.additional_datasets)
            for weight in source_sampling_weights(directory / "attempts.csv")
        ]
        sampler = WeightedRandomSampler(
            weights, len(x), generator=torch.Generator().manual_seed(settings.seed)
        )
    loaders = [
        DataLoader(
            TensorDataset(torch.from_numpy(values), torch.from_numpy(labels).float()),
            batch_size=settings.batch_size,
            shuffle=name == "train" and sampler is None,
            sampler=sampler if name == "train" else None,
            num_workers=settings.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(settings.seed),
        )
        for name in ("train", "dev")
        for values, labels, _ in (data[name],)
    ]
    run = new_run_dir(cfg.output_dir, "attendance_mlp")
    optimizer = make_optimizer(model, settings)
    dev_values, dev_labels, _ = data["dev"]
    dev_inputs = torch.from_numpy(dev_values).to(device)

    def evaluate_dev(task, _device):
        if fixed_validation is not None:
            return fixed_validation(task, _device)
        scores = task(dev_inputs).sigmoid().cpu().numpy()
        threshold = select_threshold(
            dev_labels, scores, cfg.target_false_accept, masks["dev"]
        )
        metrics, _ = score_metrics(
            dev_labels,
            scores,
            target_fpr=cfg.target_false_accept,
            threshold=threshold,
        )
        return {"val_tar": metrics["tar"], "val_auc": metrics["auc"]}

    fit = Trainer(
        model,
        optimizer,
        settings,
        run,
        scheduler=make_scheduler(optimizer, settings),
        evaluator=evaluate_dev,
        metadata={
            **metadata,
            "task": "attendance_confidence",
            "hidden_dim": cfg.hidden_dim,
        },
    ).fit(*loaders)
    best = torch.load(fit.best_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(best["task"])
    model.cpu().eval()
    baseline = LogisticRegression(random_state=settings.seed)
    baseline.fit((x - model.mean.numpy()) / model.scale.numpy(), y)
    results = {}
    by_source = {}
    for name in (("dev", "test") if fixed_validation is None else ("dev",)):
        values, labels, kinds = data[name]
        with torch.inference_mode():
            mlp = model(torch.from_numpy(values)).sigmoid().numpy()
        logistic = baseline.predict_proba(
            (values - model.mean.numpy()) / model.scale.numpy()
        )[:, 1]
        cosine = np.where(
            values[:, feature_names.index("winner_fraction")] >= cfg.minimum_consensus,
            values[:, feature_names.index("cosine_mean")],
            -2.0,
        )
        scores_by_method = {
            "mlp": mlp,
            "logistic": logistic,
            "cosine_consensus": cosine,
        }
        results[name] = {}
        by_source[name] = {source: {} for source in sources}
        for method, scores in scores_by_method.items():
            threshold = (
                results["dev"][method]["score_threshold"]
                if name == "test"
                else select_threshold(
                    labels, scores, cfg.target_false_accept, masks[name]
                )
            )
            if fixed_validation is not None and method == "mlp":
                threshold = best["val_metrics"]["val_gallery_threshold"]
            metrics = evaluate(
                labels, scores, kinds, cfg.target_false_accept, threshold
            )
            if (
                name == "dev"
                and method == "cosine_consensus"
                and metrics["score_threshold"] <= -2.0
            ):
                metrics = evaluate(labels, scores, kinds, cfg.target_false_accept, -1.0)
            if method != "cosine_consensus":
                metrics["brier"] = float(brier_score_loss(labels, scores))
            results[name][method] = metrics
            if combined:
                for source, mask in masks[name].items():
                    by_source[name][source][method] = evaluate(
                        labels[mask],
                        scores[mask],
                        kinds[mask],
                        cfg.target_false_accept,
                        metrics["score_threshold"],
                    )
    torch.save(
        {**best, "threshold": results["dev"]["mlp"]["score_threshold"]},
        run / "attendance.pt",
    )
    write_json(run / "metrics.json", results)
    if combined:
        write_json(run / "metrics_by_source.json", by_source)
    write_json(run / "resolved_config.json", cfg.model_dump(mode="json"))
    return run


if __name__ == "__main__":
    print(train(load_config(__doc__)))
