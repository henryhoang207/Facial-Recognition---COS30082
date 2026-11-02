"""Visible notebook processing previews and bounded integration-test settings."""

from __future__ import annotations

from dataclasses import replace
import gc
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ExperimentConfig
from .data import FaceDataset, PreparedData, PKBatchSampler, make_loader
from .losses import ArcFaceHead, TripletMetricLoss
from .models import IResNet34
from .training import configure_runtime, environment_compatibility, environment_details, seed_everything, write_json


def processing_preview(prepared: PreparedData, data_root: Path, output_path: Path):
    """Show raw RGB crops alongside the exact deterministic training input crop."""
    import matplotlib.pyplot as plt
    from PIL import Image
    sample = prepared.train.groupby("identity_id", sort=True).head(1).head(4)
    dataset = FaceDataset(sample, data_root, train=False)
    figure, axes = plt.subplots(len(sample), 2, figsize=(7, 3 * len(sample)), squeeze=False)
    for index, row in enumerate(sample.itertuples()):
        with Image.open(Path(data_root) / row.rel_path) as image:
            axes[index, 0].imshow(image.convert("RGB"))
        tensor = dataset[index]["image"]
        axes[index, 1].imshow(((tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1))
        axes[index, 0].set_title(f"Original: {row.identity_id}")
        axes[index, 1].set_title("Shared square crop: RGB 112 x 112")
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    return figure


def smoke_prepared(prepared: PreparedData) -> PreparedData:
    """Use small real-image subsets only for code-path verification, never accuracy claims."""
    frames = []
    for source, identity_count in [(prepared.train, 8), (prepared.dev, 3), (prepared.test, 3)]:
        identities = sorted(source.identity_id.unique())[:identity_count]
        frames.append(source[source.identity_id.isin(identities)].groupby("identity_id", sort=True).head(4).copy().reset_index(drop=True))
    train, dev, test = frames
    mapping = {identity: index for index, identity in enumerate(sorted(train.identity_id.unique()))}
    train["label"] = train.identity_id.map(mapping)

    def pairs(frame):
        groups = {identity: list(group.rel_path) for identity, group in frame.groupby("identity_id", sort=True)}
        records = []
        for identity, paths in groups.items():
            records.extend({"path_a": a, "path_b": b, "is_same": True,
                            "identity_a": identity, "identity_b": identity} for a, b in combinations(paths, 2))
        for first, second in combinations(groups, 2):
            records.extend({"path_a": a, "path_b": b, "is_same": False,
                            "identity_a": first, "identity_b": second} for a, b in product(groups[first], groups[second]))
        return pd.DataFrame(records)

    groups = [list(group.rel_path) for _, group in train.groupby("identity_id", sort=True)]
    probe = pd.DataFrame([{"anchor": paths[index], "positive": paths[(index + 1) % len(paths)],
                           "negative": groups[(group_index + 1) % len(groups)][index]}
                          for group_index, paths in enumerate(groups) for index in range(4)])
    return PreparedData(train, dev, test, pairs(dev), pairs(test), probe, mapping,
                        {**prepared.metadata, "smoke_subset": True, "smoke_counts": [len(frame) for frame in frames]})


def smoke_config(config: ExperimentConfig) -> ExperimentConfig:
    """Keep smoke artifacts and altered curriculum out of scientific experiment outputs."""
    return replace(config, artifacts_root=Path(config.artifacts_root) / "smoke",
                   epochs=5 if config.method == "arcface" else 2,
                   num_workers=0, warmup_epochs=0, minimum_epochs=2, confirmation_seeds=(),
                   max_batches_per_epoch=1, evaluation_batch_size=16,
                   gate_minimum_epochs=1, gate_consecutive_checks=1,
                   gate_ranking_rate=0.0, gate_mean_gap=-4.0, gate_feature_spread=-1.0)


def pilot_runtime(config: ExperimentConfig, prepared: PreparedData) -> ExperimentConfig:
    """Measure both objectives with fresh weights, choosing one common memory profile."""
    device = configure_runtime(config)
    records = []
    profile_path = Path(config.artifacts_root) / "shared" / "runtime_pilot.json"
    compatibility = environment_compatibility()
    previous_profile = None
    if profile_path.exists():
        import json
        previous_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if previous_profile.get("environment_compatibility") != compatibility:
            raise ValueError("Runtime pilot environment changed or compatibility metadata is missing. Use a new artifact root.")
    existing_runs = any(Path(config.artifacts_root).glob("*/runs/*/weights/last.pt"))
    if existing_runs and not profile_path.exists():
        raise RuntimeError("Existing checkpoints have no shared pilot profile. Use a new artifact root.")
    locked_profile = None
    if existing_runs:
        locked_profile = previous_profile
        if locked_profile["K"] != config.images_per_identity:
            raise ValueError("The existing comparison uses a different K. Use a new artifact root.")
    choices = [locked_profile["P"] if locked_profile else config.identities_per_batch]
    if not locked_profile and config.identities_per_batch > 4:
        choices.append(4)
    for identities_per_batch in choices:
        candidate = replace(config, identities_per_batch=identities_per_batch)
        success = True
        for method in ("triplet", "arcface"):
            model = head = optimizer = images = labels = embeddings = loss = None
            parameters = scaler = batch = None
            try:
                seed_everything(42)
                sampler = PKBatchSampler(prepared.train.label.to_numpy(), P=identities_per_batch,
                                         K=config.images_per_identity, batches_per_epoch=1)
                batch = next(iter(make_loader(prepared.train, config.data_root, train=True,
                                              batch_sampler=sampler, num_workers=0)))
                images, labels = batch["image"].to(device), batch["label"].to(device)
                model = IResNet34(config.embedding_dim).to(device).train()
                head = ArcFaceHead(len(prepared.train_labels), config.embedding_dim, margin=0.4, scale=config.scale).to(device) if method == "arcface" else None
                parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
                optimizer = torch.optim.AdamW([p for p in parameters if p.requires_grad], lr=config.learning_rate)
                scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                # Two updates include optimizer-state allocation and a real AMP backward.
                updates = 0
                attempts = 0
                for _ in range(12):
                    attempts += 1
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16,
                                        enabled=config.amp and device.type == "cuda"):
                        embeddings = model(images)
                    loss = (torch.nn.functional.cross_entropy(head(embeddings, labels), labels)
                            if head is not None else TripletMetricLoss(0.3)(embeddings, labels, strategy="random")[0])
                    if not torch.isfinite(loss):
                        raise FloatingPointError("The runtime pilot produced a non-finite loss.")
                    scaler.scale(loss).backward()
                    previous = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updates += int(scaler.get_scale() >= previous)
                    if updates >= 2:
                        break
                if updates < 2:
                    raise FloatingPointError("The AMP pilot could not complete two optimizer updates in twelve attempts.")
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                records.append({"method": method, "batch_size": candidate.batch_size, "passed": True,
                                "optimizer_updates": updates, "attempts": attempts, "loss": float(loss.detach()),
                                "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/2**20 if device.type == "cuda" else 0.0,
                                "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/2**20 if device.type == "cuda" else 0.0})
            except torch.cuda.OutOfMemoryError:
                success = False
                records.append({"method": method, "batch_size": candidate.batch_size, "passed": False, "reason": "CUDA out of memory"})
            finally:
                del model, head, optimizer, images, labels, embeddings, loss, parameters, scaler, batch
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if not success:
                break
        if success:
            write_json(profile_path, {
                "environment": environment_details(device), "records": records,
                "environment_compatibility": compatibility,
                "selected_batch_size": candidate.batch_size, "P": identities_per_batch,
                "K": config.images_per_identity, "scope": "Two bounded optimization steps per objective; not a trained model.",
            })
            return candidate
    raise RuntimeError("The common GPU profile failed. Free GPU memory or restart both experiments in a new artifact root; do not mix batch profiles.")
