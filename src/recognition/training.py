"""Reproducible training, checkpointing, margin sweeps, and final evaluation.

References: FaceNet https://arxiv.org/abs/1503.03832 and
ArcFace https://arxiv.org/abs/1801.07698. No pretrained weights are loaded.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import PreparedData, make_loader, PKBatchSampler
from .evaluation import (
    evaluate_identification, evaluate_probe, extract_embeddings,
    plot_sweep, save_training_graph, save_verification_graphs,
    subject_bootstrap_ci, verification_metrics,
)
from .losses import ArcFaceHead, TripletMetricLoss
from .models import IResNet34


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact atomically, preserving valid existing files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configure_runtime(config: ExperimentConfig) -> torch.device:
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A notebook may already have initialized its inter-op thread pool.
            pass
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select the mtrlearn kernel with CUDA-enabled PyTorch.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_details(device: torch.device) -> dict:
    import platform
    import sys
    import torchvision
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def _fingerprint(config: ExperimentConfig, prepared: PreparedData, margin: float, seed: int) -> str:
    settings = config.to_dict()
    settings.pop("resume")
    payload = {"config": settings, "margin": margin, "seed": seed, "data": prepared.metadata}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _optimizer(model: nn.Module, head: nn.Module | None, config: ExperimentConfig):
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": config.weight_decay},
        {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
    ], lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, trainable


def _learning_rate(config: ExperimentConfig, progress: float) -> float:
    warmup = min(config.warmup_epochs, config.epochs)
    if progress < warmup and warmup > 0:
        return config.learning_rate * (0.01 + 0.99 * progress / warmup)
    fraction = min(1.0, max(0.0, (progress - warmup) / max(config.epochs - warmup, 1e-12)))
    return config.minimum_learning_rate + 0.5 * (config.learning_rate - config.minimum_learning_rate) * (1 + math.cos(math.pi * fraction))


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    # These files are generated locally by this training pipeline, not downloaded.
    return torch.load(path, map_location=device, weights_only=False)


def _evaluation_frame(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    paths = set(pairs.path_a) | set(pairs.path_b)
    return frame[frame.rel_path.isin(paths)].reset_index(drop=True)


def train_run(config: ExperimentConfig, prepared: PreparedData, margin: float, seed: int = 42) -> dict:
    """Train one independent margin/seed, with epoch-boundary resume support."""
    device = configure_runtime(config)
    seed_everything(seed)
    tag = f"margin_{margin:.1f}_seed_{seed}"
    run_dir = config.output_dir / "runs" / tag
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    signature = _fingerprint(config, prepared, margin, seed)
    result_path = run_dir / "result.json"
    if result_path.exists() and config.resume:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["signature"] != signature:
            raise ValueError(f"Run configuration changed: {run_dir}. Use a new artifacts directory.")
        if not Path(result["best_checkpoint"]).exists():
            raise FileNotFoundError(result["best_checkpoint"])
        return result
    if (weights_dir / "last.pt").exists() and not config.resume:
        raise FileExistsError(f"Existing run must be resumed or placed in a new directory: {run_dir}")

    model = IResNet34(embedding_dim=config.embedding_dim).to(device)
    initial_hash = hashlib.sha256()
    for tensor in model.state_dict().values():
        initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    head = (ArcFaceHead(len(prepared.train_labels), config.embedding_dim,
                        margin=margin, scale=config.scale, seed=seed + 100_000).to(device)
            if config.method == "arcface" else None)
    triplet = TripletMetricLoss(margin=margin)
    optimizer, parameters = _optimizer(model, head, config)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    mining_generator = torch.Generator(device=device).manual_seed(seed + 200_000)
    sampler = PKBatchSampler(prepared.train.label.to_numpy(),
                             P=config.identities_per_batch, K=config.images_per_identity,
                             seed=seed)
    loader = make_loader(prepared.train, config.data_root, train=True, batch_sampler=sampler,
                         num_workers=config.num_workers, seed=seed)
    dev_frame = _evaluation_frame(prepared.dev, prepared.dev_pairs)
    history: list[dict] = []
    strategy = "random" if config.method == "triplet" else "arcface"
    switch_epoch: int | None = None
    gate_passes = 0
    patience_count = 0
    best_tar, best_auc = -1.0, -1.0
    best_epoch = 0
    start_epoch = 1
    elapsed_before = 0.0
    last = weights_dir / "last.pt"
    if last.exists():
        state = _load_checkpoint(last, device)
        if state["signature"] != signature:
            raise ValueError("Cannot resume a checkpoint from different data or settings.")
        model.load_state_dict(state["model"])
        if head is not None:
            head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        strategy, switch_epoch = state["strategy"], state["switch_epoch"]
        gate_passes, patience_count = state["gate_passes"], state["patience_count"]
        best_tar, best_auc, best_epoch = state["best_tar"], state["best_auc"], state["best_epoch"]
        elapsed_before = state["elapsed_seconds"]
        start_epoch = state["epoch"] + 1
        semi_hard_ready = (switch_epoch is None
                           or state["epoch"] - switch_epoch + 1 >= config.minimum_semi_hard_epochs)
        if (state["epoch"] >= config.minimum_epochs
                and patience_count >= config.patience and semi_hard_ready):
            # Finalize an interrupted early-stop epoch instead of training an extra epoch.
            start_epoch = config.epochs + 1
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda_rng"]])
        mining_generator.set_state(state["mining_rng"].cpu())
    write_json(run_dir / "config.json", {
        **config.to_dict(), "margin": margin, "run_seed": seed, "signature": signature,
        "initial_encoder_sha256": initial_hash.hexdigest(), "data": prepared.metadata,
        "environment": environment_details(device),
    })
    timer = time.perf_counter()
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_timer = time.perf_counter()
        sampler.set_epoch(epoch - 1)
        model.train()
        if head is not None:
            head.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums: dict[str, float] = {}
        batch_count = update_count = empty_batches = amp_skips = image_count = 0
        epoch_strategy = strategy
        planned_batches = min(len(loader), config.max_batches_per_epoch or len(loader))
        progress = tqdm(loader, total=planned_batches, desc=f"{config.method} m={margin} epoch={epoch}", leave=False)
        for batch_index, batch in enumerate(progress):
            if batch_index >= planned_batches:
                break
            learning_rate = _learning_rate(config, epoch - 1 + batch_index / planned_batches)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            labels = batch["label"].to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config.amp and device.type == "cuda"):
                embeddings = model(images)
            if head is None:
                loss, stats = triplet(embeddings, labels, strategy=epoch_strategy, generator=mining_generator)
                no_update = stats["selected_triplets"] == 0 or stats["active_triplets"] == 0
                empty_batches += int(stats["selected_triplets"] == 0)
            else:
                logits = head(embeddings, labels)
                loss = nn.functional.cross_entropy(logits, labels)
                stats = {"train_accuracy": float((logits.argmax(1) == labels).float().mean().detach())}
                no_update = False
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {config.method} loss at epoch {epoch}, batch {batch_index}.")
            if not no_update:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(parameters, config.gradient_clip, error_if_nonfinite=False)
                if not torch.isfinite(norm) and not scaler.is_enabled():
                    raise FloatingPointError("Non-finite gradients in float32 training.")
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < previous_scale
                update_count += int(not skipped)
                amp_skips += int(skipped)
            sums["train_loss"] = sums.get("train_loss", 0.0) + float(loss.detach())
            for key, value in stats.items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    sums[key] = sums.get(key, 0.0) + float(value)
            batch_count += 1
            image_count += images.shape[0]
            progress.set_postfix(loss=round(float(loss.detach()), 4), phase=epoch_strategy)
        if batch_count == 0:
            raise RuntimeError("No full P x K training batches were produced.")
        embeddings = extract_embeddings(model, dev_frame, config.data_root, device,
                                        batch_size=config.evaluation_batch_size, num_workers=config.num_workers)
        metrics = verification_metrics(embeddings, prepared.dev_pairs, target_fmr=config.target_fmr)
        tar = float(metrics["tar_at_fmr"])
        auc = float(metrics["auc"])
        strict_improvement = tar > best_tar
        improved = strict_improvement or (tar == best_tar and auc > best_auc)
        patience_count = 0 if strict_improvement else patience_count + 1
        row = {"epoch": epoch, "strategy": epoch_strategy,
               **{key: value / batch_count for key, value in sums.items()},
               "dev_tar": tar, "dev_auc": auc, "dev_eer": float(metrics["eer"]),
               "dev_threshold": float(metrics["threshold"]), "dev_fmr": float(metrics["fmr"]),
               "learning_rate": learning_rate, "batches_forward": batch_count,
               "images_forward": image_count, "optimizer_updates": update_count,
               "amp_skips": amp_skips, "empty_batch_rate": empty_batches / batch_count,
               "epoch_seconds": time.perf_counter() - epoch_timer,
               "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
               "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0}
        if config.method == "triplet":
            probe = evaluate_probe(model, prepared.train, prepared.probe_triplets, config.data_root,
                                   device, batch_size=config.evaluation_batch_size, num_workers=config.num_workers)
            row.update({f"probe_{key}": value for key, value in probe.items()})
            ready = (probe["ranking_rate"] >= config.gate_ranking_rate
                     and probe["mean_gap"] >= config.gate_mean_gap
                     and probe["feature_spread"] > config.gate_feature_spread)
            if strategy == "random":
                gate_passes = gate_passes + 1 if ready else 0
                if epoch >= config.gate_minimum_epochs and gate_passes >= config.gate_consecutive_checks:
                    strategy, switch_epoch, patience_count = "semi_hard", epoch + 1, 0
            row["curriculum_not_ready"] = epoch >= 15 and switch_epoch is None
            row["mining_signal_warning"] = (epoch_strategy == "semi_hard" and row["empty_batch_rate"] > 0.2
                                             and bool(history) and history[-1]["empty_batch_rate"] > 0.2)
        history.append(row)
        if improved:
            best_tar, best_auc, best_epoch = tar, auc, epoch
        state = {
            "signature": signature, "config": config.to_dict(), "margin": margin, "seed": seed,
            "epoch": epoch, "model": model.state_dict(), "head": head.state_dict() if head is not None else None,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "strategy": strategy, "switch_epoch": switch_epoch, "gate_passes": gate_passes,
            "patience_count": patience_count, "best_tar": best_tar, "best_auc": best_auc, "best_epoch": best_epoch,
            "dev_metrics": metrics, "data_metadata": prepared.metadata,
            "initial_encoder_sha256": initial_hash.hexdigest(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "mining_rng": mining_generator.get_state(), "elapsed_seconds": elapsed_before + time.perf_counter() - timer,
        }
        if improved:
            _save_checkpoint(weights_dir / "best.pt", state)
        _save_checkpoint(last, state)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_training_graph(pd.DataFrame(history), run_dir / "graphs" / "loss.png")
        print(f"{config.method} margin={margin} seed={seed} epoch={epoch}: loss={row['train_loss']:.4f}, dev TAR={tar:.4f}, phase={epoch_strategy}")
        semi_hard_ready = (switch_epoch is None or epoch - switch_epoch + 1 >= config.minimum_semi_hard_epochs)
        if epoch >= config.minimum_epochs and patience_count >= config.patience and semi_hard_ready:
            break
    total_updates = sum(item["optimizer_updates"] for item in history)
    if total_updates == 0:
        raise RuntimeError("No optimizer updates completed. Inspect AMP skips or empty mining before accepting this run.")
    best = _load_checkpoint(weights_dir / "best.pt", device)
    model.load_state_dict(best["model"])
    embeddings = extract_embeddings(model, dev_frame, config.data_root, device,
                                    batch_size=config.evaluation_batch_size, num_workers=config.num_workers)
    save_verification_graphs(embeddings, prepared.dev_pairs, run_dir / "graphs",
                             threshold=best["dev_metrics"]["threshold"], target_fmr=config.target_fmr)
    _save_checkpoint(weights_dir / "embedding.pt", {
        "model": {key: value.cpu() for key, value in model.state_dict().items()},
        "embedding_dim": config.embedding_dim, "method": config.method, "margin": margin,
        "seed": seed, "threshold": best["dev_metrics"]["threshold"],
        "preprocessing": {"size": 112, "scale": 1.0, "color": "RGB", "mean": [0.5]*3, "std": [0.5]*3},
        "data_metadata": prepared.metadata, "signature": signature,
    })
    result = {"method": config.method, "margin": margin, "seed": seed, "signature": signature,
              "best_epoch": best_epoch, "dev_tar": best_tar, "dev_auc": best_auc,
              "dev_eer": float(best["dev_metrics"]["eer"]),
              "threshold": float(best["dev_metrics"]["threshold"]), "switch_epoch": switch_epoch,
              "curriculum_completed": switch_epoch is not None and history[-1]["epoch"] >= switch_epoch,
              "best_checkpoint_strategy": best["history"][-1]["strategy"],
              "best_checkpoint_has_semi_hard_updates": any(
                  item["strategy"] == "semi_hard" and item["optimizer_updates"] > 0
                  for item in best["history"]),
              "epochs_completed": len(history), "elapsed_seconds": elapsed_before + time.perf_counter() - timer,
              "total_optimizer_updates": total_updates,
              "best_checkpoint": str(weights_dir / "best.pt"), "embedding_checkpoint": str(weights_dir / "embedding.pt"),
              "history_path": str(run_dir / "history.csv"), "initial_encoder_sha256": initial_hash.hexdigest()}
    write_json(result_path, result)
    del loader, model, optimizer, head, best, state, parameters
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_sweep(config: ExperimentConfig, prepared: PreparedData, margins=None,
              confirmation_seeds=None) -> pd.DataFrame:
    """Select margins using dev only, then repeat the selected margin with new seeds."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    margins = tuple(config.margins if margins is None else margins)
    confirmations = config.confirmation_seeds if confirmation_seeds is None else tuple(confirmation_seeds)
    rows = []
    for margin in margins:
        rows.append({**train_run(config, prepared, float(margin), config.seed), "stage": "screening"})
        pd.DataFrame(rows).to_csv(config.output_dir / "sweep_val.csv", index=False)
    screening = pd.DataFrame(rows)
    chosen = screening.sort_values(["dev_tar", "dev_auc", "best_epoch", "margin"],
                                    ascending=[False, False, True, True]).iloc[0]
    for seed in confirmations:
        if seed == config.seed:
            continue
        rows.append({**train_run(config, prepared, float(chosen.margin), int(seed)), "stage": "confirmation"})
        pd.DataFrame(rows).to_csv(config.output_dir / "sweep_val.csv", index=False)
    sweep = pd.DataFrame(rows)
    write_json(config.output_dir / "selection.json", {
        "method": config.method, "selected_margin": float(chosen.margin),
        "criterion": "dev TAR at target FMR; ties: AUC, earlier epoch, smaller margin",
        "target_fmr": config.target_fmr, "screening_seed": config.seed,
        "test_used_for_selection": False,
    })
    plot_sweep(sweep, config.output_dir / "graphs" / "sweep_val.png")
    return sweep


def evaluate_selected(config: ExperimentConfig, prepared: PreparedData, sweep: pd.DataFrame) -> pd.DataFrame:
    """Evaluate only the selected margin, preserving each seed's dev threshold."""
    device = configure_runtime(config)
    selection = json.loads((config.output_dir / "selection.json").read_text(encoding="utf-8"))
    selected = sweep[np.isclose(sweep.margin, selection["selected_margin"])]
    rows = []
    for record in selected.to_dict("records"):
        model = IResNet34(config.embedding_dim).to(device)
        state = _load_checkpoint(Path(record["best_checkpoint"]), device)
        model.load_state_dict(state["model"])
        dev = extract_embeddings(model, prepared.dev, config.data_root, device,
                                 batch_size=config.evaluation_batch_size, num_workers=config.num_workers)
        test = extract_embeddings(model, prepared.test, config.data_root, device,
                                  batch_size=config.evaluation_batch_size, num_workers=config.num_workers)
        verification = verification_metrics(test, prepared.test_pairs, target_fmr=config.target_fmr,
                                            threshold=float(record["threshold"]))
        enrolled = min(30, len(prepared.dev.identity_id.unique()) - 1,
                        len(prepared.test.identity_id.unique()) - 1)
        dev_open = evaluate_identification(dev, prepared.dev, enrolled=enrolled,
                                            target_fpir=config.target_fpir, seed=42)
        test_open = evaluate_identification(test, prepared.test, enrolled=enrolled, seed=42,
                                             target_fpir=config.target_fpir, threshold=dev_open["threshold"])
        result = {"method": config.method, "margin": float(record["margin"]), "seed": int(record["seed"]),
                  "verification": verification, "dev_open_set": dev_open, "test_open_set": test_open,
                  "verification_ci": subject_bootstrap_ci(test, prepared.test_pairs, float(record["threshold"]))}
        write_json(config.output_dir / "test" / f"seed_{int(record['seed'])}.json", result)
        save_verification_graphs(test, prepared.test_pairs,
                                 config.output_dir / "test" / f"seed_{int(record['seed'])}" / "graphs",
                                 threshold=float(record["threshold"]), target_fmr=config.target_fmr)
        rows.append({"method": config.method, "margin": record["margin"], "seed": record["seed"],
                     "test_auc": verification["auc"], "test_tar": verification["tar"],
                     "test_fmr": verification["fmr"], "test_eer": verification["eer"],
                     "threshold_from_dev": record["threshold"],
                     "test_rank1": test_open["rank1"], "test_tpir": test_open["tpir"],
                     "test_fpir": test_open["fpir"]})
        del model, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(config.output_dir / "test_results.csv", index=False)
    frame.select_dtypes("number").drop(columns=["seed", "margin"], errors="ignore").agg(["mean", "std"]).to_csv(config.output_dir / "test_summary.csv")
    return frame
