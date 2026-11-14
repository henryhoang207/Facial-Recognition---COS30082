"""Reproducible fixed-margin training, checkpointing, and final evaluation.

References: FaceNet https://arxiv.org/abs/1503.03832 and
ArcFace https://arxiv.org/abs/1801.07698. No pretrained weights are loaded.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import pickle
import platform
import random
import shutil
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.training import Trainer, TrainingPolicy

from .config import ExperimentConfig
from .data import PKBatchSampler, PreparedData, bbox_array, make_loader
from .embedding_evaluation import (
    evaluate_identification,
    evaluate_probe,
    extract_embeddings,
    plot_sweep,
    save_training_graph,
    save_verification_graphs,
    subject_bootstrap_ci,
    verification_metrics,
)
from .models import IResNet34
from .predictor import METRIC_PREPROCESSING
from .task import MetricLearningTask


def write_json(path: Path, value: Any) -> None:
    """Replace a JSON artifact atomically, preserving valid existing files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )
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
        raise RuntimeError(
            "CUDA is unavailable. Select the mtrlearn kernel with CUDA-enabled PyTorch."
        )
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


def environment_compatibility() -> dict:
    """Identify the numerical runtime without machine-specific installation paths.

    A changed runtime starts a new experiment rather than silently resuming or
    accepting results from a different Python, CUDA, or numerical-library stack.
    Python patch versions are intentionally compatible within one minor release.
    """
    packages = {}
    for name in ("torchvision", "duckdb", "Pillow", "opencv-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "schema": 1,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": platform.system(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def environment_details(device: torch.device) -> dict:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "compatibility": environment_compatibility(),
    }


def _fingerprint(
    config: ExperimentConfig,
    prepared: PreparedData,
    margin: float,
    seed: int,
    compatibility: dict | None = None,
) -> str:
    settings = config.to_dict()
    settings.pop("resume")
    payload = {
        "training_contract": "shared-trainer-v1",
        "config": settings,
        "margin": margin,
        "seed": seed,
        "data": prepared.metadata,
        "environment_compatibility": (
            environment_compatibility() if compatibility is None else compatibility
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _require_compatible_artifact(
    artifact: dict, compatibility: dict, signature: str, source: Path
) -> None:
    metadata = artifact.get("metadata", artifact)
    if metadata.get("environment_compatibility") != compatibility:
        raise ValueError(
            f"Run environment changed or compatibility metadata is missing: {source}. "
            "Use a new artifacts directory; old runs cannot be resumed "
            "or reused across runtimes."
        )
    if metadata.get("signature") != signature:
        raise ValueError(
            f"Run configuration changed (data or settings): {source}. "
            "Use a new artifacts directory."
        )


def _optimizer(model: nn.Module, head: nn.Module | None, config: ExperimentConfig):
    parameters = list(model.parameters()) + (
        list(head.parameters()) if head is not None else []
    )
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "weights",
                "params": [p for p in trainable if p.ndim >= 2],
                "weight_decay": config.weight_decay,
            },
            {
                "name": "scalars",
                "params": [p for p in trainable if p.ndim < 2],
                "weight_decay": 0.0,
            },
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True if config.fused_adamw else None,
    )
    return optimizer, trainable


def _learning_rate(config: ExperimentConfig, progress: float) -> float:
    warmup = min(config.warmup_epochs, config.epochs)
    if progress < warmup and warmup > 0:
        return config.learning_rate * (0.01 + 0.99 * progress / warmup)
    fraction = min(
        1.0, max(0.0, (progress - warmup) / max(config.epochs - warmup, 1e-12))
    )
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1 + math.cos(math.pi * fraction))


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
    temporary.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Read the shared safe checkpoint schema and reject private-loop artifacts."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Legacy metric checkpoint is not a shared Trainer artifact. "
            "Use a new artifacts directory."
        ) from exc
    if (
        not isinstance(state, dict)
        or state.get("format_version") != 1
        or state.get("checkpoint_kind") not in {"training", "inference"}
        or state.get("metadata", {}).get("task") != "recognition_metric_learning"
    ):
        raise ValueError(
            "Legacy or unsupported metric checkpoint schema. "
            "Use a new artifacts directory."
        )
    return state


def _evaluation_frame(frame: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    paths = set(pairs.path_a) | set(pairs.path_b)
    return frame[frame.rel_path.isin(paths)].reset_index(drop=True)


class MetricTrainingPolicy(TrainingPolicy):
    """Keep scientific protocol decisions separate from shared optimization."""

    def __init__(self, config: ExperimentConfig, prepared: PreparedData, run_dir: Path):
        self.config, self.prepared, self.run_dir = config, prepared, run_dir
        self.elapsed_before = 0.0
        self.started = time.perf_counter()
        self.switched = False
        # Preserve the pre-fit encoder-initialization RNG stream on fresh runs.
        self.initial_torch_rng = torch.get_rng_state()
        self.initial_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        )

    def on_epoch_start(self, trainer: Trainer, epoch: int, loader) -> None:
        del loader
        if epoch == 1 and self.initial_torch_rng is not None:
            torch.set_rng_state(self.initial_torch_rng)
            if self.initial_cuda_rng:
                torch.cuda.set_rng_state_all(self.initial_cuda_rng)
        self.initial_torch_rng = None
        self.initial_cuda_rng = []
        # Match the notebook protocol's deterministic best-effort setting.
        torch.use_deterministic_algorithms(True, warn_only=True)
        trainer.task.reset_epoch_counts()
        self.epoch_strategy = trainer.task.strategy
        self.switched = False
        self.epoch_started = time.perf_counter()
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_batch_start(
        self, trainer: Trainer, epoch: int, batch_index: int, batches: int
    ) -> None:
        learning_rate = _learning_rate(self.config, epoch - 1 + batch_index / batches)
        for group in trainer.optimizer.param_groups:
            group["lr"] = learning_rate

    def on_epoch_end(self, trainer: Trainer, epoch: int, row: dict) -> None:
        task = trainer.task
        batches = task.batches_forward
        updates = int(row["train_optimizer_steps"])
        row.update(
            {
                "strategy": self.epoch_strategy,
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "batches_forward": batches,
                "images_forward": task.images_forward,
                "optimizer_updates": updates,
                "amp_skips": batches - task.no_update_batches - updates,
                "empty_batch_rate": task.empty_batches / batches,
                "epoch_seconds": time.perf_counter() - self.epoch_started,
                "peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(trainer.device) / 2**20
                    if trainer.device.type == "cuda"
                    else 0.0
                ),
                "peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(trainer.device) / 2**20
                    if trainer.device.type == "cuda"
                    else 0.0
                ),
            }
        )
        if self.config.method == "triplet":
            for name, value in list(row.items()):
                if name.startswith("train_") and name not in {
                    "train_loss",
                    "train_optimizer_steps",
                }:
                    row[name.removeprefix("train_")] = value
            probe = evaluate_probe(
                task.model,
                self.prepared.train,
                self.prepared.probe_triplets,
                self.config.data_root,
                trainer.device,
                batch_size=self.config.evaluation_batch_size,
                **self.config.evaluation_loader_options,
            )
            row.update({f"probe_{key}": value for key, value in probe.items()})
            ready = (
                probe["ranking_rate"] >= self.config.gate_ranking_rate
                and probe["mean_gap"] >= self.config.gate_mean_gap
                and probe["feature_spread"] > self.config.gate_feature_spread
            )
            if task.strategy == "random":
                task.gate_passes = task.gate_passes + 1 if ready else 0
                if (
                    epoch >= self.config.gate_minimum_epochs
                    and task.gate_passes >= self.config.gate_consecutive_checks
                ):
                    task.strategy, task.switch_epoch = "semi_hard", epoch + 1
                    self.switched = True
            row["curriculum_not_ready"] = epoch >= 15 and task.switch_epoch is None
            row["mining_signal_warning"] = (
                self.epoch_strategy == "semi_hard"
                and row["empty_batch_rate"] > 0.2
                and bool(trainer.history)
                and trainer.history[-1]["empty_batch_rate"] > 0.2
            )
        save_training_graph(
            pd.DataFrame([*trainer.history, row]), self.run_dir / "graphs" / "loss.png"
        )

    def selection(self, trainer, row, best_row) -> tuple[bool, bool]:
        del trainer
        strict = best_row is None or row["dev_tar"] > best_row["dev_tar"]
        improved = strict or (
            row["dev_tar"] == best_row["dev_tar"]
            and row["dev_auc"] > best_row["dev_auc"]
        )
        return improved, strict or self.switched

    def should_stop(self, trainer: Trainer, epoch: int) -> bool:
        del epoch
        if not trainer.history:
            return False
        completed = int(trainer.history[-1]["epoch"])
        switch = trainer.task.switch_epoch
        semi_hard_ready = (
            switch is None
            or completed - switch + 1 >= self.config.minimum_semi_hard_epochs
        )
        return (
            completed >= self.config.minimum_epochs
            and trainer.bad_epochs >= self.config.patience
            and semi_hard_ready
        )

    def state_dict(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_before + time.perf_counter() - self.started
        }

    def load_state_dict(self, state: dict) -> None:
        self.elapsed_before = state["elapsed_seconds"]
        self.started = time.perf_counter()
        self.initial_torch_rng = None
        self.initial_cuda_rng = []


def _publish_checkpoint(source: Path, destination: Path) -> None:
    """Publish shared-engine files atomically without changing their schema."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        *run_dir.glob("weights/last.pt"),
        *run_dir.glob("trainer_runs/*/last.pt"),
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns, default=None)


def train_run(
    config: ExperimentConfig,
    prepared: PreparedData,
    margin: float,
    seed: int = 42,
) -> dict:
    """Run one margin through the shared Trainer with a metric-learning task."""
    if config.require_bboxes:
        for frame in (prepared.train, prepared.dev, prepared.test):
            bbox_array(frame, required=True)
    device = configure_runtime(config)
    seed_everything(seed)
    run_dir = config.output_dir / "runs" / f"margin_{margin:.1f}_seed_{seed}"
    weights_dir = run_dir / "weights"
    compatibility = environment_compatibility()
    signature = _fingerprint(config, prepared, margin, seed, compatibility)
    result_path = run_dir / "result.json"
    if result_path.exists() and config.resume:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _require_compatible_artifact(result, compatibility, signature, result_path)
        best_path = Path(result["best_checkpoint"])
        best = _load_checkpoint(best_path, torch.device("cpu"))
        _require_compatible_artifact(best, compatibility, signature, best_path)
        return result
    last = _resume_checkpoint(run_dir)
    initial_encoder_sha256 = None
    if last is not None:
        if not config.resume:
            raise FileExistsError(
                f"Existing run must be resumed or placed in a new directory: {run_dir}"
            )
        previous = _load_checkpoint(last, torch.device("cpu"))
        _require_compatible_artifact(previous, compatibility, signature, last)
        initial_encoder_sha256 = previous["metadata"]["initial_encoder_sha256"]
        del previous

    model = IResNet34(embedding_dim=config.embedding_dim).to(device)
    initial_hash = hashlib.sha256()
    for tensor in model.state_dict().values():
        initial_hash.update(tensor.detach().cpu().numpy().tobytes())
    if config.channels_last:
        model.to(memory_format=torch.channels_last)
    task = MetricLearningTask(model, config, len(prepared.train_labels), margin, seed)
    optimizer, _ = _optimizer(task.model, task.head, config)
    sampler = PKBatchSampler(
        prepared.train.label.to_numpy(),
        P=config.identities_per_batch,
        K=config.images_per_identity,
        seed=seed,
    )
    if config.max_batches_per_epoch is not None:
        sampler.batches_per_epoch = min(len(sampler), config.max_batches_per_epoch)
    loader = make_loader(
        prepared.train,
        config.data_root,
        train=True,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        require_bboxes=config.require_bboxes,
        seed=seed,
    )
    dev_frame = _evaluation_frame(prepared.dev, prepared.dev_pairs)
    metadata = {
        "task": "recognition_metric_learning",
        "training_contract": "shared-trainer-v1",
        "experiment_config": {
            name: value for name, value in config.to_dict().items() if name != "resume"
        },
        "method": config.method,
        "margin": margin,
        "seed": seed,
        "class_count": len(prepared.train_labels),
        "embedding_dim": config.embedding_dim,
        "preprocessing": dict(METRIC_PREPROCESSING),
        "signature": signature,
        "initial_encoder_sha256": initial_encoder_sha256 or initial_hash.hexdigest(),
        "environment_compatibility": compatibility,
        "data": prepared.metadata,
    }

    def evaluate(current_task, current_device):
        embeddings = extract_embeddings(
            current_task.model,
            dev_frame,
            config.data_root,
            current_device,
            batch_size=config.evaluation_batch_size,
            **config.evaluation_loader_options,
        )
        metrics = verification_metrics(
            embeddings, prepared.dev_pairs, target_fmr=config.target_fmr
        )
        return {
            "dev_tar": float(metrics["tar_at_fmr"]),
            "dev_tar_at_threshold": float(metrics["tar"]),
            "dev_auc": float(metrics["auc"]),
            "dev_eer": float(metrics["eer"]),
            "dev_threshold": float(metrics["threshold"]),
            "dev_fmr": float(metrics["fmr"]),
        }

    policy = MetricTrainingPolicy(config, prepared, run_dir)
    trainer = Trainer(
        task,
        optimizer,
        config.trainer_config(seed),
        run_dir / "trainer_runs" / uuid4().hex[:12],
        metadata=metadata,
        evaluator=evaluate,
        policy=policy,
    )
    write_json(
        run_dir / "config.json",
        {
            **config.to_dict(),
            "margin": margin,
            "run_seed": seed,
            "signature": signature,
            "initial_encoder_sha256": trainer.metadata["initial_encoder_sha256"],
            "data": prepared.metadata,
            "environment": environment_details(device),
            "environment_compatibility": compatibility,
            "training_contract": "shared-trainer-v1",
        },
    )
    fit = trainer.fit(loader, resume=last)
    history = fit.history
    total_updates = sum(item["optimizer_updates"] for item in history)
    if total_updates == 0:
        raise RuntimeError(
            "No optimizer updates completed. Inspect AMP skips or empty mining "
            "before accepting this run."
        )
    _publish_checkpoint(fit.best_checkpoint, weights_dir / "best.pt")
    _publish_checkpoint(fit.last_checkpoint, weights_dir / "last.pt")
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    save_training_graph(pd.DataFrame(history), run_dir / "graphs" / "loss.png")
    best = _load_checkpoint(weights_dir / "best.pt", torch.device("cpu"))
    _require_compatible_artifact(
        best, compatibility, signature, weights_dir / "best.pt"
    )
    encoder_state = {
        name.removeprefix("model."): value
        for name, value in best["task"].items()
        if name.startswith("model.")
    }
    model.load_state_dict(encoder_state)
    best_row = best["val_metrics"]
    embeddings = extract_embeddings(
        model,
        dev_frame,
        config.data_root,
        device,
        batch_size=config.evaluation_batch_size,
        **config.evaluation_loader_options,
    )
    save_verification_graphs(
        embeddings,
        prepared.dev_pairs,
        run_dir / "graphs",
        threshold=best_row["dev_threshold"],
        target_fmr=config.target_fmr,
    )
    # The encoder export uses the same inference schema, without training heads.
    export = {
        **best,
        "task": {
            name: value
            for name, value in best["task"].items()
            if name.startswith("model.")
        },
    }
    _save_checkpoint(weights_dir / "embedding.pt", export)
    best_epoch = fit.best_epoch
    best_history = history[:best_epoch]
    result = {
        "method": config.method,
        "margin": margin,
        "seed": seed,
        "signature": signature,
        "training_contract": "shared-trainer-v1",
        "environment_compatibility": compatibility,
        "best_epoch": best_epoch,
        "dev_tar": best_row["dev_tar"],
        "dev_auc": best_row["dev_auc"],
        "dev_eer": best_row["dev_eer"],
        "dev_tar_at_threshold": best_row["dev_tar_at_threshold"],
        "dev_fmr": best_row["dev_fmr"],
        "threshold": best_row["dev_threshold"],
        "threshold_source": "calibrated_on_dev",
        "switch_epoch": task.switch_epoch,
        "curriculum_completed": task.switch_epoch is not None
        and history[-1]["epoch"] >= task.switch_epoch,
        "best_checkpoint_strategy": best_row["strategy"],
        "best_checkpoint_has_semi_hard_updates": any(
            item["strategy"] == "semi_hard" and item["optimizer_updates"] > 0
            for item in best_history
        ),
        "epochs_completed": len(history),
        "elapsed_seconds": policy.state_dict()["elapsed_seconds"],
        "total_optimizer_updates": total_updates,
        "best_checkpoint": str(weights_dir / "best.pt"),
        "embedding_checkpoint": str(weights_dir / "embedding.pt"),
        "history_path": str(run_dir / "history.csv"),
        "initial_encoder_sha256": trainer.metadata["initial_encoder_sha256"],
    }
    write_json(result_path, result)
    del trainer, fit, loader, task, model, optimizer, best, export
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_experiment(
    config: ExperimentConfig,
    prepared: PreparedData,
) -> pd.DataFrame:
    """Train one configured margin and seed, calibrating its threshold on dev."""
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    result = train_run(config, prepared, config.margin, config.seed)
    runs = pd.DataFrame([{**result, "stage": "fixed"}])
    runs.to_csv(config.output_dir / "runs.csv", index=False)
    write_json(
        config.output_dir / "selection.json",
        {
            "method": config.method,
            "selected_margin": config.margin,
            "threshold": result["threshold"],
            "threshold_source": "calibrated_on_dev",
            "criterion": "configured margin and seed; no margin sweep",
            "checkpoint_criterion": "dev TAR at target FMR; ties: AUC",
            "target_fmr": config.target_fmr,
            "seed": config.seed,
            "test_used_for_selection": False,
            "environment_compatibility": environment_compatibility(),
        },
    )
    return runs


def run_sweep(
    config: ExperimentConfig,
    prepared: PreparedData,
    margins=None,
    confirmation_seeds=None,
) -> pd.DataFrame:
    """Opt into explicit margin comparisons; notebooks use ``run_experiment``."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    margins = tuple(config.margins if margins is None else margins)
    confirmations = (
        config.confirmation_seeds
        if confirmation_seeds is None
        else tuple(confirmation_seeds)
    )
    rows = []
    for margin in margins:
        rows.append(
            {
                **train_run(config, prepared, float(margin), config.seed),
                "stage": "screening",
            }
        )
        pd.DataFrame(rows).to_csv(config.output_dir / "sweep_val.csv", index=False)
    screening = pd.DataFrame(rows)
    chosen = screening.sort_values(
        ["dev_tar", "dev_auc", "best_epoch", "margin"],
        ascending=[False, False, True, True],
    ).iloc[0]
    for seed in confirmations:
        if seed == config.seed:
            continue
        rows.append(
            {
                **train_run(config, prepared, float(chosen.margin), int(seed)),
                "stage": "confirmation",
            }
        )
        pd.DataFrame(rows).to_csv(config.output_dir / "sweep_val.csv", index=False)
    sweep = pd.DataFrame(rows)
    write_json(
        config.output_dir / "selection.json",
        {
            "method": config.method,
            "selected_margin": float(chosen.margin),
            "criterion": (
                "dev TAR at target FMR; ties: AUC, earlier epoch, smaller margin"
            ),
            "target_fmr": config.target_fmr,
            "screening_seed": config.seed,
            "test_used_for_selection": False,
            "environment_compatibility": environment_compatibility(),
        },
    )
    plot_sweep(sweep, config.output_dir / "graphs" / "sweep_val.png")
    return sweep


def evaluate_selected(
    config: ExperimentConfig, prepared: PreparedData, sweep: pd.DataFrame
) -> pd.DataFrame:
    """Evaluate trained runs using their unchanged dev-calibrated thresholds."""
    device = configure_runtime(config)
    selection = json.loads(
        (config.output_dir / "selection.json").read_text(encoding="utf-8")
    )
    compatibility = environment_compatibility()
    if selection.get("environment_compatibility") != compatibility:
        raise ValueError(
            "Selection environment changed or compatibility metadata is missing. "
            "Use a new artifacts directory."
        )
    selected = sweep[np.isclose(sweep.margin, selection["selected_margin"])]
    rows = []
    for record in selected.to_dict("records"):
        model = IResNet34(config.embedding_dim).to(device)
        state = _load_checkpoint(Path(record["best_checkpoint"]), device)
        signature = _fingerprint(
            config,
            prepared,
            float(record["margin"]),
            int(record["seed"]),
            compatibility,
        )
        _require_compatible_artifact(
            state, compatibility, signature, Path(record["best_checkpoint"])
        )
        model.load_state_dict(
            {
                name.removeprefix("model."): value
                for name, value in state["task"].items()
                if name.startswith("model.")
            }
        )
        dev = extract_embeddings(
            model,
            prepared.dev,
            config.data_root,
            device,
            batch_size=config.evaluation_batch_size,
            **config.evaluation_loader_options,
        )
        test = extract_embeddings(
            model,
            prepared.test,
            config.data_root,
            device,
            batch_size=config.evaluation_batch_size,
            **config.evaluation_loader_options,
        )
        verification = verification_metrics(
            test,
            prepared.test_pairs,
            target_fmr=config.target_fmr,
            threshold=float(record["threshold"]),
        )
        enrolled = min(
            30,
            len(prepared.dev.identity_id.unique()) - 1,
            len(prepared.test.identity_id.unique()) - 1,
        )
        dev_open = evaluate_identification(
            dev,
            prepared.dev,
            enrolled=enrolled,
            target_fpir=config.target_fpir,
            seed=42,
        )
        test_open = evaluate_identification(
            test,
            prepared.test,
            enrolled=enrolled,
            seed=42,
            target_fpir=config.target_fpir,
            threshold=dev_open["threshold"],
        )
        result = {
            "method": config.method,
            "margin": float(record["margin"]),
            "seed": int(record["seed"]),
            "environment_compatibility": compatibility,
            "verification": verification,
            "dev_open_set": dev_open,
            "test_open_set": test_open,
            "verification_ci": subject_bootstrap_ci(
                test, prepared.test_pairs, float(record["threshold"])
            ),
        }
        write_json(
            config.output_dir / "test" / f"seed_{int(record['seed'])}.json", result
        )
        save_verification_graphs(
            test,
            prepared.test_pairs,
            config.output_dir / "test" / f"seed_{int(record['seed'])}" / "graphs",
            threshold=float(record["threshold"]),
            target_fmr=config.target_fmr,
        )
        rows.append(
            {
                "method": config.method,
                "margin": record["margin"],
                "seed": record["seed"],
                "test_auc": verification["auc"],
                "test_tar": verification["tar"],
                "test_fmr": verification["fmr"],
                "test_eer": verification["eer"],
                "threshold_from_dev": record["threshold"],
                "test_rank1": test_open["rank1"],
                "test_tpir": test_open["tpir"],
                "test_fpir": test_open["fpir"],
            }
        )
        del model, state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(config.output_dir / "test_results.csv", index=False)
    frame.select_dtypes("number").drop(columns=["seed", "margin"], errors="ignore").agg(
        ["mean", "std"]
    ).to_csv(config.output_dir / "test_summary.csv")
    return frame
