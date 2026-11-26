"""Task-independent single-device training with epoch-boundary recovery."""

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import MetricMonitor, TrainingConfig, resolve_device, seed_everything
from .selection import improves
from .task import TrainingTask

Evaluator = Callable[[TrainingTask, torch.device], Mapping[str, float]]


@dataclass(frozen=True)
class FitResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    history: list[dict[str, Any]]


class TrainingPolicy:
    """Optional epoch policy; the trainer retains all optimization mechanics.

    Hooks may adjust learning rates and add scalar history annotations. Policy
    state must contain only tensors and values supported by weights_only loads.
    """

    def on_epoch_start(
        self, trainer: "Trainer", epoch: int, loader: DataLoader
    ) -> None:
        pass

    def on_batch_start(
        self, trainer: "Trainer", epoch: int, batch_index: int, batches: int
    ) -> None:
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, row: dict[str, Any]) -> None:
        pass

    def selection(
        self,
        trainer: "Trainer",
        row: dict[str, Any],
        best_row: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        return trainer._selection(row, best_row)

    def should_stop(self, trainer: "Trainer", epoch: int) -> bool:
        return (
            trainer.config.patience is not None
            and trainer.bad_epochs >= trainer.config.patience
        )

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        if state:
            raise ValueError("this training policy has no restorable state")


def _cpu_state(value: Any) -> Any:
    """Clone module state, including optional non-tensor extra state."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy(value)
        result.update({key: _cpu_state(item) for key, item in value.items()})
        if hasattr(value, "_metadata"):
            result._metadata = deepcopy(value._metadata)
        return result
    if isinstance(value, tuple):
        return tuple(_cpu_state(item) for item in value)
    if isinstance(value, list):
        return [_cpu_state(item) for item in value]
    return deepcopy(value)


def new_run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / f"{name}_{stamp}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _rng_state() -> dict:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    try:
        # Use Python's file API rather than PyTorch's separate path opener.
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Trainer:
    """The task contains every trainable parameter, including loss heads.

    Evaluators return whole-validation metrics such as val_auc. Loaders must
    be nonempty and restartable. For repeatable resume use explicit generators,
    nonpersistent workers and stateless samplers, as in the recognition loader.
    """

    def __init__(
        self,
        task: TrainingTask,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        run_dir: Path,
        *,
        scheduler=None,
        metadata: dict | None = None,
        evaluator: Evaluator | None = None,
        policy: TrainingPolicy | None = None,
    ) -> None:
        self.device = resolve_device(config.device)
        self.amp_enabled = config.amp and self.device.type == "cuda"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[
            config.amp_dtype
        ]
        if self.amp_enabled and self.amp_dtype == torch.bfloat16:
            # Capability belongs to the selected GPU, which may differ from cuda:0.
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise ValueError(
                        f"{self.device} does not support native BF16; "
                        "select amp_dtype=fp16 or disable amp"
                    )
        self.task = task.to(self.device)
        owned = {id(parameter) for parameter in self.task.parameters()}
        trainable = {
            id(parameter)
            for parameter in self.task.parameters()
            if parameter.requires_grad
        }
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if not trainable.issubset(optimized) or not optimized.issubset(owned):
            raise ValueError(
                "optimizer must cover every trainable task parameter; "
                "move task to its device before building the optimizer"
            )
        self.optimizer = optimizer
        self.config = config
        self.run_dir = Path(run_dir)
        self.scheduler = scheduler
        self.metadata = metadata or {}
        self.evaluator = evaluator
        self.policy = policy
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled and self.amp_dtype == torch.float16,
            init_scale=config.amp_init_scale,
        )
        self.history: list[dict[str, Any]] = []
        self.best: float | None = None
        self.best_epoch = 0
        self.best_task: dict | None = None
        self.best_row: dict[str, Any] | None = None
        self.bad_epochs = 0
        self.start_epoch = 1
        self.current_epoch = 0
        self.stopping_best: dict[str, float] | None = None
        self.selection_metrics = (
            MetricMonitor(
                metric=config.monitor, mode=config.mode, min_delta=config.min_delta
            ),
            *config.checkpoint_tiebreakers,
        )

    def _epoch(self, loader: DataLoader, *, train: bool) -> dict[str, float]:
        self.task.train(train)
        totals: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        expected_metrics = None
        optimizer_steps = 0
        window_weight = 0
        window_batches = 0
        reference_weight = 0
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            loader,
            desc="train" if train else "val",
            leave=False,
            disable=not self.config.progress,
        )
        with torch.set_grad_enabled(train):
            for batch_index, batch in enumerate(progress):
                if train and self.policy is not None:
                    self.policy.on_batch_start(
                        self, self.current_epoch, batch_index, len(loader)
                    )
                with torch.autocast(
                    self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    result = self.task.step(_move(batch, self.device))
                if result.weight <= 0 or result.loss.ndim != 0:
                    raise ValueError("step must return scalar mean loss and weight > 0")
                if not torch.isfinite(result.loss).item():
                    raise ValueError("nonfinite loss; checkpoint has not been advanced")
                names = set(result.metrics)
                if names & {"loss", "optimizer_steps"} or (
                    expected_metrics is not None and names != expected_metrics
                ):
                    raise ValueError(
                        "step metrics must be stable and exclude loss/optimizer_steps"
                    )
                expected_metrics = names
                if train and result.optimize:
                    if window_batches == 0:
                        reference_weight = (
                            result.weight * self.config.accumulation_steps
                        )
                    self.scaler.scale(
                        result.loss * (result.weight / reference_weight)
                    ).backward()
                    window_weight += result.weight
                    window_batches += 1
                    if window_batches == self.config.accumulation_steps:
                        optimizer_steps += self._optimizer_step(
                            reference_weight / window_weight
                        )
                        window_weight, window_batches = 0, 0
                totals["loss"] += result.loss.detach().item() * result.weight
                counts["loss"] += result.weight
                for name, metric in result.metrics.items():
                    if metric.count <= 0 or not math.isfinite(metric.total):
                        raise ValueError(f"invalid metric total/count: {name}")
                    totals[name] += metric.total
                    counts[name] += metric.count
                progress.set_postfix(loss=totals["loss"] / counts["loss"])
        if train and window_batches:
            optimizer_steps += self._optimizer_step(reference_weight / window_weight)
        if not counts:
            raise ValueError("training and validation loaders must both be nonempty")
        metrics = {name: total / counts[name] for name, total in totals.items()}
        if train:
            metrics["optimizer_steps"] = optimizer_steps
        return metrics

    def _optimizer_step(self, gradient_factor: float) -> int:
        # Correct the final/unequal microbatches before AMP checks for overflow.
        if gradient_factor != 1.0:
            for group in self.optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(gradient_factor)
        previous_scale = self.scaler.get_scale()
        self.scaler.unscale_(self.optimizer)
        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.task.parameters(),
                self.config.clip_grad_norm,
                error_if_nonfinite=not self.scaler.is_enabled(),
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        return int(self.scaler.get_scale() >= previous_scale)

    def _resume(self, path: Path, loaders: dict[str, DataLoader]) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (
            saved.get("format_version") != 1
            or saved.get("checkpoint_kind") != "training"
        ):
            raise ValueError("resume requires a supported last.pt training checkpoint")
        previous = TrainingConfig.model_validate(saved["training_config"]).model_dump(
            mode="json"
        )
        current = self.config.model_dump(mode="json")
        allowed = {"epochs", "device", "progress", "num_workers"}
        changed = {key for key in previous if previous[key] != current[key]} - allowed
        if changed or saved["metadata"] != self.metadata:
            raise ValueError(f"incompatible resume config/metadata: {sorted(changed)}")
        if self.scheduler is not None and previous["epochs"] != current["epochs"]:
            raise ValueError("cannot change epochs when resuming a scheduled run")
        if self.policy is None and saved["epoch"] >= self.config.epochs:
            raise ValueError("resume checkpoint already reached the configured epochs")
        if saved["scheduler_type"] != self._scheduler_type():
            raise ValueError("resume scheduler type differs")
        if saved.get("policy_type") != self._policy_type():
            raise ValueError("resume training policy type differs")
        if self.policy is not None:
            self.policy.load_state_dict(saved.get("policy_state", {}))
        self.task.load_state_dict(saved["task"])
        group_names = [
            group.get("name", f"group_{index}")
            for index, group in enumerate(self.optimizer.param_groups)
        ]
        self.optimizer.load_state_dict(saved["optimizer"])
        for group, name in zip(self.optimizer.param_groups, group_names, strict=True):
            group["name"] = name
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved["scheduler"])
        if saved["scaler"] and self.scaler.is_enabled():
            self.scaler.load_state_dict(saved["scaler"])
        self.history = saved["history"]
        self.best = saved["best"]
        self.best_epoch = saved["best_epoch"]
        self.best_task = saved["best_task"]
        self.best_row = saved.get("best_row") or dict(self.history[self.best_epoch - 1])
        self.bad_epochs = saved["bad_epochs"]
        self.stopping_best = saved.get("stopping_best")
        if (
            self.policy is None
            and self.config.patience is not None
            and self.bad_epochs >= self.config.patience
        ):
            raise ValueError("resume checkpoint already reached early stopping")
        self.start_epoch = saved["epoch"] + 1
        for name, loader in loaders.items():
            generator_state = saved["loader_rng"].get(name)
            if generator_state is not None:
                if loader.generator is None:
                    raise ValueError(
                        f"resume needs an explicit {name} loader generator"
                    )
                loader.generator.set_state(generator_state)
        _restore_rng(saved["rng"])
        _atomic_save(self._best_checkpoint(), self.run_dir / "best.pt")

    def _scheduler_type(self) -> str | None:
        return type(self.scheduler).__qualname__ if self.scheduler is not None else None

    def _policy_type(self) -> str | None:
        if self.policy is None:
            return None
        cls = type(self.policy)
        return f"{cls.__module__}.{cls.__qualname__}"

    def _policy_checkpoint(self) -> dict:
        if self.policy is None:
            return {}
        return {
            "policy_type": self._policy_type(),
            "policy_state": self.policy.state_dict(),
        }

    def _checkpoint(self, epoch: int, loaders: dict[str, DataLoader]) -> dict:
        return {
            "format_version": 1,
            "checkpoint_kind": "training",
            "task": self.task.state_dict(),
            "best_task": self.best_task,
            "best_row": self.best_row,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scheduler_type": self._scheduler_type(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best": self.best,
            "best_epoch": self.best_epoch,
            "bad_epochs": self.bad_epochs,
            "stopping_best": self.stopping_best,
            "history": self.history,
            "training_config": self.config.model_dump(mode="json"),
            "metadata": self.metadata,
            "rng": _rng_state(),
            "loader_rng": {
                name: (
                    loader.generator.get_state()
                    if loader.generator is not None
                    else None
                )
                for name, loader in loaders.items()
            },
            **self._policy_checkpoint(),
        }

    def _best_checkpoint(self) -> dict:
        return {
            "format_version": 1,
            "checkpoint_kind": "inference",
            "task": self.best_task,
            "epoch": self.best_epoch,
            "best_epoch": self.best_epoch,
            "best": self.best,
            "training_config": self.config.model_dump(mode="json"),
            "metadata": self.metadata,
            "val_metrics": self.best_row,
            "best_row": self.best_row,
            "selection_metrics": {
                item.metric: self.best_row[item.metric]
                for item in self.selection_metrics
            },
            **self._policy_checkpoint(),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        resume: Path | None = None,
    ) -> FitResult:
        started = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if any(self.run_dir.iterdir()):
            raise ValueError(
                "run_dir must be empty; use a new directory even for resume"
            )
        if val_loader is None and self.evaluator is None:
            raise ValueError("val_loader or a whole-validation evaluator is required")
        loaders = {"train": train_loader}
        if val_loader is not None:
            loaders["val"] = val_loader
        seed_everything(self.config.seed, self.config.deterministic)
        if resume is not None:
            self._resume(Path(resume), loaders)
        initial_epochs = len(self.history)
        settings = {
            "training": self.config.model_dump(mode="json"),
            "metadata": self.metadata,
        }
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(settings, sort_keys=False), encoding="utf-8"
        )
        environment = {
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "device": str(self.device),
            "precision": (
                f"{self.config.amp_dtype}-mixed" if self.amp_enabled else "fp32"
            ),
            "gradient_scaling": self.scaler.is_enabled(),
            "gpu": (
                torch.cuda.get_device_name(self.device)
                if self.device.type == "cuda"
                else None
            ),
        }
        (self.run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            if self._should_stop(epoch):
                break
            self.current_epoch = epoch
            samplers = (train_loader.sampler, train_loader.batch_sampler)
            seen = set()
            for sampler in samplers:
                if id(sampler) not in seen and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch - 1)
                    seen.add(id(sampler))
            if self.policy is not None:
                self.policy.on_epoch_start(self, epoch, train_loader)
            scheduled = self.task.on_epoch_start(epoch)
            learning_rate = self.optimizer.param_groups[0]["lr"]
            group_learning_rates = {
                f"lr_{group.get('name', f'group_{index}')}": group["lr"]
                for index, group in enumerate(self.optimizer.param_groups)
            }
            train = self._epoch(train_loader, train=True)
            val = self._epoch(val_loader, train=False) if val_loader is not None else {}
            row = {"epoch": epoch, "lr": learning_rate}
            row.update(group_learning_rates)
            row.update({f"train_{key}": value for key, value in train.items()})
            row.update({f"val_{key}": value for key, value in val.items()})
            if row.keys() & scheduled.keys():
                raise ValueError(
                    "task schedule metrics must not overwrite epoch metrics"
                )
            row.update(scheduled)
            if self.evaluator is not None:
                self.task.eval()
                with torch.inference_mode():
                    extra = dict(self.evaluator(self.task, self.device))
                if row.keys() & extra.keys():
                    raise ValueError(
                        "evaluator metrics must not overwrite existing metrics"
                    )
                row.update(extra)
            if self.policy is not None:
                self.policy.on_epoch_end(self, epoch, row)
            required = {item.metric for item in self.selection_metrics}
            stop_monitor = self.config.early_stopping_monitor
            if stop_monitor is not None:
                required.add(stop_monitor.metric)
            if missing := required - row.keys():
                raise ValueError(f"monitor metric unavailable: {sorted(missing)}")
            if any(
                not isinstance(value, (int, float, str))
                or (not isinstance(value, str) and not math.isfinite(value))
                for value in row.values()
            ):
                raise ValueError("epoch values must be finite numbers or strings")
            if any(isinstance(row[metric], str) for metric in required):
                raise ValueError("monitor metrics must be numeric")
            score = row[self.config.monitor]
            if self.policy is None:
                improved, reset_patience = self._selection(row, self.best_row)
            else:
                improved, reset_patience = self.policy.selection(
                    self, row, self.best_row
                )
            if improved:
                self.best, self.best_epoch = score, epoch
                self.best_task = _cpu_state(self.task.state_dict())
                self.best_row = dict(row)
            if reset_patience:
                self.bad_epochs = 0
            elif self._early_stopping_active(epoch):
                self.bad_epochs += 1
            if self.scheduler is not None and train["optimizer_steps"] > 0:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(score)
                else:
                    self.scheduler.step()
            self.history.append(row)
            payload = self._checkpoint(epoch, loaders)
            if improved:
                _atomic_save(self._best_checkpoint(), self.run_dir / "best.pt")
            _atomic_save(payload, self.run_dir / "last.pt")
            self._write_history()
            if self.config.progress:
                tqdm.write(
                    f"epoch {epoch}: {self.config.monitor}={score:.6f}; "
                    f"best epoch {self.best_epoch}"
                )
        if not self.history:
            raise ValueError("training policy stopped before any epoch completed")
        if len(self.history) == initial_epochs:
            # A completed policy run can be finalized into a fresh session.
            self.current_epoch = int(self.history[-1]["epoch"])
            _atomic_save(
                self._checkpoint(self.current_epoch, loaders), self.run_dir / "last.pt"
            )
            self._write_history()
        (self.run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "epochs_in_checkpoint": self.history[-1]["epoch"],
                    "epochs_this_invocation": len(self.history) - initial_epochs,
                    "best_epoch": self.best_epoch,
                    "monitor": self.config.monitor,
                    "best_value": self.best,
                    "selection_metrics": self._best_checkpoint()["selection_metrics"],
                    "seconds_this_invocation": time.perf_counter() - started,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return FitResult(
            self.run_dir,
            self.run_dir / "best.pt",
            self.run_dir / "last.pt",
            self.best_epoch,
            list(self.history),
        )

    def _selection(
        self, row: dict[str, Any], best_row: dict[str, Any] | None
    ) -> tuple[bool, bool]:
        improved = improves(row, best_row, self.selection_metrics)
        stopping_improved = improved
        stop_monitor = self.config.early_stopping_monitor
        if stop_monitor is not None:
            stopping_improved = improves(row, self.stopping_best, (stop_monitor,))
            if stopping_improved:
                self.stopping_best = {stop_monitor.metric: row[stop_monitor.metric]}
        return improved, stopping_improved

    def _early_stopping_active(self, epoch: int) -> bool:
        if self.config.early_stopping_start_epoch is not None:
            return epoch >= self.config.early_stopping_start_epoch
        if not self.task.early_stopping_ready(epoch):
            return False
        if self.config.warmup_epochs and self.scheduler is not None:
            return self.scheduler.last_epoch >= self.config.warmup_epochs
        return epoch > self.config.warmup_epochs

    def _should_stop(self, epoch: int) -> bool:
        if self.policy is not None:
            return self.policy.should_stop(self, epoch)
        return (
            self.config.patience is not None and self.bad_epochs >= self.config.patience
        )

    def _write_history(self) -> None:
        with (self.run_dir / "history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            columns = list(dict.fromkeys(key for row in self.history for key in row))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history)
