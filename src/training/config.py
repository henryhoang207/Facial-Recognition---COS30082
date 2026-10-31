"""Validated training settings and YAML loading shared by task entry points."""

import os
import random
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ParameterGroupConfig(Settings):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    modules: tuple[str, ...] = Field(min_length=1)
    lr: float | None = Field(default=None, gt=0)
    lr_scale: float = Field(default=1.0, gt=0)
    weight_decay: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_group(self) -> Self:
        if self.name == "default":
            raise ValueError("default is reserved for ungrouped task parameters")
        if any(not name for name in self.modules):
            raise ValueError("parameter group modules must be nonempty module paths")
        if self.lr is not None and self.lr_scale != 1:
            raise ValueError("choose an explicit group lr or lr_scale, not both")
        return self


class TrainingConfig(Settings):
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    optimizer: Literal["adamw", "sgd"]
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = Field(default=1e-8, gt=0)
    adam_amsgrad: bool = False
    sgd_momentum: float = Field(default=0.0, ge=0)
    sgd_dampening: float = Field(default=0.0, ge=0)
    sgd_nesterov: bool = False
    parameter_groups: tuple[ParameterGroupConfig, ...] = ()
    scheduler: Literal["none", "cosine"]
    eta_min: float = Field(default=0.0, ge=0)
    cosine_t_max: int | None = Field(default=None, gt=0)
    warmup_epochs: int = Field(default=0, ge=0)
    warmup_start_factor: float = Field(default=0.1, gt=0, le=1)
    accumulation_steps: int = Field(default=1, gt=0)
    seed: int = Field(ge=0, le=2**32 - 1)
    device: str
    amp: bool
    amp_init_scale: float = Field(default=65536.0, gt=0)
    deterministic: bool
    num_workers: int = Field(ge=0)
    monitor: str
    mode: Literal["min", "max"]
    patience: int | None = Field(default=None, gt=0)
    min_delta: float = Field(default=0, ge=0)
    early_stopping_start_epoch: int | None = Field(default=None, gt=0)
    clip_grad_norm: float | None = Field(default=None, gt=0)
    progress: bool = True

    @model_validator(mode="after")
    def validate_device(self) -> Self:
        if self.device != "auto":
            try:
                device = torch.device(self.device)
            except (RuntimeError, ValueError) as exc:
                raise ValueError("device must be auto, cpu, cuda, or cuda:N") from exc
            if device.type not in {"cpu", "cuda"}:
                raise ValueError("only CPU and CUDA devices are supported")
        if any(not 0 <= beta < 1 for beta in self.adam_betas):
            raise ValueError("Adam betas must be in [0,1)")
        if (
            self.optimizer == "sgd"
            and self.sgd_nesterov
            and (self.sgd_momentum <= 0 or self.sgd_dampening != 0)
        ):
            raise ValueError("SGD Nesterov requires momentum > 0 and dampening = 0")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be smaller than epochs")
        if (
            self.scheduler == "cosine"
            and self.cosine_t_max is not None
            and self.cosine_t_max < self.epochs - self.warmup_epochs
        ):
            raise ValueError("cosine_t_max must cover all epochs after warmup")
        if (
            self.early_stopping_start_epoch is not None
            and self.early_stopping_start_epoch > self.epochs
        ):
            raise ValueError("early_stopping_start_epoch cannot exceed epochs")
        names = [group.name for group in self.parameter_groups]
        if len(set(names)) != len(names):
            raise ValueError("parameter group names must be unique")
        learning_rates = [
            group.lr if group.lr is not None else self.lr * group.lr_scale
            for group in self.parameter_groups
        ] or [self.lr]
        if self.scheduler == "cosine" and self.eta_min > min(learning_rates):
            raise ValueError(
                "eta_min cannot exceed any parameter group's learning rate"
            )
        return self


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently discarding a YAML setting."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"YAML keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.load(stream, Loader=UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError("config must contain a YAML mapping")
    return value


def override_config(raw: dict, key: str, value: Any) -> None:
    """Override an existing YAML key; the caller validates the complete schema."""
    parts = key.split(".")
    target = raw
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise ValueError(f"unknown config key: {key}")
        target = target[part]
    if parts[-1] not in target:
        raise ValueError(f"unknown config key: {key}")
    target[parts[-1]] = value


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Call before constructing a model or loaders."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {device}")
    return device


def seed_worker(worker_id: int) -> None:
    """Top-level function so Windows workers can import it."""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_optimizer(
    task: torch.nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    params = [parameter for parameter in task.parameters() if parameter.requires_grad]
    groups = []
    assigned: set[int] = set()
    for group in config.parameter_groups:
        selected = set()
        for module_path in group.modules:
            try:
                module = task.get_submodule(module_path)
            except AttributeError as exc:
                raise ValueError(
                    f"unknown parameter group module: {module_path}"
                ) from exc
            selected.update(
                id(parameter)
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        if not selected:
            raise ValueError(
                f"parameter group has no trainable parameters: {group.name}"
            )
        if assigned & selected:
            raise ValueError(
                f"parameters overlap between optimizer groups: {group.name}"
            )
        assigned.update(selected)
        groups.append(
            {
                "name": group.name,
                "params": [
                    parameter for parameter in params if id(parameter) in selected
                ],
                "lr": group.lr if group.lr is not None else config.lr * group.lr_scale,
                "weight_decay": (
                    group.weight_decay
                    if group.weight_decay is not None
                    else config.weight_decay
                ),
            }
        )
    remaining = [parameter for parameter in params if id(parameter) not in assigned]
    if remaining:
        groups.append({"name": "default", "params": remaining})
    options = {"lr": config.lr, "weight_decay": config.weight_decay}
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            groups,
            betas=config.adam_betas,
            eps=config.adam_eps,
            amsgrad=config.adam_amsgrad,
            **options,
        )
    return torch.optim.SGD(
        groups,
        momentum=config.sgd_momentum,
        dampening=config.sgd_dampening,
        nesterov=config.sgd_nesterov,
        **options,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler == "cosine" and config.eta_min > min(
        group["lr"] for group in optimizer.param_groups
    ):
        raise ValueError("eta_min exceeds an actual optimizer group's learning rate")
    warmup = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        if config.warmup_epochs
        else None
    )
    if config.scheduler == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.cosine_t_max or config.epochs - config.warmup_epochs,
            eta_min=config.eta_min,
        )
        if warmup is not None:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[config.warmup_epochs],
            )
        return cosine
    return warmup
