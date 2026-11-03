"""Configuration boundary for Models A and C."""

import argparse
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, model_validator

from src.recognition.verification import VerificationConfig
from src.training.config import Settings, TrainingConfig, override_config, read_yaml


class ModelConfig(Settings):
    variant: Literal["a", "c"]
    backbone: Literal["resnet18"]
    embedding_dim: int = Field(gt=0)
    initialization: Literal["random", "imagenet"]
    arcface_scale: float = Field(gt=0)
    arcface_margin: float = Field(gt=0, lt=1.5707963267948966)


class PreprocessConfig(Settings):
    size: int = Field(ge=32)
    crop_scale: float = Field(gt=0)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    horizontal_flip: float = Field(ge=0, le=1)
    brightness: float = Field(ge=0, le=1)
    contrast: float = Field(default=0, ge=0, le=1)
    saturation: float = Field(default=0, ge=0, le=1)
    rotation_degrees: float = Field(default=0, ge=0, le=45)
    translate: float = Field(default=0, ge=0, le=0.25)

    @model_validator(mode="after")
    def positive_std(self) -> Self:
        if any(value <= 0 for value in self.std):
            raise ValueError("normalization std must be positive")
        return self


class DataConfig(Settings):
    root: Path
    train_dir: str
    val_dir: str
    test_dir: str
    manifest: Path | None
    cache: Literal["auto", "none"] = "auto"
    cache_max_fraction: float = Field(default=0.8, gt=0, lt=1)
    cache_reserve_gib: float = Field(default=2.0, ge=0)


class RecognitionConfig(Settings):
    model: ModelConfig
    preprocess: PreprocessConfig
    data: DataConfig
    training: TrainingConfig
    output_dir: Path
    evaluator: str | None = None
    verification: VerificationConfig | None = None


def load_config(argv: list[str] | None = None) -> tuple[RecognitionConfig, Path | None]:
    parser = argparse.ArgumentParser(description="Train recognition A/C from YAML")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--group-lr",
        action="append",
        default=[],
        metavar="NAME=LR",
        help="override a named optimizer group, e.g. backbone=0.0001",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override any existing YAML key; parsed and validated using its schema",
    )
    aliases = {
        "lr": ("training.lr", float),
        "epochs": ("training.epochs", int),
        "batch-size": ("training.batch_size", int),
        "weight-decay": ("training.weight_decay", float),
        "seed": ("training.seed", int),
        "device": ("training.device", str),
        "num-workers": ("training.num_workers", int),
        "patience": ("training.patience", int),
        "embedding-dim": ("model.embedding_dim", int),
        "data-root": ("data.root", str),
        "output-dir": ("output_dir", str),
        "monitor": ("training.monitor", str),
        "mode": ("training.mode", str),
        "evaluator": ("evaluator", str),
        "optimizer": ("training.optimizer", str),
        "scheduler": ("training.scheduler", str),
        "warmup-epochs": ("training.warmup_epochs", int),
        "accumulation-steps": ("training.accumulation_steps", int),
        "eta-min": ("training.eta_min", float),
        "cache": ("data.cache", str),
        "amp-dtype": ("training.amp_dtype", str),
    }
    for flag, (_, value_type) in aliases.items():
        parser.add_argument(f"--{flag}", type=value_type, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    raw = read_yaml(args.config)
    for override in args.set:
        key, separator, text = override.partition("=")
        if not separator:
            parser.error("--set expects SECTION.KEY=VALUE")
        override_config(raw, key, yaml.safe_load(text))
    for flag, (key, _) in aliases.items():
        if hasattr(args, flag.replace("-", "_")):
            override_config(raw, key, getattr(args, flag.replace("-", "_")))
    for override in args.group_lr:
        name, separator, value = override.partition("=")
        if not separator:
            parser.error("--group-lr expects NAME=LR")
        groups = raw["training"].get("parameter_groups", [])
        matching = [group for group in groups if group.get("name") == name]
        if len(matching) != 1:
            parser.error(f"optimizer group must exist exactly once: {name}")
        matching[0]["lr"] = yaml.safe_load(value)
        matching[0]["lr_scale"] = 1.0
    config = RecognitionConfig.model_validate(raw)
    # All paths are relative to the config file, independent of the working directory.
    base = args.config.resolve().parent
    root = (base / config.data.root).resolve()
    manifest = config.data.manifest
    data = config.data.model_copy(
        update={
            "root": root,
            "manifest": (base / manifest).resolve() if manifest is not None else None,
        }
    )
    config = config.model_copy(
        update={
            "data": data,
            "output_dir": (base / config.output_dir).resolve(),
            "verification": (
                config.verification.model_copy(
                    update={
                        "root": (base / config.verification.root).resolve(),
                        "pairs": (base / config.verification.pairs).resolve(),
                        "bbox_manifest": (
                            (base / config.verification.bbox_manifest).resolve()
                            if config.verification.bbox_manifest is not None
                            else None
                        ),
                    }
                )
                if config.verification is not None
                else None
            ),
        }
    )
    return config, args.resume.resolve() if args.resume else None
