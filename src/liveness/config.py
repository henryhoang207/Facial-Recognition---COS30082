"""YAML configuration with backwards-compatible liveness CLI flags."""

import argparse
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from src.training.config import Settings, TrainingConfig, override_config, read_yaml

DEFAULTS = {
    "mobilenetv3": {
        "lr": 3e-4,
        "freeze_blocks": 4,
        "batch_size": 64,
        "accumulation_steps": 1,
    },
    "deeppixbis": {
        "lr": 1e-4,
        "freeze_blocks": 0,
        "batch_size": 32,
        "accumulation_steps": 2,
    },
}


class LivenessConfig(Settings):
    model: Literal["mobilenetv3", "deeppixbis"]
    pretrained: bool = True
    freeze_blocks: int = Field(default=0, ge=0, le=8)
    img_size: int = Field(default=224, ge=32, multiple_of=16)
    crop_scale: float = Field(default=2.0, gt=0)
    manifest: tuple[Path, ...] = Field(min_length=1)
    data_root: Path
    output_dir: Path
    fig_dir: Path = Path("reports/figures")
    tag: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]+$")
    max_train: int | None = Field(default=None, ge=2)
    max_val: int | None = Field(default=None, ge=2)
    training: TrainingConfig


def defaults(model: str) -> dict:
    values = DEFAULTS[model]
    return {
        "model": model,
        "pretrained": True,
        "freeze_blocks": values["freeze_blocks"],
        "img_size": 224,
        "crop_scale": 2.0,
        "manifest": [],
        "data_root": ".",
        "output_dir": "checkpoints",
        "fig_dir": "reports/figures",
        "tag": None,
        "max_train": None,
        "max_val": None,
        "training": {
            "epochs": 15,
            "batch_size": values["batch_size"],
            "lr": values["lr"],
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "seed": 42,
            "device": "auto",
            "amp": True,
            "deterministic": False,
            "num_workers": 2,
            "monitor": "val_acer",
            "mode": "min",
            "patience": 4,
            "accumulation_steps": values["accumulation_steps"],
        },
    }


def load_config(argv=None) -> tuple[LivenessConfig, Path | None]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--model", choices=tuple(DEFAULTS))
    parser.add_argument("--manifest", nargs="+", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--set", action="append", default=[])
    flags = {
        "data-root": ("data_root", str),
        "ckpt-dir": ("output_dir", str),
        "fig-dir": ("fig_dir", str),
        "tag": ("tag", str),
        "img-size": ("img_size", int),
        "crop-scale": ("crop_scale", float),
        "freeze-blocks": ("freeze_blocks", int),
        "max-train": ("max_train", int),
        "max-val": ("max_val", int),
        **{
            name: ("training." + name.replace("-", "_"), kind)
            for name, kind in [
                ("epochs", int),
                ("patience", int),
                ("batch-size", int),
                ("lr", float),
                ("weight-decay", float),
                ("num-workers", int),
                ("seed", int),
                ("device", str),
                ("accumulation-steps", int),
            ]
        },
    }
    for flag, (_, kind) in flags.items():
        parser.add_argument("--" + flag, type=kind, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.config:
        raw, base = read_yaml(args.config), args.config.resolve().parent
        if args.model and args.model != raw["model"]:
            parser.error(
                "choose the matching model YAML instead of overriding its architecture"
            )
    else:
        if not args.model or not args.manifest:
            parser.error("provide --config, or --model and --manifest")
        raw, base = defaults(args.model), Path.cwd()
    if args.manifest:
        raw["manifest"] = [str(path.resolve()) for path in args.manifest]
    for flag, (key, _) in flags.items():
        attr = flag.replace("-", "_")
        if hasattr(args, attr):
            value = getattr(args, attr)
            if key in {"data_root", "output_dir", "fig_dir"}:
                value = str(Path(value).resolve())
            override_config(raw, key, value)
    if args.no_pretrained:
        raw["pretrained"] = False
    for override in args.set:
        key, sep, value = override.partition("=")
        if not sep:
            parser.error("--set requires KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    config = LivenessConfig.model_validate(raw)
    config = config.model_copy(
        update={
            "manifest": tuple((base / path).resolve() for path in config.manifest),
            **{
                key: (base / getattr(config, key)).resolve()
                for key in ("data_root", "output_dir", "fig_dir")
            },
        }
    )
    if config.training.monitor != "val_acer" or config.training.mode != "min":
        raise ValueError("liveness selects by minimum val_acer")
    return config, args.resume.resolve() if args.resume else None
