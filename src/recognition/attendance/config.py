"""Configuration for attendance data preparation and confidence training."""

import argparse
from pathlib import Path

import yaml
from pydantic import Field

from src.training import TrainingConfig
from src.training.config import Settings, override_config, read_yaml


class AttendanceConfig(Settings):
    root: Path
    catalog: Path | None = None
    selection_pairs: Path | None = None
    encoder_checkpoint: Path
    dataset_dir: Path
    additional_datasets: tuple[Path, ...] = ()
    balance_training_sources: bool = False
    fixed_gallery_dev: Path | None = None
    output_dir: Path
    frame_count: int = Field(ge=2)
    gallery_size: int = Field(ge=2)
    enrollment_count: int = Field(gt=0)
    windows_per_identity: int = Field(ge=2)
    dev_fraction: float = Field(gt=0, lt=1)
    embedding_batch_size: int = Field(gt=0)
    embedding_workers: int = Field(gt=0)
    embedding_device: str
    hidden_dim: int = Field(gt=0)
    target_false_accept: float = Field(ge=0, lt=1)
    minimum_consensus: float = Field(gt=0, le=1)
    training: TrainingConfig


class VidTIMITSettings(Settings):
    detector_config: Path
    test_fraction: float = Field(gt=0, lt=1)
    frame_stride: int = Field(gt=0)
    hard_mining_min_cosine: float | None = Field(default=None, ge=-1, le=1)


class VidTIMITConfig(AttendanceConfig):
    vidtimit: VidTIMITSettings


class YouTubeFacesConfig(AttendanceConfig):
    frame_stride: int = Field(gt=0)


def load_config(description: str) -> AttendanceConfig:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    return read_config(args.config, args.set)


def read_config(path: Path, overrides=()) -> AttendanceConfig:
    raw = read_yaml(path)
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError("--set expects KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    if "vidtimit" in raw:
        cfg = VidTIMITConfig.model_validate(raw)
    elif "frame_stride" in raw:
        cfg = YouTubeFacesConfig.model_validate(raw)
    else:
        cfg = AttendanceConfig.model_validate(raw)
    base = path.resolve().parent
    resolved = {
        key: (base / value).resolve()
        for key in type(cfg).model_fields
        if isinstance(value := getattr(cfg, key), Path)
    }
    resolved["additional_datasets"] = tuple(
        (base / directory).resolve() for directory in cfg.additional_datasets
    )
    if isinstance(cfg, VidTIMITConfig):
        resolved["vidtimit"] = cfg.vidtimit.model_copy(
            update={
                "detector_config": (base / cfg.vidtimit.detector_config).resolve(),
            }
        )
    return cfg.model_copy(update=resolved)
