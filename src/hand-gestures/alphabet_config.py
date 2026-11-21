"""Configuration for ASL Alphabet image experiments."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from src.training.config import Settings, TrainingConfig, read_yaml


class AlphabetConfig(Settings):
    dataset_root: Path
    cache_dir: Path
    artifacts_dir: Path
    hand_model: Path
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    landmark_device: Literal["gpu", "cpu"] = "gpu"
    extraction_workers: int = Field(default=8, gt=0)
    flip_probability: float = Field(default=0.5, ge=0, le=1)
    rotation_degrees: float = Field(default=8, ge=0, le=180)
    jitter_std: float = Field(default=0.008, ge=0)
    dropout: float = Field(default=0.15, ge=0, lt=1)
    xgb_rounds: int = Field(default=100, gt=0)
    xgb_patience: int = Field(default=80, gt=0)
    training: TrainingConfig


def load_config(path: Path | str) -> AlphabetConfig:
    path = Path(path).resolve()
    config = AlphabetConfig.model_validate(read_yaml(path))
    return config.model_copy(
        update={
            name: (path.parent / getattr(config, name)).resolve()
            for name in ("dataset_root", "cache_dir", "artifacts_dir", "hand_model")
        }
    )
