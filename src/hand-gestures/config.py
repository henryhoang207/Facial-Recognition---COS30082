"""Settings for extracting landmarks and comparing gesture classifiers."""

from pathlib import Path

from pydantic import Field

from src.training.config import Settings, TrainingConfig, read_yaml


class GestureConfig(Settings):
    dataset_root: Path
    manifest: Path
    cache_dir: Path
    artifacts_dir: Path
    hand_model: Path
    pose_model: Path | None = None
    sample_fps: float = Field(default=20, gt=0)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    sequence_length: int = Field(default=32, ge=4)
    include_body: bool = False
    static_frames: int = Field(default=8, gt=0)
    min_hand_coverage: float = Field(default=0.8, gt=0, le=1)
    max_hand_gap_sec: float = Field(default=0.1, gt=0)
    flip_train: bool = True
    xgb_rounds: int = Field(default=160, gt=0)
    training: TrainingConfig


def load_config(path: Path | str) -> GestureConfig:
    path = Path(path).resolve()
    config = GestureConfig.model_validate(read_yaml(path))
    paths = (
        "dataset_root",
        "manifest",
        "cache_dir",
        "artifacts_dir",
        "hand_model",
        "pose_model",
    )
    return config.model_copy(
        update={
            name: (path.parent / value).resolve()
            for name in paths
            if (value := getattr(config, name)) is not None
        }
    )
