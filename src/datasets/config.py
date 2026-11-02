"""Configuration for reproducible dataset exports."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from src.training.config import Settings


class SourceConfig(Settings):
    root: Path
    database: str


class RecognitionSource(SourceConfig):
    val_fraction: float = Field(gt=0, lt=1)
    test_fraction: float = Field(gt=0, lt=1)
    verification_val_fraction: float = Field(gt=0, lt=1)
    positive_pairs: int = Field(gt=0)
    negative_pairs: int = Field(gt=0)

    @model_validator(mode="after")
    def split_leaves_training_data(self) -> Self:
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError(
                "classification val/test fractions must leave training data"
            )
        return self


class LivenessSource(SourceConfig):
    val_fraction: float = Field(gt=0, lt=1)
    attacks: list[Literal["print", "replay", "paper_mask", "3d_mask"]]


class EmotionSource(SourceConfig):
    splits_dir: str
    val_fraction: float = Field(gt=0, lt=1)
    expected_frames: int = Field(gt=0)


class TeamSource(Settings):
    root: Path
    role: Literal["gallery"]
    detector_max_side: int = Field(gt=0)
    detector_scale_factor: float = Field(gt=1)
    detector_min_neighbors: int = Field(ge=0)
    detector_min_size: int = Field(gt=0)
    min_face_area_fraction: float = Field(gt=0, lt=1)


class PreparationConfig(Settings):
    output_dir: Path
    seed: int = Field(ge=0, le=2**32 - 1)
    workers: int = Field(gt=0)
    recognition: RecognitionSource
    liveness: LivenessSource
    emotion: EmotionSource
    team: TeamSource
