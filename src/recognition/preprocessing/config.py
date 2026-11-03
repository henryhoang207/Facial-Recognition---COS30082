"""Validated configuration for offline face bbox generation."""

from pathlib import Path

from pydantic import Field

from src.training.config import Settings


class DetectorConfig(Settings):
    model: Path
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_size: int = Field(ge=32)
    score_threshold: float = Field(gt=0, le=1)
    nms_threshold: float = Field(gt=0, le=1)
    top_k: int = Field(gt=0)
    min_face_pixels: int = Field(gt=0)
    min_face_area_fraction: float = Field(gt=0, lt=1)
    max_center_distance: float = Field(gt=0, le=1)


class FacePreparationConfig(Settings):
    root: Path
    classification: Path
    verification: Path
    output_dir: Path
    detector: DetectorConfig
    overrides: Path | None = None
    seed: int = Field(ge=0, le=2**32 - 1)
    workers: int = Field(gt=0)
    chunk_size: int = Field(gt=0)
    positive_pairs: int = Field(gt=0)
    negative_pairs: int = Field(gt=0)
    progress: bool = True
