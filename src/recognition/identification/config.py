"""Settings for metric-learning identification experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.training.config import TrainingConfig


@dataclass
class ExperimentConfig:
    method: Literal["triplet", "arcface"] = "triplet"
    data_root: Path = Path("data/VGGFace2")
    artifacts_root: Path = Path("artifacts")
    embedding_dim: int = 256
    identities_per_batch: int = 8
    images_per_identity: int = 4
    epochs: int = 60
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    gradient_clip: float = 5.0
    minimum_epochs: int = 15
    patience: int = 10
    minimum_semi_hard_epochs: int = 5
    scale: float = 64.0
    seed: int = 42
    confirmation_seeds: tuple[int, ...] = (123, 2026)
    num_workers: int = 2
    evaluation_batch_size: int = 64
    target_fmr: float = 1e-3
    target_fpir: float = 1e-2
    device: str = "cuda"
    amp: bool = True
    cpu_threads: int = 8
    resume: bool = True
    # Only bounded integration tests should set these limits.
    max_batches_per_epoch: int | None = None
    gate_minimum_epochs: int = 5
    gate_consecutive_checks: int = 3
    gate_ranking_rate: float = 0.75
    gate_mean_gap: float = 0.05
    gate_feature_spread: float = 0.001

    @property
    def margins(self) -> tuple[float, ...]:
        return (0.2, 0.3, 0.4) if self.method == "triplet" else (0.3, 0.4, 0.5)

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.images_per_identity

    @property
    def output_dir(self) -> Path:
        return Path(self.artifacts_root).resolve() / self.method

    def validate(self) -> None:
        if self.method not in {"triplet", "arcface"}:
            raise ValueError("method must be 'triplet' or 'arcface'.")
        if self.embedding_dim != 256:
            raise ValueError(
                "The controlled experiment requires 256-dimensional embeddings."
            )
        if self.identities_per_batch < 2 or self.images_per_identity < 2:
            raise ValueError("P and K must each be at least two.")
        if self.epochs < 1 or self.patience < 1 or self.num_workers < 0:
            raise ValueError("Invalid epoch, patience, or worker count.")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch < 1:
            raise ValueError("max_batches_per_epoch must be positive.")
        if not 0 < self.target_fmr < 1 or not 0 < self.target_fpir < 1:
            raise ValueError("Target error rates must be between zero and one.")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["data_root"] = str(Path(self.data_root).resolve())
        result["artifacts_root"] = str(Path(self.artifacts_root).resolve())
        result["confirmation_seeds"] = list(self.confirmation_seeds)
        return result

    def trainer_config(self, seed: int) -> TrainingConfig:
        """Adapt notebook settings to the shared engine without changing this API.

        The experiment policy owns the original per-batch learning-rate schedule
        and curriculum-aware stopping. No epoch scheduler is attached to Trainer.
        """
        self.validate()
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer="adamw",
            scheduler="none",
            warmup_epochs=min(self.warmup_epochs, self.epochs - 1),
            seed=seed,
            device=self.device,
            amp=self.amp,
            deterministic=True,
            num_workers=self.num_workers,
            monitor="dev_tar",
            mode="max",
            patience=self.patience,
            early_stopping_start_epoch=1,
            clip_grad_norm=self.gradient_clip,
        )
