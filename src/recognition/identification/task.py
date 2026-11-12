"""Metric-learning objectives adapted to the shared training engine."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.training import MetricTotal, StepResult, TrainingTask

from .augmentation import augment_training_batch
from .config import ExperimentConfig
from .losses import ArcFaceHead, TripletMetricLoss


class MetricLearningTask(TrainingTask):
    """Own the encoder, optional trainable head, and resumable mining state."""

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        class_count: int,
        margin: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.model = model
        device = next(model.parameters()).device
        self.head = (
            ArcFaceHead(
                class_count,
                config.embedding_dim,
                margin=margin,
                scale=config.scale,
                seed=seed + 100_000,
            ).to(device)
            if config.method == "arcface"
            else None
        )
        self.triplet = TripletMetricLoss(margin)
        self.mining_generator = torch.Generator(device=device).manual_seed(
            seed + 200_000
        )
        self.strategy = "random" if self.head is None else "arcface"
        self.switch_epoch: int | None = None
        self.gate_passes = 0
        self.reset_epoch_counts()

    def reset_epoch_counts(self) -> None:
        self.batches_forward = 0
        self.images_forward = 0
        self.empty_batches = 0
        self.no_update_batches = 0

    def step(self, batch) -> StepResult:
        images, labels = batch["image"], batch["label"]
        if images.dtype == torch.uint8:
            images = augment_training_batch(images, batch["augmentation_params"])
        embeddings = self.model(images)
        if self.head is None:
            loss, diagnostics = self.triplet(
                embeddings,
                labels,
                strategy=self.strategy,
                generator=self.mining_generator,
            )
            optimize = (
                diagnostics["selected_triplets"] > 0
                and diagnostics["active_triplets"] > 0
            )
            self.empty_batches += int(diagnostics["selected_triplets"] == 0)
        else:
            logits = self.head(embeddings, labels)
            loss = nn.functional.cross_entropy(logits, labels)
            diagnostics = {
                "accuracy": float((logits.argmax(1) == labels).float().mean().detach())
            }
            optimize = True
        self.batches_forward += 1
        self.images_forward += images.shape[0]
        self.no_update_batches += int(not optimize)
        return StepResult(
            loss,
            images.shape[0],
            {
                name: MetricTotal(float(value), 1)
                for name, value in diagnostics.items()
                if math.isfinite(float(value))
            },
            optimize=optimize,
        )

    def get_extra_state(self) -> dict:
        """Store protocol state inside the common checkpoint's task field."""
        return {
            "version": 1,
            "strategy": self.strategy,
            "switch_epoch": self.switch_epoch,
            "gate_passes": self.gate_passes,
            "mining_rng": self.mining_generator.get_state().cpu(),
        }

    def set_extra_state(self, state: dict) -> None:
        if state.get("version") != 1:
            raise ValueError("Unsupported metric-learning task state.")
        self.strategy = state["strategy"]
        self.switch_epoch = state["switch_epoch"]
        self.gate_passes = state["gate_passes"]
        self.mining_generator.set_state(state["mining_rng"].cpu())
