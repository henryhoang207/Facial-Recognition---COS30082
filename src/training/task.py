"""A task owns its trainable modules and the meaning of one batch."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class MetricTotal:
    total: float
    count: int


@dataclass(frozen=True)
class StepResult:
    """Loss is a scalar mean over weight; metrics carry their own denominators.

    Set optimize=False when a valid batch has no training signal. Its metrics
    still contribute to the epoch, but it never updates optimizer state.
    """

    loss: Tensor
    weight: int
    metrics: dict[str, MetricTotal] = field(default_factory=dict)
    optimize: bool = True


class TrainingTask(nn.Module, ABC):
    def on_epoch_start(self, epoch: int) -> dict[str, float]:
        """Set deterministic task schedules from the absolute one-based epoch."""
        return {}

    def early_stopping_ready(self, epoch: int) -> bool:
        """Whether task-specific objective warmup has finished."""
        return True

    @abstractmethod
    def step(self, batch: Any) -> StepResult:
        """Compute loss and metrics, without backward or optimizer operations."""
        raise NotImplementedError
