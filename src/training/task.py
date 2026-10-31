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
    """Loss is a scalar mean over weight; metrics carry their own denominators."""

    loss: Tensor
    weight: int
    metrics: dict[str, MetricTotal] = field(default_factory=dict)


class TrainingTask(nn.Module, ABC):
    @abstractmethod
    def step(self, batch: Any) -> StepResult:
        """Compute loss and metrics, without backward or optimizer operations."""
        raise NotImplementedError
