"""Reusable training mechanics. Individual packages own datasets and objectives."""

from .config import (
    ParameterGroupConfig,
    TrainingConfig,
    make_optimizer,
    make_scheduler,
    seed_everything,
    seed_worker,
)
from .task import MetricTotal, StepResult, TrainingTask
from .trainer import FitResult, Trainer, TrainingPolicy, new_run_dir

__all__ = [
    "FitResult",
    "MetricTotal",
    "ParameterGroupConfig",
    "StepResult",
    "Trainer",
    "TrainingConfig",
    "TrainingPolicy",
    "TrainingTask",
    "make_optimizer",
    "make_scheduler",
    "new_run_dir",
    "seed_everything",
    "seed_worker",
]
