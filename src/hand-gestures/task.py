"""MLP + GRU classification on the shared PyTorch trainer."""

from torch.nn.functional import cross_entropy

from src.training import MetricTotal, StepResult, TrainingTask

from .models import GestureGRU


class GestureTask(TrainingTask):
    def __init__(self, model: GestureGRU):
        super().__init__()
        self.model = model

    def step(self, batch) -> StepResult:
        inputs, labels = batch
        logits = self.model(inputs)
        return StepResult(
            cross_entropy(logits, labels),
            len(labels),
            {
                "accuracy": MetricTotal(
                    (logits.argmax(1) == labels).sum().item(), len(labels)
                )
            },
        )
