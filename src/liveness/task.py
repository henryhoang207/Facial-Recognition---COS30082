"""Adapt both liveness losses to the shared PyTorch trainer."""

from src.training.task import StepResult, TrainingTask


class LivenessTask(TrainingTask):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def step(self, batch):
        images, labels = batch
        return StepResult(self.model.loss(self.model(images), labels), len(labels))
