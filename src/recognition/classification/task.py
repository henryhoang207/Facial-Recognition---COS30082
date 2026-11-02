"""CE training for A and ArcFace training for C using the same engine."""

from torch.nn import functional as F

from src.training import MetricTotal, StepResult, TrainingTask

from .augmentation import FaceBatchTransform
from .config import PreprocessConfig
from .models import ClassificationModel


class RecognitionTask(TrainingTask):
    def __init__(
        self, model: ClassificationModel, preprocess: PreprocessConfig | None = None
    ):
        super().__init__()
        self.model = model
        self.transform = FaceBatchTransform(preprocess) if preprocess else None

    def step(self, batch) -> StepResult:
        images, labels = batch
        if self.transform is not None:
            images = self.transform(images)
        logits, predictions = self.model(images, labels)
        return StepResult(
            F.cross_entropy(logits, labels),
            len(labels),
            {
                "accuracy": MetricTotal(
                    (predictions.argmax(dim=1) == labels).sum().item(), len(labels)
                )
            },
        )
