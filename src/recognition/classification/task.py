"""Model A cross-entropy training through the shared trainer."""

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
        logits = self.model(images)
        return StepResult(
            F.cross_entropy(logits, labels),
            len(labels),
            {
                "accuracy": MetricTotal(
                    (logits.argmax(dim=1) == labels).sum().item(), len(labels)
                )
            },
        )
