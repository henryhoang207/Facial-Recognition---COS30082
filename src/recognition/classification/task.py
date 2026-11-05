"""CE training for A and ArcFace training for C using the same engine."""

import math

import torch
from torch.nn import functional as F

from src.training import MetricTotal, StepResult, TrainingTask

from .augmentation import FaceBatchTransform
from .config import PreprocessConfig
from .models import ArcFaceHead, ClassificationModel


class RecognitionTask(TrainingTask):
    def __init__(
        self, model: ClassificationModel, preprocess: PreprocessConfig | None = None
    ):
        super().__init__()
        self.model = model
        self.transform = FaceBatchTransform(preprocess) if preprocess else None

    def on_epoch_start(self, epoch: int) -> dict[str, float]:
        if not isinstance(self.model.head, ArcFaceHead):
            return {}
        config = self.model.config
        self.model.head.margin = config.arcface_margin_warmup.value(
            epoch, config.arcface_margin
        )
        return {"arcface_margin": self.model.head.margin}

    def early_stopping_ready(self, epoch: int) -> bool:
        config = self.model.config
        warmup = config.arcface_margin_warmup
        return not warmup.enabled or epoch > warmup.hold_epochs + warmup.ramp_epochs

    def step(self, batch) -> StepResult:
        images, labels = batch
        if self.transform is not None:
            images = self.transform(images)
        logits, predictions = self.model(images, labels)
        classification_loss = F.cross_entropy(logits, labels)
        loss = classification_loss
        metrics = {
            "accuracy": MetricTotal(
                (predictions.argmax(dim=1) == labels).sum().item(), len(labels)
            )
        }
        if isinstance(self.model.head, ArcFaceHead):
            with torch.autocast(logits.device.type, enabled=False):
                prototype_mean_norm = (
                    F.normalize(self.model.head.weight.float(), dim=1).mean(0).norm()
                )
                loss = loss + (
                    self.model.config.prototype_regularization_weight
                    * prototype_mean_norm
                )
                no_margin_loss = F.cross_entropy(predictions, labels)
                target_cosine = predictions.detach().gather(1, labels[:, None]) / (
                    self.model.head.scale
                )
                fallback = target_cosine <= math.cos(math.pi - self.model.head.margin)
            metrics.update(
                classification_loss=MetricTotal(
                    classification_loss.detach().item() * len(labels), len(labels)
                ),
                no_margin_loss=MetricTotal(
                    no_margin_loss.detach().item() * len(labels), len(labels)
                ),
                prototype_mean_norm=MetricTotal(
                    prototype_mean_norm.detach().item() * len(labels), len(labels)
                ),
                target_cosine=MetricTotal(target_cosine.sum().item(), len(labels)),
                fallback_fraction=MetricTotal(fallback.sum().item(), len(labels)),
            )
        return StepResult(
            loss,
            len(labels),
            metrics,
        )
