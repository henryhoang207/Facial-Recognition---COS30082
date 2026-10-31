"""Use the shared pair evaluator during checkpoint selection."""

import numpy as np
import torch

from src.recognition.verification import PairEvaluator

from .config import RecognitionConfig
from .data import FaceTransform


class EncoderEmbedder:
    def __init__(self, encoder, transform: FaceTransform, device: torch.device):
        self.encoder, self.transform, self.device = encoder, transform, device

    @torch.inference_mode()
    def embed(self, face: np.ndarray) -> np.ndarray:
        images = self.transform(face).unsqueeze(0).to(self.device)
        return self.encoder.embed(images)[0].cpu().numpy()


def make_evaluator(config: RecognitionConfig):
    if config.verification is None:
        raise ValueError("configure verification paths before monitoring val_auc")
    evaluator = PairEvaluator(
        config.verification,
        size=config.preprocess.size,
        crop_scale=config.preprocess.crop_scale,
        progress=config.training.progress,
    )
    transform = FaceTransform(config.preprocess)

    def evaluate(task, device):
        embedder = EncoderEmbedder(task.model.encoder, transform, device)
        metrics, _, _, _ = evaluator.evaluate(embedder)
        selected = metrics[config.verification.selection_metric]
        return {
            "val_auc": selected["auc"],
            "val_tar": selected["tar"],
            "val_fpr": selected["fpr"],
            "val_score_threshold": selected["score_threshold"],
        }

    return evaluate
