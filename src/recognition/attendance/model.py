"""Small MLP and the interface used to score a completed frame window."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.training import StepResult, TrainingTask

from .features import FEATURES, TOP2_FEATURES, VARIABLE_GALLERY_FEATURES, attempt_features


class AttendanceMLP(TrainingTask):
    def __init__(self, hidden_dim: int, input_dim: int = len(FEATURES)):
        super().__init__()
        self.register_buffer("mean", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network((features - self.mean) / self.scale).squeeze(-1)

    def step(self, batch) -> StepResult:
        features, labels = batch
        loss = F.binary_cross_entropy_with_logits(self(features), labels)
        return StepResult(loss, len(labels))


@dataclass(frozen=True)
class AttendanceDecision:
    candidate: str
    score: float
    accepted: bool


class AttendancePredictor:
    def __init__(self, checkpoint: Path, *, encoder_sha256: str):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = saved["metadata"]
        self.feature_names = tuple(metadata["feature_names"])
        if (
            metadata.get("task") != "attendance_confidence"
            or self.feature_names not in (FEATURES, TOP2_FEATURES, VARIABLE_GALLERY_FEATURES)
            or metadata["encoder_sha256"] != encoder_sha256
        ):
            raise ValueError(
                "attendance checkpoint is incompatible with this Model A encoder"
            )
        self.frame_count = metadata["frame_count"]
        self.gallery_size = metadata["gallery_size"]
        if self.feature_names == VARIABLE_GALLERY_FEATURES:
            if self.gallery_size is not None:
                raise ValueError("variable-gallery checkpoint must not require a fixed gallery size")
        elif type(self.gallery_size) is not int or self.gallery_size < 2:
            raise ValueError("legacy checkpoint must declare its fixed gallery size")
        self.threshold = saved["threshold"]
        if not isfinite(self.threshold):
            raise ValueError("attendance threshold must be finite")
        self.model = AttendanceMLP(
            metadata["hidden_dim"], len(self.feature_names)
        ).eval()
        self.model.load_state_dict(saved["task"])

    @torch.inference_mode()
    def predict(
        self, frames: np.ndarray, templates: Mapping[str, np.ndarray]
    ) -> AttendanceDecision:
        if len(frames) != self.frame_count:
            raise ValueError("frame count must match the trained protocol")
        if not templates or (self.gallery_size is not None and len(templates) != self.gallery_size):
            raise ValueError("gallery does not match the checkpoint protocol")
        identities = sorted(templates)
        winner, features = attempt_features(
            frames,
            np.stack([templates[name] for name in identities]),
            include_top2_gap=self.feature_names == TOP2_FEATURES,
            include_gallery_size=self.gallery_size is None,
        )
        score = self.score(features)
        return AttendanceDecision(identities[winner], score, score >= self.threshold)

    @torch.inference_mode()
    def score(self, features: np.ndarray) -> float:
        return self.model(torch.from_numpy(features)).sigmoid().item()
