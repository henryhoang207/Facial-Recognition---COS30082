"""ResNet-18 embeddings and classification heads for A and C.

ArcFace: Deng et al., CVPR 2019, https://arxiv.org/abs/1801.07698.
The target logit uses a monotonic angular-margin correction.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .config import ModelConfig


class FaceEncoder(nn.Module):
    def __init__(self, config: ModelConfig, *, load_initial_weights: bool = True):
        super().__init__()
        weights = (
            ResNet18_Weights.DEFAULT
            if config.initialization == "imagenet" and load_initial_weights
            else None
        )
        self.backbone = resnet18(weights=weights)
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection = nn.Linear(features, config.embedding_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.backbone(images))

    def embed(self, images: Tensor) -> Tensor:
        return F.normalize(self(images).float(), p=2, dim=1)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, classes: int, scale: float, margin: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor | None = None) -> Tensor:
        # Keep angle calculations in fp32 even inside the trainer's autocast context.
        with torch.autocast(embeddings.device.type, enabled=False):
            cosine = F.linear(
                F.normalize(embeddings.float()), F.normalize(self.weight.float())
            ).clamp(-1 + 1e-7, 1 - 1e-7)
            if labels is None:
                return cosine * self.scale
            sine = torch.sqrt((1 - cosine.square()).clamp(min=0))
            phi = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
            phi = torch.where(
                cosine > math.cos(math.pi - self.margin),
                phi,
                cosine - math.sin(math.pi - self.margin) * self.margin,
            )
            targets = F.one_hot(labels, num_classes=self.weight.shape[0]).bool()
            return torch.where(targets, phi, cosine) * self.scale


class ClassificationModel(nn.Module):
    def __init__(
        self, config: ModelConfig, classes: int, *, load_initial_weights: bool = True
    ):
        super().__init__()
        self.encoder = FaceEncoder(config, load_initial_weights=load_initial_weights)
        self.head = (
            nn.Linear(config.embedding_dim, classes)
            if config.variant == "a"
            else ArcFaceHead(
                config.embedding_dim,
                classes,
                config.arcface_scale,
                config.arcface_margin,
            )
        )

    def forward(
        self, images: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        features = self.encoder(images)
        if isinstance(self.head, ArcFaceHead):
            return self.head(features, labels), self.head(features)
        logits = self.head(features)
        return logits, logits
