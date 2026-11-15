"""ResNet-18 encoder and linear classification head for Model A."""

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


class ClassificationModel(nn.Module):
    def __init__(
        self, config: ModelConfig, classes: int, *, load_initial_weights: bool = True
    ):
        super().__init__()
        self.encoder = FaceEncoder(config, load_initial_weights=load_initial_weights)
        self.head = nn.Linear(config.embedding_dim, classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.encoder(images))
