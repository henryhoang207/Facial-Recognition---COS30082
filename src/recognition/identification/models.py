"""Randomly initialized iResNet encoders for the recognition experiments.

Architecture reference: https://github.com/deepinsight/insightface/blob/master/
recognition/arcface_torch/backbones/iresnet.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class IBasicBlock(nn.Module):
    """Pre-activation residual block used by the InsightFace iResNet."""

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels, eps=1e-5)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels, eps=1e-5),
            )

    def forward(self, images: Tensor) -> Tensor:
        identity = images if self.downsample is None else self.downsample(images)
        output = self.conv1(self.bn1(images))
        output = self.conv2(self.prelu(self.bn2(output)))
        return self.bn3(output) + identity


class IResNet34(nn.Module):
    """iResNet-34 for RGB 112x112 inputs and FP32 unit-length embeddings.

    The final BN scale is fixed at one, while its bias remains trainable.
    Mixed precision is controlled by the caller; the FC, output BN, and
    normalization always execute in FP32. No pretrained weights are loaded.
    """

    def __init__(self, embedding_dim: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding_dim = int(embedding_dim)
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4)
        self.layer3 = self._make_layer(256, 6)
        self.layer4 = self._make_layer(512, 3)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * 7 * 7, self.embedding_dim)
        self.features = nn.BatchNorm1d(self.embedding_dim, eps=1e-5)
        nn.init.ones_(self.features.weight)
        self.features.weight.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels: int, blocks: int) -> nn.Sequential:
        layers = [IBasicBlock(self.in_channels, channels, stride=2)]
        self.in_channels = channels
        layers.extend(IBasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward_features(self, images: Tensor) -> Tensor:
        """Return raw FP32 features after the final embedding BN."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 112, 112):
            raise ValueError("Expected inputs with shape (batch, 3, 112, 112)")
        output = self.prelu(self.bn1(self.conv1(images)))
        output = self.layer4(self.layer3(self.layer2(self.layer1(output))))
        output = self.dropout(torch.flatten(self.bn2(output), 1))
        with torch.autocast(device_type=output.device.type, enabled=False):
            return self.features(self.fc(output.float()))

    def forward(self, images: Tensor) -> Tensor:
        features = self.forward_features(images)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.normalize(features.float(), p=2, dim=1, eps=1e-12)


def iresnet34(embedding_dim: int = 256, dropout: float = 0.0) -> IResNet34:
    """Construct the common randomly initialized recognition encoder."""
    return IResNet34(embedding_dim=embedding_dim, dropout=dropout)
