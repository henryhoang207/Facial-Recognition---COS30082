"""Mask classifiers and their input transforms."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models, transforms

CLASSES = ["mask", "no_mask"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Ứng viên so sánh trong report: MobileNetV3 (torchvision, 2019) và MobileNetV4 (timm, 2024)
ARCHS = {
    "mobilenetv3s": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenetv3l": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
}
# Trọng số ImageNet-1k bản 224 px, cùng mean/std ImageNet nên dùng chung transform bên dưới
TIMM_ARCHS = {
    "mobilenetv4m": "mobilenetv4_conv_medium.e500_r224_in1k",
}


def build_model(arch: str, pretrained: bool = True) -> nn.Module:
    if arch in TIMM_ARCHS:
        import timm  # chỉ cần khi dùng MobileNetV4

        return timm.create_model(TIMM_ARCHS[arch], pretrained=pretrained, num_classes=len(CLASSES))
    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)
    head = model.classifier[-1]
    model.classifier[-1] = nn.Linear(head.in_features, len(CLASSES))
    return model


def train_transform(size: int) -> transforms.Compose:
    # crop nhẹ (>= 80% diện tích) để không cắt mất phần miệng/cằm nơi có khẩu trang
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
            transforms.RandomGrayscale(0.1),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    if config.get("task") != "mask":
        raise ValueError(f"{path} is not a mask checkpoint")
    model = build_model(config["arch"], pretrained=False)
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), config
