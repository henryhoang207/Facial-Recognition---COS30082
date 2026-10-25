"""The two liveness candidates. Both output a score = probability the face is live.

Each model returns a dict from forward() and owns its loss and its score, so train.py
and evaluate.py treat them the same. Label convention: 1 = live, 0 = spoof.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (DenseNet161_Weights, MobileNet_V3_Large_Weights, densenet161,
                                mobilenet_v3_large)


class MobileNetV3Binary(nn.Module):
    """ImageNet MobileNetV3-Large with a single live/spoof logit.

    The first `freeze_blocks` of the 17 feature blocks stay frozen (partial fine-tuning).
    """

    def __init__(self, pretrained: bool = True, freeze_blocks: int = 4, img_size: int = 224):
        super().__init__()
        net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
        for p in net.features[:freeze_blocks].parameters():
            p.requires_grad = False
        self.net = net

    def forward(self, x):
        return {"logit": self.net(x).squeeze(1)}

    def loss(self, out, y):
        return F.binary_cross_entropy_with_logits(out["logit"], y)

    def score(self, out):
        return torch.sigmoid(out["logit"])


class DeepPixBiS(nn.Module):
    """Deep Pixel-wise Binary Supervision (George & Marcel, ICB 2019).

    DenseNet-161 up to the second transition block gives a 384 x 14 x 14 map (for 224 input).
    A 1x1 conv turns it into a live/spoof map, and every cell is trained to the image label,
    so the model must find spoof evidence across the whole crop. A linear layer on the map
    gives the image-level output. At test time the score is the mean of the map.
    """

    def __init__(self, pretrained: bool = True, freeze_blocks: int = 0, img_size: int = 224):
        super().__init__()
        dense = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
        # conv0, norm0, relu0, pool0, denseblock1, transition1, denseblock2, transition2
        self.encoder = dense.features[:8]
        for p in self.encoder[:freeze_blocks].parameters():
            p.requires_grad = False
        self.pixel = nn.Conv2d(384, 1, kernel_size=1)
        self.fc = nn.Linear((img_size // 16) ** 2, 1)

    def forward(self, x):
        map_logit = self.pixel(self.encoder(x))
        logit = self.fc(torch.sigmoid(map_logit).flatten(1)).squeeze(1)
        return {"logit": logit, "map_logit": map_logit}

    def loss(self, out, y):
        pixel_target = y.view(-1, 1, 1, 1).expand_as(out["map_logit"])
        pixel = F.binary_cross_entropy_with_logits(out["map_logit"], pixel_target)
        binary = F.binary_cross_entropy_with_logits(out["logit"], y)
        return 0.5 * pixel + 0.5 * binary

    def score(self, out):
        return torch.sigmoid(out["map_logit"]).mean(dim=(1, 2, 3))


MODELS = {"mobilenetv3": MobileNetV3Binary, "deeppixbis": DeepPixBiS}


def build_model(name: str, **kwargs) -> nn.Module:
    return MODELS[name](**kwargs)
