"""Independent per-image tensor augmentation on the model's device."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import PreprocessConfig


class FaceBatchTransform(nn.Module):
    """Accept RGB uint8 BCHW crops; emit normalized float32 images.

    Random transforms run only in training mode, after selecting each batch.
    Buffers are derived from config and do not change checkpoint weight keys.
    """

    def __init__(self, config: PreprocessConfig):
        super().__init__()
        self.config = config
        for name, values in (("mean", config.mean), ("std", config.std)):
            self.register_buffer(
                name, torch.tensor(values).view(1, 3, 1, 1), persistent=False
            )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(images.device.type, enabled=False):
            images = images.float().div(255)
            if self.training:
                images = self._augment(images)
            return (images - self.mean) / self.std

    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        config = self.config
        count = images.shape[0]

        def uniform(amount: float):
            return torch.empty(count, 1, 1, 1, device=images.device).uniform_(
                1 - amount, 1 + amount
            )

        if config.horizontal_flip:
            mask = torch.rand(count, 1, 1, 1, device=images.device)
            images = torch.where(mask < config.horizontal_flip, images.flip(-1), images)
        if config.rotation_degrees or config.translate:
            angles = torch.empty(count, device=images.device).uniform_(
                -config.rotation_degrees, config.rotation_degrees
            ) * (math.pi / 180)
            theta = torch.zeros(count, 2, 3, device=images.device)
            theta[:, 0, 0] = theta[:, 1, 1] = angles.cos()
            theta[:, 0, 1], theta[:, 1, 0] = -angles.sin(), angles.sin()
            # affine_grid coordinates span [-1, 1], hence twice the image fraction.
            theta[:, :, 2].uniform_(-2 * config.translate, 2 * config.translate)
            grid = F.affine_grid(theta, images.shape, align_corners=False)
            images = F.grid_sample(
                images, grid, padding_mode="zeros", align_corners=False
            )
        if config.brightness:
            images = (images * uniform(config.brightness)).clamp(0, 1)
        if config.contrast or config.saturation:
            weights = images.new_tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1)
            gray = (images * weights).sum(1, keepdim=True)
            if config.contrast:
                mean = gray.mean((2, 3), keepdim=True)
                images = ((images - mean) * uniform(config.contrast) + mean).clamp(0, 1)
            if config.saturation:
                gray = (images * weights).sum(1, keepdim=True)
                images = ((images - gray) * uniform(config.saturation) + gray).clamp(
                    0, 1
                )
        return images
