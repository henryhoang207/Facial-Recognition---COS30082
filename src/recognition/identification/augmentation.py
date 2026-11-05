"""Apply deterministic per-draw augmentation on the training batch's device."""

from __future__ import annotations

import torch


@torch.no_grad()
def augment_training_batch(
    images: torch.Tensor, augmentation_params: torch.Tensor
) -> torch.Tensor:
    """Transform RGB uint8 BCHW crops into normalized float32 model inputs.

    Workers only decode/crop images and draw the small ``[flip, brightness,
    contrast]`` parameter vector. Pixel operations run here after the trainer
    transfers the batch to CUDA. Sampling parameters per draw makes augmentation
    independent of worker scheduling and leaves the loss-mining RNG untouched.
    CPU tensors use the same operations for bounded integration tests.
    """
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Training augmentation requires RGB uint8 BCHW images.")
    if augmentation_params.shape != (images.shape[0], 3):
        raise ValueError("Augmentation parameters must have shape [batch, 3].")
    if augmentation_params.device != images.device:
        raise ValueError("Images and augmentation parameters must share a device.")
    with torch.autocast(images.device.type, enabled=False):
        images = images.float().div_(255)
        params = augmentation_params.float().view(-1, 3, 1, 1)
        images = torch.where(params[:, :1].bool(), images.flip(-1), images)
        images = (images * params[:, 1:2]).clamp_(0, 1)
        weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        mean = (images * weights).sum(1, keepdim=True).mean((2, 3), keepdim=True)
        images = ((images - mean) * params[:, 2:3] + mean).clamp_(0, 1)
        return images.mul_(2).sub_(1)
