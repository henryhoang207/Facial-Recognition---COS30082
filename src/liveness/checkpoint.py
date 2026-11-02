"""Offline inference loading, independent of plotting and training entry points."""

from pathlib import Path

import numpy as np
import torch

from .models import build_model


def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if "model" not in checkpoint or "threshold" not in checkpoint:
        raise ValueError(
            "use the exported liveness_<model>_<date>.pt inference checkpoint"
        )
    config = checkpoint["config"]
    threshold = float(checkpoint["threshold"])
    if not np.isfinite(threshold):
        raise ValueError("checkpoint threshold must be finite")
    model = build_model(
        config["model"], pretrained=False, freeze_blocks=0, img_size=config["img_size"]
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, threshold
