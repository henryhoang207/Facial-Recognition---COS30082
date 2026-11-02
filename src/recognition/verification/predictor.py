"""Common-checkpoint metric encoder implementing the shared Embedder contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from src.training.config import TrainingConfig, resolve_device

from .models import IResNet34

METRIC_PREPROCESSING = {
    "size": 112,
    "scale": 1.0,
    "color": "RGB",
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}


def _validate_payload(payload: object, *, allow_training: bool) -> dict:
    kinds = {"inference", "training"} if allow_training else {"inference"}
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or not isinstance(payload.get("checkpoint_kind"), str)
        or payload.get("checkpoint_kind") not in kinds
    ):
        raise ValueError("expected a supported common metric inference checkpoint")
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("task") != "recognition_metric_learning"
    ):
        raise ValueError("checkpoint is not a metric-learning training artifact")
    experiment = metadata.get("experiment_config")
    method = metadata.get("method")
    margin = metadata.get("margin")
    class_count = metadata.get("class_count")
    if (
        not isinstance(method, str)
        or method not in {"triplet", "arcface"}
        or not isinstance(experiment, Mapping)
        or experiment.get("method") != method
        or metadata.get("embedding_dim") != 256
        or experiment.get("embedding_dim") != 256
        or not isinstance(class_count, int)
        or class_count < 2
        or not isinstance(margin, (float, int))
        or not math.isfinite(margin)
        or margin <= 0
        or not isinstance(metadata.get("signature"), str)
        or not metadata["signature"]
        or not isinstance(metadata.get("environment_compatibility"), Mapping)
    ):
        raise ValueError("invalid metric-learning checkpoint metadata")
    if metadata.get("preprocessing") != METRIC_PREPROCESSING:
        raise ValueError("unsupported metric-learning checkpoint preprocessing")
    config = payload.get("training_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint is missing the common training_config")
    TrainingConfig.model_validate(config)
    state = payload.get("task")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint is missing the metric task state")
    if any(
        not isinstance(key, str)
        or (key != "_extra_state" and not key.startswith(("model.", "head.")))
        for key in state
    ):
        raise ValueError("unsupported metric task state layout")
    return payload


def load_metric_encoder(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    *,
    allow_training: bool = False,
) -> tuple[IResNet34, dict]:
    """Load the encoder from the common schema without constructing a loss head.

    Public inference accepts best.pt or its embedding.pt copy. Internal tooling
    can explicitly allow last.pt to inspect the current training encoder.
    Legacy metric checkpoints require a new run and are rejected.
    """
    target = resolve_device(str(device))
    payload = _validate_payload(
        torch.load(Path(checkpoint), map_location="cpu", weights_only=True),
        allow_training=allow_training,
    )
    state = {
        name.removeprefix("model."): value
        for name, value in payload["task"].items()
        if name.startswith("model.")
    }
    # Constructing a model for inference must not consume the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        model = IResNet34(embedding_dim=256)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("checkpoint encoder weights do not match iResNet34") from exc
    model.to(target).eval()
    return model, payload


class MetricEmbedder:
    """Embed a shared BGR uint8 112x112 face crop using iResNet34."""

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.device = resolve_device(device)
        path = Path(checkpoint)
        self.encoder, self.payload = load_metric_encoder(path, self.device)
        self.embedding_dim = self.payload["metadata"]["embedding_dim"]
        self.preprocessing = dict(self.payload["metadata"]["preprocessing"])
        with path.open("rb") as stream:
            self.model_version = hashlib.file_digest(stream, "sha256").hexdigest()
        self.preprocessing_version = hashlib.sha256(
            json.dumps(self.preprocessing, sort_keys=True).encode()
        ).hexdigest()

    @torch.inference_mode()
    def embed(self, face: np.ndarray) -> np.ndarray:
        if (
            not isinstance(face, np.ndarray)
            or face.dtype != np.uint8
            or face.shape != (112, 112, 3)
        ):
            raise ValueError("expected BGR uint8 crop of shape (112, 112, 3)")
        rgb = face[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb / 127.5 - 1.0))
        output = self.encoder(tensor.unsqueeze(0).to(self.device))
        if not isinstance(output, torch.Tensor) or output.shape != (1, 256):
            raise ValueError("model must return one 256-dimensional embedding")
        vector = output[0].float().cpu().numpy()
        if not np.isfinite(vector).all() or not np.isclose(
            np.linalg.norm(vector), 1, atol=1e-5
        ):
            raise ValueError("model produced a nonfinite or zero-length embedding")
        return vector.astype(np.float32, copy=False)
