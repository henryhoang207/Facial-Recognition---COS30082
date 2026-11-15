"""Checkpoint-backed implementation of src.common.interfaces.Embedder."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from src.training.config import resolve_device

from .config import ModelConfig, PreprocessConfig
from .data import FaceTransform
from .models import ClassificationModel
from .task import RecognitionTask


def load_task(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata.get("task") != "recognition_classification":
        raise ValueError("checkpoint is not a Model A training artifact")
    config = ModelConfig.from_checkpoint(metadata["model_config"])
    model = ClassificationModel(
        config, len(metadata["class_to_idx"]), load_initial_weights=False
    )
    task = RecognitionTask(model)
    task.load_state_dict(payload["task"])
    task.to(device).eval()
    return task, payload


class RecognitionEmbedder:
    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        self.device = resolve_device(device)
        path = Path(checkpoint)
        task, payload = load_task(path, self.device)
        metadata = payload["metadata"]
        self.encoder = task.model.encoder
        self.preprocess_config = PreprocessConfig.model_validate(
            metadata["preprocess_config"]
        )
        self.transform = FaceTransform(self.preprocess_config)
        self.embedding_dim = metadata["model_config"]["embedding_dim"]
        with path.open("rb") as stream:
            self.model_version = hashlib.file_digest(stream, "sha256").hexdigest()
        self.preprocessing_version = hashlib.sha256(
            self.preprocess_config.model_dump_json().encode()
        ).hexdigest()

    @torch.inference_mode()
    def embed(self, face: np.ndarray) -> np.ndarray:
        tensor = self.transform(face).unsqueeze(0).to(self.device)
        result = self.encoder.embed(tensor)[0].cpu().numpy()
        if not np.isfinite(result).all() or not np.isclose(
            np.linalg.norm(result), 1, atol=1e-5
        ):
            raise ValueError("model produced a nonfinite or zero-length embedding")
        return result.astype(np.float32, copy=False)
