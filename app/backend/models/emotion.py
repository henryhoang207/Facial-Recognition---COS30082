"""Load MobileNet + GRU emotion checkpoints for live inference.

The predictor carries GRU hidden state forward per face track.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch import nn


class EmotionNet(nn.Module):
    def __init__(
        self, num_classes: int, hidden: int, dropout: float, *, model_name: str = ""
    ):
        super().__init__()
        if model_name == "mobilenetv4_conv_small_gru":
            import timm

            self.backbone = timm.create_model(
                "mobilenetv4_conv_small", pretrained=False, num_classes=0
            )
            feature_dim = self.backbone.head_hidden_size
        else:
            self.backbone = torchvision.models.mobilenet_v3_large().features
            feature_dim = 960
        self.gru = nn.GRU(feature_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        features = self.backbone(x)
        if features.ndim == 4:
            features = features.mean(dim=(2, 3))
        output, hidden = self.gru(features.unsqueeze(1), hidden)
        return self.head(output[:, -1]), hidden


class EmotionGRUPredictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        config = payload["config"]
        if config.get("task") != "emotion":
            raise ValueError("checkpoint is not an emotion model")
        self.labels: list[str] = list(config["class_names"])
        self.img_size = int(config.get("image_size", 224))
        self.crop_scale = float(config.get("crop_scale", 1.0))
        self.mean = np.array(config["mean"], dtype=np.float32)
        self.std = np.array(config["std"], dtype=np.float32)
        self.model = EmotionNet(
            len(self.labels),
            int(config.get("gru_hidden_size", 128)),
            float(config.get("head_dropout", 0.3)),
            model_name=config.get("model", ""),
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.version = config.get("preprocess_id", "emotion")
        self._hidden: dict[int, torch.Tensor] = {}

    def _tensor(self, face: np.ndarray) -> torch.Tensor:
        if face.shape[:2] != (self.img_size, self.img_size):
            face = cv2.resize(face, (self.img_size, self.img_size))
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

    def _result(self, logits: torch.Tensor) -> tuple[str, dict[str, float]]:
        probs = torch.softmax(logits[0].float(), dim=0).cpu().numpy()
        scores = {label: float(p) for label, p in zip(self.labels, probs, strict=True)}
        return max(scores, key=scores.get), scores

    @torch.inference_mode()
    def predict(self, face: np.ndarray) -> tuple[str, dict[str, float]]:
        logits, _ = self.model(self._tensor(face))
        return self._result(logits)

    @torch.inference_mode()
    def predict_track(self, track_id: int, face: np.ndarray):
        logits, hidden = self.model(self._tensor(face), self._hidden.get(track_id))
        self._hidden[track_id] = hidden
        return self._result(logits)

    def drop_track(self, track_id: int) -> None:
        self._hidden.pop(track_id, None)
