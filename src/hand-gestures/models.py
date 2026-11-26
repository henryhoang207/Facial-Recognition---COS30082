"""Landmark classifiers and inference artifact loading."""

import json
from pathlib import Path

import torch
from torch import nn

MODEL_NAMES = ("static_xgb", "motion_xgb", "mlp_gru")
ARTIFACT_VERSION = 1


class GestureGRU(nn.Module):
    def __init__(self, feature_count: int, class_count: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_count, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.classifier = nn.Linear(64, class_count)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, state = self.gru(self.encoder(sequence))
        return self.classifier(state[-1])


def load_predictor(run: Path):
    """Return run metadata and a CPU callable from feature batches to probabilities."""
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("version") != ARTIFACT_VERSION
        or metadata.get("model") not in MODEL_NAMES
    ):
        raise ValueError(f"incompatible gesture artifact: {run}")
    feature_count = len(metadata["columns"])
    class_count = len(metadata["classes"])
    if metadata["model"] == "mlp_gru":
        from .task import GestureTask

        saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        if (
            saved.get("checkpoint_kind") != "inference"
            or saved.get("format_version") != 1
            or saved.get("metadata") != metadata
        ):
            raise ValueError(f"gesture checkpoint metadata mismatch: {run}")
        task = GestureTask(GestureGRU(feature_count, class_count))
        task.load_state_dict(saved["task"])
        task.eval()

        @torch.inference_mode()
        def predict(inputs):
            return task.model(torch.from_numpy(inputs)).softmax(dim=1).numpy()

        return metadata, predict

    import xgboost as xgb

    booster = xgb.Booster(params={"nthread": 1})
    booster.load_model(run / "model.json")
    if (
        booster.num_features() != feature_count
        or booster.attr("gesture_schema") != metadata["schema"]
    ):
        raise ValueError(f"gesture booster feature schema mismatch: {run}")
    return metadata, booster.inplace_predict
