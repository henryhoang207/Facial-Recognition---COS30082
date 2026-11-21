"""Train a static ASL landmark MLP with the shared PyTorch trainer."""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from src.training import (
    MetricTotal,
    StepResult,
    Trainer,
    TrainingTask,
    make_optimizer,
    make_scheduler,
    seed_everything,
)
from src.training.config import resolve_device

from .alphabet_features import augment_points, class_weights, pose_features
from .features import STATIC_COLUMNS


class AlphabetMLP(nn.Sequential):
    def __init__(self, class_count: int, dropout: float):
        super().__init__(
            nn.Linear(len(STATIC_COLUMNS), 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, class_count),
        )


class AlphabetTask(TrainingTask):
    def __init__(self, class_count: int, config, weights: torch.Tensor):
        super().__init__()
        self.model = AlphabetMLP(class_count, config.dropout)
        self.register_buffer("weights", weights, persistent=False)
        self.augmentation = (
            config.flip_probability,
            config.rotation_degrees,
            config.jitter_std,
        )

    def step(self, batch) -> StepResult:
        points, labels = batch
        with torch.no_grad(), torch.autocast(points.device.type, enabled=False):
            if self.training:
                points = augment_points(points, *self.augmentation)
            features = pose_features(points)
        logits = self.model(features)
        return StepResult(
            cross_entropy(logits.float(), labels, weight=self.weights),
            len(labels),
            {
                "accuracy": MetricTotal(
                    (logits.argmax(1) == labels).sum().item(), len(labels)
                )
            },
        )


def train(
    config,
    points: np.ndarray,
    labels: np.ndarray,
    splits: np.ndarray,
    metadata: dict,
    run: Path,
    resume: Path | None = None,
):
    settings = config.training
    if settings.num_workers != 0:
        raise ValueError(
            "alphabet training keeps landmarks on-device and needs zero loader workers"
        )
    seed_everything(settings.seed, settings.deterministic)
    device = resolve_device(settings.device)
    class_count = len(metadata["classes"])
    indices = {name: splits == name for name in ("train", "val")}
    resident = {}
    for name, mask in indices.items():
        targets = labels[mask]
        if set(targets.tolist()) != set(range(class_count)):
            raise ValueError(f"{name} needs usable landmarks for every alphabet class")
        resident[name] = (
            torch.as_tensor(points[mask], dtype=torch.float32, device=device),
            torch.as_tensor(targets, dtype=torch.long, device=device),
        )
    dataset = TensorDataset(*resident["train"])
    generator = torch.Generator().manual_seed(settings.seed)
    batches = BatchSampler(
        RandomSampler(dataset, generator=generator),
        settings.batch_size,
        drop_last=False,
    )
    # Each sampler index is a full batch, avoiding per-row device operations.
    loader = DataLoader(dataset, batch_size=None, sampler=batches, generator=generator)
    clean = {}
    with torch.no_grad():
        for name, (raw, targets) in resident.items():
            clean[name] = (pose_features(raw), targets)

    def metrics(task, _device):
        results = {}
        for name, (features, targets) in clean.items():
            logits = torch.cat([task.model(part) for part in features.split(8192)])
            predicted = logits.argmax(1)
            prefix = "train_clean" if name == "train" else "val"
            results[f"{prefix}_loss"] = cross_entropy(logits, targets).item()
            results[f"{prefix}_accuracy"] = (predicted == targets).float().mean().item()
            if name == "val":
                results["val_macro_f1"] = f1_score(
                    labels[indices["val"]],
                    predicted.cpu().numpy(),
                    labels=range(class_count),
                    average="macro",
                    zero_division=0,
                )
        return results

    weights = torch.from_numpy(class_weights(labels[indices["train"]], class_count))
    task = AlphabetTask(class_count, config, weights).to(device)
    optimizer = make_optimizer(task, settings)
    return Trainer(
        task,
        optimizer,
        settings,
        run,
        scheduler=make_scheduler(optimizer, settings),
        metadata=metadata,
        evaluator=metrics,
    ).fit(loader, resume=resume)


def load_alphabet_predictor(run: Path, device: str = "cpu"):
    """Load the best validation checkpoint and predict from unaugmented landmarks."""
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    metadata = saved.get("metadata", {})
    if (
        saved.get("format_version") != 1
        or saved.get("checkpoint_kind") != "inference"
        or metadata.get("model") != "mlp"
    ):
        raise ValueError(f"incompatible alphabet MLP checkpoint: {run}")
    model = AlphabetMLP(len(metadata["classes"]), dropout=0)
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in saved["task"].items()}
    )
    model.to(device).eval()

    @torch.inference_mode()
    def predict(points: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
        return model(pose_features(tensor)).softmax(1).cpu().numpy()

    return metadata, predict
