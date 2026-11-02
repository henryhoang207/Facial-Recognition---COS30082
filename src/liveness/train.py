"""Train liveness with the shared resumable trainer; export the app format."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

from src.training import (
    Trainer,
    make_optimizer,
    make_scheduler,
    new_run_dir,
    seed_everything,
)
from src.training.config import resolve_device
from src.training.trainer import _atomic_save

from .config import LivenessConfig, load_config
from .data import load_manifests, make_loader
from .metrics import eer_threshold, summarize
from .models import build_model
from .provenance import audit_images, make_provenance
from .task import LivenessTask

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def source_identity():
    root = Path(__file__).parents[2]
    bundle = root / "SOURCE_REVISION.json"
    if bundle.exists():
        revision = json.loads(bundle.read_text(encoding="utf-8"))["commit"]
    else:
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unavailable"
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("src/liveness", "src/training", "src/datasets")
        for path in sorted((root / folder).glob("*.py"))
    }
    return {
        "commit": revision,
        "code_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


@torch.inference_mode()
def predict_scores(model, loader, device) -> np.ndarray:
    model.eval()
    scores = [
        model.score(model(images.to(device))).double().cpu() for images, _ in loader
    ]
    if not scores:
        raise ValueError("empty evaluation loader")
    return torch.cat(scores).numpy()


def plot_history(history: pd.DataFrame, path: Path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for column in ("train_loss", "val_loss"):
        axes[0].plot(history.epoch, history[column], label=column)
    for column in ("val_acer", "val_eer"):
        axes[1].plot(history.epoch, history[column] * 100, label=column)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Validation error (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def subset(frame, maximum, seed):
    """Deterministic stratified pilots; full data is the default."""
    if maximum is None or maximum >= len(frame):
        return frame.reset_index(drop=True)
    from sklearn.model_selection import train_test_split

    if maximum < frame.attack.nunique():
        raise ValueError("pilot cap must cover every attack group")
    selected, _ = train_test_split(
        frame, train_size=maximum, stratify=frame.attack, random_state=seed
    )
    return selected.sort_values("path").reset_index(drop=True)


def train(config: LivenessConfig, *, resume: Path | None = None):
    seed_everything(config.training.seed, config.training.deterministic)
    frame = audit_images(load_manifests(list(config.manifest)), config.data_root)
    for split in ("train", "val"):
        if set(frame.loc[frame.split == split, "label"]) != {0, 1}:
            raise ValueError(f"{split} requires live and spoof")
    train_frame = subset(
        frame[frame.split == "train"], config.max_train, config.training.seed
    )
    val_frame = subset(
        frame[frame.split == "val"], config.max_val, config.training.seed
    )
    loader_options = dict(
        data_root=config.data_root,
        size=config.img_size,
        crop_scale=config.crop_scale,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        seed=config.training.seed,
    )
    train_loader = make_loader(train_frame, train=True, **loader_options)
    val_loader = make_loader(val_frame, train=False, **loader_options)
    score_loader = make_loader(val_frame, train=False, **loader_options)
    task = LivenessTask(
        build_model(
            config.model,
            pretrained=config.pretrained and resume is None,
            freeze_blocks=config.freeze_blocks,
            img_size=config.img_size,
        )
    )
    task.to(resolve_device(config.training.device))

    def evaluator(task, device):
        scores = predict_scores(task.model, score_loader, device)
        threshold, _ = eer_threshold(scores, val_frame.label)
        metrics = summarize(scores, val_frame.label, val_frame.attack, threshold)
        return {
            "val_" + key: metrics[key]
            for key in ("acer", "eer", "auc", "threshold", "apcer", "bpcer")
        }

    portable_config = config.model_dump(
        mode="json",
        exclude={"data_root", "manifest", "output_dir", "fig_dir", "training"},
    )
    metadata = {
        "task": "liveness",
        "source": source_identity(),
        "model_config": portable_config,
        "provenance": make_provenance(frame),
        "selected_train_paths": sorted(train_frame.path.tolist()),
        "selected_val_paths": sorted(val_frame.path.tolist()),
    }
    directory = new_run_dir(
        config.output_dir, "liveness_" + (config.tag or config.model)
    )
    optimizer = make_optimizer(task, config.training)
    trainer = Trainer(
        task,
        optimizer,
        config.training,
        directory,
        scheduler=make_scheduler(optimizer, config.training),
        metadata=metadata,
        evaluator=evaluator,
    )
    result = trainer.fit(train_loader, val_loader, resume=resume)
    (directory / "runtime.json").write_text(
        json.dumps(
            {
                "gpu_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
                "training_examples": len(train_frame),
                "validation_examples": len(val_frame),
                "parameters": sum(parameter.numel() for parameter in task.parameters()),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in task.parameters()
                    if parameter.requires_grad
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best = torch.load(result.best_checkpoint, map_location="cpu", weights_only=True)
    cfg = {
        **portable_config,
        "manifest": [str(path) for path in config.manifest],
        "provenance": metadata["provenance"],
        "training": config.training.model_dump(mode="json"),
        "run_id": directory.name,
        "source": metadata["source"],
    }
    date = datetime.now(UTC).date().isoformat()
    checkpoint = directory / f"liveness_{config.tag or config.model}_{date}.pt"
    _atomic_save(
        {
            "model": {
                key.removeprefix("model."): value for key, value in best["task"].items()
            },
            "config": cfg,
            "threshold": best["val_metrics"]["val_threshold"],
            "epoch": best["best_epoch"],
            "val_metrics": best["val_metrics"],
        },
        checkpoint,
    )
    frame.to_csv(directory / "manifest.csv", index=False)
    (directory / "resolved_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / "source.json").write_text(
        json.dumps(
            {**metadata["source"], "resume_from": str(resume) if resume else None}
        ),
        encoding="utf-8",
    )
    config.fig_dir.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(result.history)
    tag = config.tag or config.model
    history.to_csv(config.fig_dir / f"liveness_{tag}_history.csv", index=False)
    plot_history(history, config.fig_dir / f"liveness_{tag}_training.png")
    (directory / "export.json").write_text(
        json.dumps({"checkpoint": str(checkpoint), "best_epoch": result.best_epoch}),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "run_dir": str(directory),
                "best_epoch": result.best_epoch,
            }
        )
    )
    return result


def main(argv=None):
    config, resume = load_config(argv)
    train(config, resume=resume)


if __name__ == "__main__":
    main()
