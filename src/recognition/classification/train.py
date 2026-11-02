"""Train A/C without changing another recognition owner's modules."""

import gc
import importlib
import json
import shutil
import warnings
from datetime import date
from pathlib import Path

import torch

from src.training import (
    Trainer,
    make_optimizer,
    make_scheduler,
    new_run_dir,
    seed_everything,
)
from src.training.config import resolve_device

from .cache import cache_loaders
from .config import RecognitionConfig, load_config
from .data import make_loaders
from .models import ClassificationModel
from .task import RecognitionTask
from .verification import make_evaluator


def build_evaluator(config: RecognitionConfig):
    """The configured factory(config) returns an evaluator(task, device)."""
    if config.evaluator is None:
        if config.verification is not None:
            return make_evaluator(config)
        if config.training.monitor not in {"val_loss", "val_accuracy"}:
            raise ValueError("custom monitor requires --evaluator module:factory")
        return None
    module, separator, attribute = config.evaluator.partition(":")
    if not separator:
        raise ValueError("evaluator must be module:factory")
    evaluator = getattr(importlib.import_module(module), attribute)(config)
    if not callable(evaluator):
        raise ValueError("evaluator factory must return a callable(task, device)")
    return evaluator


def train(config: RecognitionConfig, *, resume: Path | None = None):
    directory = new_run_dir(config.output_dir, f"recognition_{config.model.variant}")
    pipeline = {}
    try:
        return _train_once(config, resume, directory, pipeline, cache_allowed=True)
    except torch.cuda.OutOfMemoryError:
        if pipeline.get("mode") != "cuda":
            raise
        warnings.warn(
            "CUDA ran out of memory with cached crops. Releasing the cache and "
            "retrying with batch streaming from the last completed epoch, or "
            "the original initialization if no epoch was saved.",
            stacklevel=2,
        )
    # Leave the exception scope first to release its traceback and live CUDA tensors.
    gc.collect()
    torch.cuda.empty_cache()
    last = directory / "last.pt"
    (directory / "data_pipeline.json").write_text(
        json.dumps({**pipeline, "outcome": "training_cuda_oom"}, indent=2),
        encoding="utf-8",
    )
    return _train_once(
        config,
        last if last.exists() else resume,
        new_run_dir(config.output_dir, f"recognition_{config.model.variant}_streaming"),
        pipeline,
        cache_allowed=False,
    )


def _train_once(config, resume, directory, pipeline, *, cache_allowed):
    evaluator = build_evaluator(config)
    seed_everything(config.training.seed, config.training.deterministic)
    train_loader, val_loader, mapping = make_loaders(config)
    task = RecognitionTask(
        ClassificationModel(
            config.model, len(mapping), load_initial_weights=resume is None
        ),
        config.preprocess,
    ).to(resolve_device(config.training.device))
    optimizer = make_optimizer(task, config.training)
    (train_loader, val_loader), report = cache_loaders(
        (train_loader, val_loader),
        config,
        next(task.parameters()).device,
        allowed=cache_allowed,
    )
    pipeline.update(report)
    print(json.dumps({"data_pipeline": report}))
    metadata = {
        "task": "recognition_classification",
        "model_config": config.model.model_dump(mode="json"),
        "preprocess_config": config.preprocess.model_dump(mode="json"),
        "data_config": config.data.model_dump(mode="json"),
        "class_to_idx": mapping,
        "evaluator": config.evaluator,
        "verification": (
            config.verification.model_dump(mode="json")
            if config.verification is not None
            else None
        ),
        "splits": {
            name: [
                {
                    "path": str(record.path),
                    "identity": record.identity,
                    "box": list(record.box) if record.box is not None else None,
                    "session": record.session,
                    "group": record.group,
                    "sha256": record.sha256,
                }
                for record in loader.dataset.records
            ]
            for name, loader in (("train", train_loader), ("val", val_loader))
        },
    }
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
    (directory / "data_pipeline.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (directory / "resolved_config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8"
    )
    checkpoint = (
        directory / f"recognition_{config.model.variant}_{date.today().isoformat()}.pt"
    )
    shutil.copy2(result.best_checkpoint, checkpoint)
    plot_history(result.history, directory / "training.png")
    print(json.dumps({"checkpoint": str(checkpoint), "best_epoch": result.best_epoch}))
    return result


def plot_history(history: list[dict], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (losses, metrics) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    for key in ("train_loss", "val_loss"):
        losses.plot(epochs, [row[key] for row in history], label=key, marker="o")
    for key in ("train_accuracy", "val_accuracy", "val_auc"):
        if key in history[0]:
            metrics.plot(epochs, [row[key] for row in history], label=key, marker="o")
    losses.set(xlabel="Epoch", ylabel="Loss")
    metrics.set(xlabel="Epoch", ylabel="Accuracy / AUC", ylim=(0, 1))
    losses.legend()
    metrics.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    config, resume = load_config(argv)
    train(config, resume=resume)


if __name__ == "__main__":
    main()
