"""Train one static or dynamic ASL gesture classifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.datasets.catalog import write_json
from src.training import (
    Trainer,
    make_optimizer,
    make_scheduler,
    new_run_dir,
    seed_everything,
    seed_worker,
)

from .config import GestureConfig, load_config
from .data import Samples, load_samples
from .evaluate import (
    clip_groups,
    clip_macro_f1,
    evaluate,
    manifest_digest,
    preprocess_settings,
    sample_metadata,
    schema_digest,
)
from .models import ARTIFACT_VERSION, MODEL_NAMES, GestureGRU
from .task import GestureTask


def train_xgboost(
    config, training: Samples, validation: Samples, run: Path, schema: str
):
    import xgboost as xgb

    weights = training.weights / training.weights.mean()
    dtrain = xgb.DMatrix(
        training.inputs, label=training.labels, weight=weights, nthread=1
    )
    dval = xgb.DMatrix(validation.inputs, label=validation.labels, nthread=1)
    groups = clip_groups(validation)

    def validation_metric(probabilities, _data):
        probabilities = probabilities.reshape(
            len(validation.labels), len(validation.classes)
        )
        return "clip_macro_f1", clip_macro_f1(groups, probabilities)

    params = {
        "objective": "multi:softprob",
        "num_class": len(training.classes),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": 3,
        "eta": 0.1,
        "seed": config.seed,
        "nthread": 1,
    }
    history = {}
    booster = xgb.train(
        params,
        dtrain,
        config.xgb_rounds,
        evals=[(dval, "val")],
        evals_result=history,
        custom_metric=validation_metric,
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=20,
                metric_name="clip_macro_f1",
                data_name="val",
                maximize=True,
                save_best=True,
            )
        ],
        verbose_eval=False,
    )
    booster.set_attr(gesture_schema=schema)
    booster.save_model(run / "model.json")
    write_json(
        run / "history.json",
        {
            "xgboost": xgb.__version__,
            "params": params,
            "best_round": booster.best_iteration + 1,
            "history": history,
        },
    )


def train_gru(
    config, training: Samples, validation: Samples, run: Path, metadata: dict, resume
):
    settings = config.training
    seed_everything(settings.seed, settings.deterministic)
    generator = torch.Generator().manual_seed(settings.seed)
    # The shared trainer checkpoints the loader generator used by the sampler.
    sampler = WeightedRandomSampler(
        training.weights, len(training.labels), generator=generator
    )
    options = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(training.inputs), torch.from_numpy(training.labels)
        ),
        generator=generator,
        sampler=sampler,
        **options,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation.inputs), torch.from_numpy(validation.labels)
        ),
        **options,
    )
    task = GestureTask(GestureGRU(training.inputs.shape[-1], len(training.classes)))
    optimizer = make_optimizer(task, settings)
    groups = clip_groups(validation)

    def validation_metrics(task, device):
        probabilities = np.concatenate(
            [
                task.model(inputs.to(device)).softmax(dim=1).cpu().numpy()
                for inputs, _ in val_loader
            ]
        )
        return {"val_clip_macro_f1": clip_macro_f1(groups, probabilities)}

    trainer = Trainer(
        task,
        optimizer,
        settings,
        run,
        scheduler=make_scheduler(optimizer, settings),
        metadata=metadata,
        evaluator=validation_metrics,
    )
    trainer.fit(train_loader, val_loader, resume=resume)


def train(config: GestureConfig, model: str, *, resume: Path | None = None) -> Path:
    if resume is not None and model != "mlp_gru":
        raise ValueError("resume is available for mlp_gru")
    torch.set_num_threads(1)
    training = load_samples(config, "train", model)
    validation = load_samples(config, "val", model)
    schema = schema_digest(training)
    if (
        schema_digest(validation) != schema
        or validation.extractor != training.extractor
    ):
        raise ValueError("training and validation schemas or extractors differ")
    if set(training.clip_ids).intersection(validation.clip_ids):
        raise ValueError("a source clip occurs in both training and validation")
    counts = np.bincount(training.labels, minlength=len(training.classes))
    missing = [
        name for name, count in zip(training.classes, counts, strict=True) if not count
    ]
    if missing:
        raise ValueError(
            f"no usable training samples for classes {missing}; {training.rejected}"
        )

    metadata = {
        "version": ARTIFACT_VERSION,
        "model": model,
        "seed": config.seed,
        "classes": list(training.classes),
        "columns": list(training.columns),
        "schema": schema,
        "preprocess": preprocess_settings(config),
        "extractor": training.extractor,
        "manifest_sha256": manifest_digest(config.manifest),
        "train": sample_metadata(training),
        "val": sample_metadata(validation),
    }
    run = new_run_dir(config.artifacts_dir, model)
    if model == "mlp_gru":
        train_gru(config, training, validation, run, metadata, resume)
    else:
        train_xgboost(config, training, validation, run, schema)
    write_json(run / "resolved_config.json", config.model_dump(mode="json"))
    write_json(run / "metadata.json", metadata)
    return run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    run = train(config, args.model, resume=args.resume)
    report = evaluate(config, run, "val")
    print(
        json.dumps(
            {
                "run": str(run),
                "val_accuracy": report["accuracy"],
                "val_macro_f1": report["macro_f1"],
            }
        )
    )


if __name__ == "__main__":
    main()
