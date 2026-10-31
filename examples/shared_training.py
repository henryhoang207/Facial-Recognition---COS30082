"""Train a small synthetic regression task with the shared YAML-driven trainer."""

import argparse
import json
from pathlib import Path

import torch
import yaml
from pydantic import Field
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    StepResult,
    Trainer,
    TrainingConfig,
    TrainingTask,
    make_optimizer,
    make_scheduler,
    new_run_dir,
    seed_everything,
    seed_worker,
)
from src.training.config import Settings, override_config, read_yaml, resolve_device


class ExampleDataConfig(Settings):
    train_samples: int = Field(gt=0)
    val_samples: int = Field(gt=0)
    input_features: int = Field(gt=0)
    noise_std: float = Field(ge=0)


class ExampleConfig(Settings):
    training: TrainingConfig
    data: ExampleDataConfig
    output_dir: Path


class RegressionTask(TrainingTask):
    def __init__(self, input_features: int):
        super().__init__()
        self.model = nn.Linear(input_features, 1)

    def step(self, batch) -> StepResult:
        inputs, targets = batch
        return StepResult(F.mse_loss(self.model(inputs), targets), len(targets))


def make_loaders(config: ExampleConfig) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(config.training.seed)
    coefficients = torch.arange(1, config.data.input_features + 1).float()
    loaders = []
    for split, count in (
        ("train", config.data.train_samples),
        ("val", config.data.val_samples),
    ):
        inputs = torch.randn(count, config.data.input_features, generator=generator)
        noise = torch.randn(count, 1, generator=generator) * config.data.noise_std
        targets = inputs @ coefficients.unsqueeze(1) + noise
        loaders.append(
            DataLoader(
                TensorDataset(inputs, targets),
                batch_size=config.training.batch_size,
                shuffle=split == "train",
                num_workers=config.training.num_workers,
                generator=torch.Generator().manual_seed(config.training.seed),
                worker_init_fn=seed_worker,
                persistent_workers=False,
            )
        )
    return loaders[0], loaders[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    for flag, value_type in (("epochs", int), ("lr", float), ("device", str)):
        parser.add_argument(f"--{flag}", type=value_type, default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    raw = read_yaml(args.config)
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error("--set expects KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    for key in ("epochs", "lr", "device"):
        if hasattr(args, key):
            override_config(raw, f"training.{key}", getattr(args, key))
    config = ExampleConfig.model_validate(raw)
    directory = (args.config.resolve().parent / config.output_dir).resolve()
    seed_everything(config.training.seed, config.training.deterministic)
    task = RegressionTask(config.data.input_features).to(
        resolve_device(config.training.device)
    )
    optimizer = make_optimizer(task, config.training)
    trainer = Trainer(
        task,
        optimizer,
        config.training,
        new_run_dir(directory, "shared_example"),
        scheduler=make_scheduler(optimizer, config.training),
        metadata={
            "example": "synthetic_regression",
            "data": config.data.model_dump(mode="json"),
        },
    )
    result = trainer.fit(*make_loaders(config), resume=args.resume)
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "best_checkpoint": str(result.best_checkpoint),
                "last_checkpoint": str(result.last_checkpoint),
            }
        )
    )


if __name__ == "__main__":
    main()
