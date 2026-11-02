"""Compare checkpoints and export ROC, AUC and validation-calibrated thresholds."""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import Field

from src.training.config import Settings, override_config, read_yaml
from src.training.trainer import new_run_dir

from .config import VerificationConfig
from .evaluator import PairEvaluator


class ModelSpec(Settings):
    checkpoint: Path
    factory: str = "src.recognition.classification.predictor:RecognitionEmbedder"


class EvaluationConfig(Settings):
    verification: VerificationConfig
    models: dict[str, ModelSpec]
    baseline: bool
    split: Literal["val", "test"]
    size: int = Field(ge=32)
    crop_scale: float = Field(gt=0)
    device: str
    output_dir: Path
    calibration: Path | None


class RawPixelEmbedder:
    """No-learning baseline: flatten the BGR crop in [0,1] and L2-normalize."""

    model_version = "raw_pixels_bgr_01_v1"

    def embed(self, face: np.ndarray) -> np.ndarray:
        vector = face.astype(np.float32).reshape(-1) / 255
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("raw-pixel baseline cannot embed an entirely black crop")
        return vector / norm


class VerificationRunner:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def run(self) -> Path:
        config = self.config
        if config.split == "test" and config.calibration is None:
            raise ValueError("test evaluation requires --calibration from validation")
        if config.split == "val" and config.calibration is not None:
            raise ValueError("calibration is fitted on val; only test consumes it")
        if not config.models and not config.baseline:
            raise ValueError(
                "configure at least one model or enable the raw-pixel baseline"
            )
        if config.baseline and "raw_pixels" in config.models:
            raise ValueError("raw_pixels is reserved for the baseline")
        evaluator = PairEvaluator(
            config.verification, size=config.size, crop_scale=config.crop_scale
        )
        calibration = None
        if config.calibration is not None:
            calibration = json.loads(config.calibration.read_text(encoding="utf-8"))
            if (
                calibration["split"] != "val"
                or calibration["pairs_sha256"] == evaluator.pairs_sha256
            ):
                raise ValueError(
                    "test requires calibration from a separate validation pairs file"
                )
            if (
                calibration["size"] != config.size
                or calibration["crop_scale"] != config.crop_scale
            ):
                raise ValueError(
                    "test preprocessing differs from validation calibration"
                )
        directory = new_run_dir(config.output_dir, f"verification_{config.split}")
        models = {}
        for name, spec in config.models.items():
            module, separator, symbol = spec.factory.partition(":")
            if not separator:
                raise ValueError("embedder factory must be module:callable")
            factory = getattr(importlib.import_module(module), symbol)
            embedder = factory(spec.checkpoint, device=config.device)
            with spec.checkpoint.open("rb") as stream:
                version = hashlib.file_digest(stream, "sha256").hexdigest()
            models[name] = (embedder, version)
        if config.baseline:
            models["raw_pixels"] = (RawPixelEmbedder(), RawPixelEmbedder.model_version)
        report = {
            "split": config.split,
            "protocol": "1:1 verification; not a calibrated 1:N gallery threshold",
            "pairs_sha256": evaluator.pairs_sha256,
            "target_fpr": config.verification.target_fpr,
            "size": config.size,
            "crop_scale": config.crop_scale,
            "models": {},
        }
        rows, curves = [], {}
        pair_rows = []
        for name, (embedder, version) in models.items():
            thresholds = None
            if calibration is not None:
                source = calibration["models"].get(name)
                if source is None or source["model_version"] != version:
                    raise ValueError(
                        f"model {name} differs from its validation calibration"
                    )
                thresholds = {
                    metric: values["score_threshold"]
                    for metric, values in source["metrics"].items()
                }
            metrics, model_curves, labels, scores = evaluator.evaluate(
                embedder, thresholds=thresholds
            )
            report["models"][name] = {"model_version": version, "metrics": metrics}
            for metric, values in metrics.items():
                rows.append(
                    {"model": name, "metric": metric, "split": config.split, **values}
                )
                curves[f"{name}/{metric}"] = (values["auc"], model_curves[metric])
            for index, label in enumerate(labels):
                pair_rows.append(
                    {
                        "model": name,
                        "pair_index": index,
                        "label": int(label),
                        "cosine": float(scores["cosine"][index]),
                        "negative_euclidean": float(scores["euclidean"][index]),
                    }
                )
        if config.split == "val":
            metric = config.verification.selection_metric
            candidates = [name for name in config.models]
            if candidates:
                winner = max(
                    candidates,
                    key=lambda name: report["models"][name]["metrics"][metric]["auc"],
                )
                report["selected"] = {
                    "model": winner,
                    "metric": metric,
                    **report["models"][winner]["metrics"][metric],
                }
        if config.calibration is not None:
            report["calibration"] = str(config.calibration)
            report["target_fpr"] = calibration["target_fpr"]
        (directory / "metrics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
        for filename, values in (("results.csv", rows), ("scores.csv", pair_rows)):
            keys = list(dict.fromkeys(key for row in values for key in row))
            with (directory / filename).open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(values)
        self._plot(curves, directory / "roc.png")
        (directory / "config.json").write_text(
            config.model_dump_json(indent=2), encoding="utf-8"
        )
        return directory

    @staticmethod
    def _plot(curves: dict, output: Path) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(figsize=(7, 6))
        for label, (auc, (fpr, tpr)) in curves.items():
            axes.plot(fpr, tpr, label=f"{label}, AUC={auc:.4f}")
        axes.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        axes.set(
            xlabel="False positive rate",
            ylabel="True positive rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        axes.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--split", choices=["val", "test"])
    parser.add_argument("--target-fpr", type=float)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    raw = read_yaml(args.config)
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            parser.error("--set expects KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    for key in ("calibration", "output_dir", "device", "split"):
        if getattr(args, key) is not None:
            raw[key] = getattr(args, key)
    for argument, key in (
        ("pairs", "pairs"),
        ("data_root", "root"),
        ("target_fpr", "target_fpr"),
    ):
        if getattr(args, argument) is not None:
            raw["verification"][key] = getattr(args, argument)
    for item in args.checkpoint:
        name, separator, path = item.partition("=")
        if not separator or not name:
            parser.error("--checkpoint expects NAME=PATH")
        raw["models"][name] = {**raw["models"].get(name, {}), "checkpoint": path}
    config = EvaluationConfig.model_validate(raw)
    base = args.config.resolve().parent
    config = config.model_copy(
        update={
            "verification": config.verification.model_copy(
                update={
                    "root": (base / config.verification.root).resolve(),
                    "pairs": (base / config.verification.pairs).resolve(),
                }
            ),
            "models": {
                name: spec.model_copy(
                    update={"checkpoint": (base / spec.checkpoint).resolve()}
                )
                for name, spec in config.models.items()
            },
            "output_dir": (base / config.output_dir).resolve(),
            "calibration": (
                (base / config.calibration).resolve()
                if config.calibration is not None
                else None
            ),
        }
    )
    print(VerificationRunner(config).run())


if __name__ == "__main__":
    main()
