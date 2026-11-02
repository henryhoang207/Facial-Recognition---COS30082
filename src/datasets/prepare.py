"""Build local manifests from read-only source metadata and images."""

import argparse
import json
from pathlib import Path

import yaml

from src.training.config import override_config, read_yaml

from .catalog import write_json
from .config import PreparationConfig
from .emotion import EmotionPreparer
from .liveness import LivenessPreparer
from .recognition import RecognitionPreparer
from .team import TeamPreparer

PREPARERS = {
    "recognition": RecognitionPreparer,
    "liveness": LivenessPreparer,
    "emotion": EmotionPreparer,
    "team": TeamPreparer,
}


def load_config(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--only", nargs="+", choices=tuple(PREPARERS), default=list(PREPARERS)
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    raw = read_yaml(args.config)
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            parser.error("--set requires KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    config = PreparationConfig.model_validate(raw)
    base = args.config.resolve().parent
    updates = {"output_dir": (base / config.output_dir).resolve()}
    for name in PREPARERS:
        source = getattr(config, name)
        updates[name] = source.model_copy(
            update={"root": (base / source.root).resolve()}
        )
    return config.model_copy(update=updates), args.only


def prepare(config: PreparationConfig, selected: list[str]) -> dict:
    for name in selected:
        source = getattr(config, name)
        if config.output_dir == source.root or config.output_dir in source.root.parents:
            raise ValueError("output directory must not contain a source dataset")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        config.output_dir / "preparation_config.json", config.model_dump(mode="json")
    )
    report_path = config.output_dir / "dataset_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    for name in selected:
        print(f"Preparing {name}...", flush=True)
        summary = PREPARERS[name](config).prepare()
        report[name] = summary
        write_json(report_path, report)
        print(json.dumps({name: summary}, ensure_ascii=False), flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    config, selected = load_config(argv)
    prepare(config, selected)


if __name__ == "__main__":
    main()
