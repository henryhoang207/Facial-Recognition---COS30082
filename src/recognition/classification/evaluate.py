"""Classification accuracy only. ROC/AUC uses the shared verification package."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training.config import resolve_device

from .config import RecognitionConfig
from .data import FaceDataset, read_records
from .predictor import load_task


@torch.inference_mode()
def evaluate(
    checkpoint: Path,
    *,
    split: str,
    device: str,
    data_root: Path | None = None,
    batch_size: int | None = None,
) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("classification evaluation supports val and test")
    target_device = resolve_device(device)
    task, payload = load_task(checkpoint, target_device)
    metadata = payload["metadata"]
    config = RecognitionConfig.model_validate(
        {
            "model": metadata["model_config"],
            "preprocess": metadata["preprocess_config"],
            "data": metadata["data_config"],
            "training": payload["training_config"],
            "output_dir": checkpoint.parent,
            "evaluator": metadata["evaluator"],
        }
    )
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if data_root is not None:
        config = config.model_copy(
            update={
                "data": config.data.model_copy(update={"root": data_root.resolve()})
            }
        )
    records = [record for record in read_records(config) if record.split == split]
    mapping = metadata["class_to_idx"]
    if not records or {record.identity for record in records} != set(mapping):
        raise ValueError(
            "evaluation split must contain the same identities as the classifier"
        )
    loader = DataLoader(
        FaceDataset(records, mapping, config.preprocess, train=False),
        batch_size=batch_size or config.training.batch_size,
        shuffle=False,
    )
    correct, count = 0, 0
    for images, labels in tqdm(loader, desc=f"classification {split}"):
        _, logits = task.model(images.to(target_device))
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        count += len(labels)
    return {"split": split, "images": count, "accuracy": correct / count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.checkpoint,
        split=args.split,
        device=args.device,
        data_root=args.data_root,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
