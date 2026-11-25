"""Dataset over the face crops written by prepare.py."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .model import CLASSES


def read_manifest(faces_dir: Path, split: str) -> list[dict]:
    with (faces_dir / "manifest.csv").open(encoding="utf-8") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == split]


class MaskFaces(Dataset):
    def __init__(self, faces_dir: Path, split: str, transform, limit: int | None = None):
        self.faces_dir = faces_dir
        self.rows = read_manifest(faces_dir, split)[:limit]
        if not self.rows:
            raise ValueError(f"no {split} faces in {faces_dir / 'manifest.csv'}")
        self.transform = transform
        self.targets = [CLASSES.index(r["label"]) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.faces_dir / row["path"]).convert("RGB")
        return self.transform(image), self.targets[index]


def make_loader(
    dataset: Dataset, batch_size: int, shuffle: bool, workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )
