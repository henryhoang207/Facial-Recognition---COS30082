"""A/C-local crop adapters, replaceable by Thành's shared loaders.

Folder input contains already detected face images. CSV input can additionally
provide pixel bounding boxes for original frames. No automatic split is made.
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.face_crop import crop_face
from src.training import seed_worker

from .augmentation import FaceBatchTransform
from .config import PreprocessConfig, RecognitionConfig


@dataclass(frozen=True)
class FaceRecord:
    path: Path
    identity: str
    split: str
    box: tuple[int, int, int, int] | None = None
    session: str | None = None
    group: str | None = None
    sha256: str | None = None


class FaceTransform:
    def __init__(self, config: PreprocessConfig, *, train: bool = False):
        self.size = config.size
        self.transform = FaceBatchTransform(config).train(train)

    def __call__(self, face: np.ndarray) -> torch.Tensor:
        if face.dtype != np.uint8 or face.shape != (self.size, self.size, 3):
            raise ValueError(
                f"expected BGR uint8 crop of shape ({self.size}, {self.size}, 3)"
            )
        rgb = torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        return self.transform(rgb.unsqueeze(0))[0]


class FaceDataset(Dataset):
    def __init__(
        self,
        records: list[FaceRecord],
        class_to_idx: dict[str, int],
        config: PreprocessConfig,
        *,
        train: bool,
        raw: bool = False,
    ):
        self.records = records
        self.class_to_idx = class_to_idx
        self.config = config
        # Raw decoder workers carry no tensor buffers or device transforms.
        self.transform = None if raw else FaceTransform(config, train=train)
        self.raw = raw
        self.cached_images: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if self.cached_images is not None:
            return self.cached_images[index], self.class_to_idx[record.identity]
        frame = cv2.imread(str(record.path))
        if frame is None:
            raise ValueError(f"cannot decode image: {record.path}")
        height, width = frame.shape[:2]
        box = record.box or (0, 0, width, height)
        face = crop_face(frame, box, self.config.size, self.config.crop_scale)
        image = (
            torch.from_numpy(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .contiguous()
            if self.raw
            else self.transform(face)
        )
        return image, self.class_to_idx[record.identity]


def read_records(config: RecognitionConfig) -> list[FaceRecord]:
    data = config.data
    if data.manifest is not None:
        records = []
        with data.manifest.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not {"path", "identity", "split"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV requires path,identity,split columns")
            for row in reader:
                coordinates = [row.get(key, "") for key in ("x1", "y1", "x2", "y2")]
                if any(coordinates) and not all(coordinates):
                    raise ValueError("provide all four bbox coordinates or none")
                box = (
                    tuple(int(value) for value in coordinates)
                    if all(coordinates)
                    else None
                )
                if box is not None and (box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError("bbox must have positive width and height")
                if row["split"] not in {"train", "val", "test"} or not row["identity"]:
                    raise ValueError("CSV needs identity and split train, val or test")
                records.append(
                    FaceRecord(
                        Path(os.path.abspath(data.root / row["path"])),
                        row["identity"],
                        row["split"],
                        box,
                        row.get("session") or None,
                        row.get("group") or None,
                        row.get("sha256") or None,
                    )
                )
        return records
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    records = []
    for split, directory in (
        ("train", data.train_dir),
        ("val", data.val_dir),
        ("test", data.test_dir),
    ):
        for path in sorted((data.root / directory).glob("*/*")):
            if path.is_file() and path.suffix.lower() in extensions:
                records.append(FaceRecord(path.resolve(), path.parent.name, split))
    return records


def make_loaders(
    config: RecognitionConfig,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    records = read_records(config)
    # Test records are not opened during training or used for checkpoint selection.
    active = [record for record in records if record.split in {"train", "val"}]
    paths: set[Path] = set()
    sessions: dict[tuple[str, str], str] = {}
    groups: dict[tuple[str, str], str] = {}
    hashes: dict[str, str] = {}
    for record in active:
        if not record.path.is_file():
            raise ValueError(f"image does not exist: {record.path}")
        if record.path in paths:
            raise ValueError(f"duplicate image path in train/val: {record.path}")
        paths.add(record.path)
        if record.session is not None:
            key = (record.identity, record.session)
            if key in sessions and sessions[key] != record.split:
                raise ValueError(f"identity/session appears in train and val: {key}")
            sessions[key] = record.split
        if record.group is not None:
            key = (record.identity, record.group)
            if key in groups and groups[key] != record.split:
                raise ValueError(f"image group appears in train and val: {key}")
            groups[key] = record.split
        if record.sha256 is not None:
            if record.sha256 in hashes and hashes[record.sha256] != record.split:
                raise ValueError("identical image content appears in train and val")
            hashes[record.sha256] = record.split
    by_split = {
        split: [r for r in active if r.split == split] for split in ("train", "val")
    }
    train_classes = {record.identity for record in by_split["train"]}
    val_classes = {record.identity for record in by_split["val"]}
    if len(train_classes) < 2 or train_classes != val_classes:
        raise ValueError(
            "train and val must contain the same set of at least two identities"
        )
    mapping = {name: index for index, name in enumerate(sorted(train_classes))}
    loaders = []
    for split in ("train", "val"):
        generator = torch.Generator().manual_seed(config.training.seed)
        loaders.append(
            DataLoader(
                FaceDataset(
                    by_split[split],
                    mapping,
                    config.preprocess,
                    train=split == "train",
                    raw=True,
                ),
                batch_size=config.training.batch_size,
                shuffle=split == "train",
                num_workers=config.training.num_workers,
                pin_memory=(
                    config.training.device != "cpu" and torch.cuda.is_available()
                ),
                worker_init_fn=seed_worker,
                generator=generator,
                persistent_workers=False,
            )
        )
    return loaders[0], loaders[1], mapping
