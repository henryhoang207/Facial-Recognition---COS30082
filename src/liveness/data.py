"""Liveness dataset: reads manifest CSVs (see manifest.py) and crops faces on the fly."""

from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from src.common.face_crop import crop_face

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_manifests(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def eval_transform() -> v2.Compose:
    """Applied to an RGB uint8 HWC crop. The live app uses the same one (predictor.py)."""
    return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def train_transform(size: int) -> v2.Compose:
    # Mild augmentation only: colour, blur and moiré are spoof cues, so strong jitter would hide them
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomApply([v2.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.1, hue=0.02)], p=0.8),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LivenessDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, size: int, crop_scale: float, train: bool):
        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.size, self.crop_scale = size, crop_scale
        self.transform = train_transform(size) if train else eval_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        frame = cv2.imread(str(self.data_root / r.path))
        if frame is None:
            raise FileNotFoundError(self.data_root / r.path)
        face = crop_face(frame, (r.x1, r.y1, r.x2, r.y2), size=self.size, scale=self.crop_scale)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return self.transform(face), torch.tensor(float(r.label))


def balanced_sampler(labels: pd.Series) -> WeightedRandomSampler:
    """Draw live and spoof equally often; CelebA-Spoof has roughly 3 spoofs per live image."""
    counts = labels.value_counts()
    weights = labels.map(lambda y: 1.0 / counts[y]).to_numpy()
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels))


def make_loader(df: pd.DataFrame, data_root: Path, size: int, crop_scale: float, batch_size: int,
                num_workers: int, train: bool) -> DataLoader:
    ds = LivenessDataset(df, data_root, size, crop_scale, train)
    return DataLoader(ds, batch_size=batch_size, sampler=balanced_sampler(ds.df.label) if train else None,
                      shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=train and len(ds) > batch_size)
