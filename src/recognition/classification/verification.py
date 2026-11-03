"""Batched FP32 verification with an optional cache of unaugmented GPU crops."""

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.recognition.verification import PairEvaluator
from src.training import seed_worker

from .augmentation import FaceBatchTransform
from .cache import cache_loaders
from .config import RecognitionConfig


class VerificationDataset(Dataset):
    def __init__(self, pairs: PairEvaluator):
        self.pairs = pairs
        self.cached_images: torch.Tensor | None = None

    def __len__(self):
        return len(self.pairs.paths)

    def __getitem__(self, index):
        if self.cached_images is not None:
            return self.cached_images[index], index
        face = self.pairs.read_face(self.pairs.paths[index])
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous(), index


class TrainingVerificationEvaluator:
    """Cache images across epochs; recompute embeddings from current weights."""

    def __init__(self, config: RecognitionConfig):
        if config.verification is None:
            raise ValueError("configure verification paths before monitoring val_auc")
        self.config = config
        self.pairs = PairEvaluator(
            config.verification,
            size=config.preprocess.size,
            crop_scale=config.preprocess.crop_scale,
            progress=config.training.progress,
        )
        self.dataset = VerificationDataset(self.pairs)
        self.transform = FaceBatchTransform(config.preprocess).eval()
        self.loader = None
        self.cache_report = None

    def prepare(self, device: torch.device, *, allowed: bool = True) -> dict:
        """Reserve verification crop memory first, then budget classification caches."""
        config = self.config
        self.dataset.cached_images = None
        source = DataLoader(
            self.dataset,
            batch_size=config.verification.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.training.seed),
        )
        policy = config.model_copy(
            update={
                "data": config.data.model_copy(
                    update={"cache": config.verification.cache}
                )
            }
        )
        (self.loader,), report = cache_loaders(
            (source,), policy, device, allowed=allowed, names=("verification",)
        )
        self.transform.to(device)
        self.cache_report = {
            **report,
            "batch_size": config.verification.batch_size,
            "embedding_precision": "fp32",
            "embeddings_cached": False,
        }
        return self.cache_report

    @torch.inference_mode()
    def __call__(self, task, device):
        if self.loader is None:
            self.prepare(device)
        vectors = []
        # Evaluation matches standalone FP32 inference, independent of training AMP.
        with torch.autocast(device.type, enabled=False):
            for images, _ in tqdm(
                self.loader,
                desc="verification batches",
                leave=False,
                disable=not self.config.training.progress,
            ):
                images = self.transform(images.to(device, non_blocking=True))
                vectors.append(task.model.encoder.embed(images).cpu().numpy())
        metrics, _, _, _ = self.pairs.evaluate_embeddings(np.concatenate(vectors))
        selected = metrics[self.config.verification.selection_metric]
        return {
            "val_auc": selected["auc"],
            "val_tar": selected["tar"],
            "val_fpr": selected["fpr"],
            "val_score_threshold": selected["score_threshold"],
        }


def make_evaluator(config: RecognitionConfig):
    return TrainingVerificationEvaluator(config)
