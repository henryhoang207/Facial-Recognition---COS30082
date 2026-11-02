"""Optional all-or-nothing CUDA cache of unaugmented RGB uint8 face crops."""

import gc
import warnings

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.training import seed_worker

from .config import RecognitionConfig


def cache_budget(free_bytes: int, fraction: float, reserve_gib: float) -> int:
    """Keep both a free-memory fraction and a fixed training reserve available."""
    return max(
        0, min(int(free_bytes * fraction), free_bytes - int(reserve_gib * 2**30))
    )


def _allocate_and_fill(loaders, config, device):
    # This scope must unwind before OOM recovery so partial allocations are freed.
    tensors = [
        torch.empty(
            len(loader.dataset),
            3,
            config.preprocess.size,
            config.preprocess.size,
            dtype=torch.uint8,
            device=device,
        )
        for loader in loaders
    ]
    for loader, storage, split in zip(loaders, tensors, ("train", "val"), strict=True):
        source = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config.training.seed),
        )
        offset = 0
        for images, _ in tqdm(
            source,
            desc=f"cache {split} -> {device}",
            disable=not config.training.progress,
        ):
            storage[offset : offset + len(images)].copy_(images, non_blocking=True)
            offset += len(images)
    torch.cuda.synchronize(device)
    return tensors


def cache_loaders(
    loaders: tuple[DataLoader, DataLoader],
    config: RecognitionConfig,
    device: torch.device,
    *,
    allowed: bool = True,
) -> tuple[tuple[DataLoader, DataLoader], dict]:
    """Cache both active splits if they fit; otherwise keep CPU streaming loaders.

    Call after moving the model to CUDA. Test/verification images are never cached.
    The reserve covers activations, gradients, optimizer state and workspace.
    """
    required = (
        sum(len(loader.dataset) for loader in loaders) * 3 * config.preprocess.size**2
    )
    report = {
        "mode": "batch",
        "augmentation_device": str(device),
        "required_bytes": required,
        "cached_images": 0,
    }
    if not allowed or config.data.cache == "none" or device.type != "cuda":
        report["reason"] = (
            "training_cuda_oom"
            if not allowed
            else "disabled" if config.data.cache == "none" else "cpu_device"
        )
        return loaders, report
    free_bytes, _ = torch.cuda.mem_get_info(device)
    budget = cache_budget(
        free_bytes, config.data.cache_max_fraction, config.data.cache_reserve_gib
    )
    report.update(free_bytes=free_bytes, budget_bytes=budget)
    if required > budget:
        report["reason"] = "insufficient_budget"
        return loaders, report
    try:
        tensors = _allocate_and_fill(loaders, config, device)
    except torch.cuda.OutOfMemoryError:
        report["reason"] = "cache_cuda_oom"
    else:
        cached = []
        for index, (loader, tensor) in enumerate(zip(loaders, tensors, strict=True)):
            loader.dataset.cached_images = tensor
            # CUDA tensors stay in the main process; no CUDA worker subprocesses.
            cached.append(
                DataLoader(
                    loader.dataset,
                    batch_size=loader.batch_size,
                    shuffle=index == 0,
                    num_workers=0,
                    pin_memory=False,
                    generator=loader.generator,
                )
            )
        report.update(
            mode="cuda",
            reason="fits",
            cached_images=sum(len(loader.dataset) for loader in loaders),
        )
        return (cached[0], cached[1]), report
    gc.collect()
    torch.cuda.empty_cache()
    warnings.warn(
        "CUDA crop cache allocation failed; streaming batches "
        "with device augmentation.",
        stacklevel=2,
    )
    return loaders, report
