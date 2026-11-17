"""Fine-tune a mask / no-mask classifier and keep the checkpoint with the best val macro F1.

Usage (from the repo root, after `python -m src.mask.prepare`):
    python -m src.mask.train                       # MobileNetV3-Small, defaults sized for a 4 GB GPU
    python -m src.mask.train --arch mobilenetv3l   # MobileNetV3-Large, second candidate
    python -m src.mask.train --arch mobilenetv4m   # MobileNetV4-Conv-Medium (needs `pip install timm`)

Saves model/Mask/mask_<arch>_<date>.pt (weights + config + val metrics + history)
and reports/figures/mask_<arch>_training.png.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from tqdm import tqdm

from .data import MaskFaces, make_loader
from .model import CLASSES, MEAN, STD, build_model, eval_transform, train_transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Mặc định cho GPU 4 GB: batch 64 + mixed precision dùng khoảng 1-2 GB VRAM
DEFAULTS = {
    "mobilenetv3s": {"batch_size": 64, "lr": 1e-3},
    "mobilenetv3l": {"batch_size": 48, "lr": 5e-4},
    "mobilenetv4m": {"batch_size": 48, "lr": 5e-4},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool):
    model.eval()
    probs, targets = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=amp):
            logits = model(images)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 0].cpu())  # P(mask)
        targets.append(labels)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def metrics(prob_mask: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = np.where(prob_mask >= threshold, 0, 1)
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro")),
        # AUC chỉ tính được khi có đủ cả 2 lớp
        "auc": float(roc_auc_score(targets == 0, prob_mask)) if len(set(targets)) == 2 else float("nan"),
    }


def lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def plot_history(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    left.plot(epochs, [h["train_loss"] for h in history], marker="o")
    left.set_title("Train loss")
    left.set_xlabel("epoch")
    right.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", label="macro F1")
    right.plot(epochs, [h["val_accuracy"] for h in history], marker="o", label="accuracy")
    right.set_title("Validation")
    right.set_xlabel("epoch")
    right.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--faces", type=Path, default=Path("data/mask_faces"))
    parser.add_argument("--arch", choices=sorted(DEFAULTS), default="mobilenetv3s")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3, help="early stop after N epochs without gain")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("model/Mask"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--limit", type=int, default=None, help="use only N faces per split (quick test)")
    parser.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (tests)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    defaults = DEFAULTS[args.arch]
    batch_size = args.batch_size or defaults["batch_size"]
    lr = args.lr or defaults["lr"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {name} ({vram:.1f} GB), mixed precision on")
    else:
        print("WARNING: no CUDA GPU found, training on CPU will be slow")

    size = 224
    train_set = MaskFaces(args.faces, "train", train_transform(size), args.limit)
    val_set = MaskFaces(args.faces, "val", eval_transform(size), args.limit)
    train_loader = make_loader(train_set, batch_size, True, args.workers)
    val_loader = make_loader(val_set, batch_size * 2, False, args.workers)
    counts = np.bincount(train_set.targets, minlength=len(CLASSES)).tolist()
    print(f"train {len(train_set)} faces {dict(zip(CLASSES, counts, strict=True))}, val {len(val_set)}")

    model = build_model(args.arch, pretrained=not args.no_pretrained)
    model = model.to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_at(step, total_steps, warmup=len(train_loader) // 2)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    stamp = date.today().isoformat()
    ckpt_path = args.ckpt_dir / f"mask_{args.arch}_{stamp}.pt"
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    history, best_f1, stale = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running = time.perf_counter(), 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in bar:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.3f}")
        prob, target = predict(model, val_loader, device, amp)
        val = metrics(prob, target)
        record = {
            "epoch": epoch,
            "train_loss": running / len(train_loader),
            **{f"val_{k}": v for k, v in val.items()},
            "seconds": round(time.perf_counter() - started, 1),
        }
        history.append(record)
        peak = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if amp else ""
        print(
            f"epoch {epoch}: loss {record['train_loss']:.4f} | val acc {val['accuracy']:.4f} "
            f"F1 {val['macro_f1']:.4f} AUC {val['auc']:.4f} | {record['seconds']} s{peak}"
        )
        # vẽ lại sau mỗi epoch để bị ngắt giữa chừng vẫn còn hình
        plot_history(history, args.figures / f"mask_{args.arch}_training.png")
        if val["macro_f1"] > best_f1:
            best_f1, stale = val["macro_f1"], 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "task": "mask",
                        "arch": args.arch,
                        "classes": CLASSES,
                        "img_size": size,
                        "crop_scale": 1.3,
                        "mean": MEAN,
                        "std": STD,
                        "threshold": 0.5,
                    },
                    "val_metrics": val,
                    "epoch": epoch,
                    "history": history,
                },
                ckpt_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no val gain for {args.patience} epochs")
                break

    print(f"best val macro F1 {best_f1:.4f} -> {ckpt_path}")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
