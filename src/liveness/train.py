"""Train one liveness model and keep the checkpoint with the best val ACER.

Usage (from the repo root):
    python -m src.liveness.train --model mobilenetv3 --manifest DATA/liveness/celeba.csv DATA/liveness/recorded.csv \\
        --data-root DATA --ckpt-dir DRIVE/checkpoints --epochs 15

The checkpoint is saved as <ckpt-dir>/liveness_<model>_<date>.pt and stores the weights,
the config, and the threshold picked on val (equal error rate point). evaluate.py and
the app both use that threshold, so the test set never influences it.
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import load_manifests, make_loader
from .metrics import eer_threshold, summarize
from .models import build_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULTS = {
    "mobilenetv3": {"lr": 3e-4, "freeze_blocks": 4, "batch_size": 64},
    "deeppixbis": {"lr": 1e-4, "freeze_blocks": 0, "batch_size": 32},
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_scores(model, loader, device) -> np.ndarray:
    model.eval()
    scores = []
    for x, _ in tqdm(loader, desc="eval", leave=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        scores.append(model.score(out).float().cpu())
    return torch.cat(scores).numpy()


def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            loss = model.loss(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def plot_history(history: pd.DataFrame, path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.5))
    a.plot(history.epoch, history.train_loss, marker="o")
    a.set(xlabel="epoch", ylabel="train loss", title="Training loss")
    b.plot(history.epoch, history.val_acer * 100, marker="o", label="val ACER")
    b.plot(history.epoch, history.val_eer * 100, marker="o", label="val EER")
    b.set(xlabel="epoch", ylabel="%", title="Validation error")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(DEFAULTS), required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--fig-dir", type=Path, default=Path("reports/figures"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4, help="stop after this many epochs without val improvement")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--freeze-blocks", type=int, default=None)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--crop-scale", type=float, default=2.0,
                    help="crop side / face box side; 2.0 keeps phone edges and paper borders in view")
    ap.add_argument("--max-train", type=int, default=None, help="subsample training images for a quick run")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights")
    ap.add_argument("--tag", default=None, help="model name in file names (default: --model)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = {**DEFAULTS[args.model], **{k: v for k, v in vars(args).items() if v is not None}}
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    cfg["manifest"] = [str(p) for p in args.manifest]
    tag = cfg.get("tag") or args.model
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_manifests(args.manifest)
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=args.seed)
    print(f"device {device} | train {len(train_df)} | val {len(val_df)} | "
          f"live share train {train_df.label.mean():.2f}, val {val_df.label.mean():.2f}")
    overlap = set(train_df.subject) & set(val_df.subject)
    assert not overlap, f"subjects in both train and val: {sorted(overlap)[:5]}"

    loader_args = dict(data_root=args.data_root, size=args.img_size, crop_scale=args.crop_scale,
                       batch_size=cfg["batch_size"], num_workers=args.num_workers)
    train_loader = make_loader(train_df, train=True, **loader_args)
    val_loader = make_loader(val_df, train=False, **loader_args)

    model = build_model(args.model, pretrained=not args.no_pretrained, freeze_blocks=cfg["freeze_blocks"],
                        img_size=args.img_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{sum(p.numel() for p in params):,} trainable / {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / f"liveness_{tag}_{date.today().isoformat()}.pt"
    history, best, since_best = [], None, 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()
        scores = predict_scores(model, val_loader, device)
        threshold, _ = eer_threshold(scores, val_df.label.to_numpy())
        val = summarize(scores, val_df.label, val_df.attack, threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acer": val["acer"],
                        "val_eer": val["eer"], "val_auc": val["auc"], "threshold": threshold})
        improved = best is None or val["acer"] < best["acer"] or (val["acer"] == best["acer"] and val["auc"] > best["auc"])
        print(f"epoch {epoch:2d} | loss {train_loss:.4f} | val ACER {val['acer']:.2%} "
              f"(APCER {val['apcer']:.2%}, BPCER {val['bpcer']:.2%}) | AUC {val['auc']:.4f}"
              + (" | saved" if improved else ""))
        if improved:
            best, since_best = val, 0
            torch.save({"model": model.state_dict(), "config": cfg, "threshold": threshold, "epoch": epoch,
                        "val_metrics": json.loads(json.dumps(val))}, ckpt_path)
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"no val improvement for {args.patience} epochs, stopping")
                break

    history = pd.DataFrame(history)
    history.to_csv(args.fig_dir / f"liveness_{tag}_history.csv", index=False)
    plot_history(history, args.fig_dir / f"liveness_{tag}_training.png")
    print(f"best val ACER {best['acer']:.2%} at threshold {best['threshold']:.3f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
