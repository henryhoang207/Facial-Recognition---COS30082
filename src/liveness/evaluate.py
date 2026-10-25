"""Evaluate a liveness checkpoint on the test split and add it to the comparison table.

Usage (from the repo root):
    python -m src.liveness.evaluate --checkpoint DRIVE/checkpoints/liveness_mobilenetv3_2026-11-01.pt \\
        --manifest DATA/liveness/celeba.csv DATA/liveness/recorded.csv --data-root DATA

Uses the threshold stored in the checkpoint (picked on val), never one tuned on test.
Writes to --fig-dir:
    liveness_<model>_roc.png, liveness_<model>_scores.png, liveness_<model>_test.json
    liveness_results.csv / .md   one row per model x test source; re-running replaces the model's rows
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .data import load_manifests, make_loader
from .metrics import summarize
from .models import build_model
from .train import predict_scores

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLE_COLUMNS = ["model", "test_set", "n_live", "n_spoof", "acer", "apcer", "bpcer", "accuracy", "auc", "eer",
                 "bpcer_at_apcer_5", "worst_attack", "checkpoint"]


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], pretrained=False, freeze_blocks=0, img_size=cfg["img_size"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt["threshold"]


def plot_roc(curves: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (labels, scores) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr * 100, (1 - tpr) * 100, label=name)
    ax.set(xlabel="APCER (%)", ylabel="BPCER (%)", xscale="symlog", yscale="symlog",
           xlim=(0, 100), ylim=(0, 100), title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(df: pd.DataFrame, threshold: float, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for attack, group in df.groupby("attack"):
        ax.hist(group.score, bins=40, range=(0, 1), alpha=0.5, density=True, label=f"{attack} ({len(group)})")
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold:.2f}")
    ax.set(xlabel="score = P(live)", ylabel="density", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fig-dir", type=Path, default=Path("reports/figures"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, threshold = load_checkpoint(args.checkpoint, device)
    tag = cfg.get("tag") or cfg["model"]

    df = load_manifests(args.manifest)
    df = df[df.split == args.split].reset_index(drop=True)
    train_subjects = set(load_manifests([Path(p) for p in cfg["manifest"]]).query("split == 'train'").subject) \
        if all(Path(p).exists() for p in cfg["manifest"]) else set()
    leaked = set(df.subject) & train_subjects
    assert not leaked, f"test subjects also in training: {sorted(leaked)[:5]}"

    loader = make_loader(df, args.data_root, cfg["img_size"], cfg["crop_scale"], args.batch_size,
                         args.num_workers, train=False)
    df["score"] = predict_scores(model, loader, device)

    results, curves = {}, {}
    for name, part in [("all", df), *df.groupby("source")]:
        if name != "all" and df.source.nunique() == 1:
            continue
        results[name] = summarize(part.score, part.label, part.attack, threshold)
        curves[name] = (part.label.to_numpy(), part.score.to_numpy())

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_roc(curves, args.fig_dir / f"liveness_{tag}_roc.png", f"{tag}: APCER vs BPCER on {args.split}")
    plot_scores(df, threshold, args.fig_dir / f"liveness_{tag}_scores.png", f"{tag}: test score distribution")
    (args.fig_dir / f"liveness_{tag}_test.json").write_text(json.dumps(results, indent=2))

    rows = []
    for name, r in results.items():
        worst = max(r["apcer_per_attack"], key=r["apcer_per_attack"].get) if r["apcer_per_attack"] else ""
        rows.append({**{k: r[k] for k in TABLE_COLUMNS if k in r}, "model": tag, "test_set": name,
                     "worst_attack": worst, "checkpoint": args.checkpoint.name})
    table_path = args.fig_dir / "liveness_results.csv"
    table = pd.read_csv(table_path) if table_path.exists() else pd.DataFrame(columns=TABLE_COLUMNS)
    table = pd.concat([table[table.model != tag], pd.DataFrame(rows)], ignore_index=True)[TABLE_COLUMNS]
    table = table.sort_values(["test_set", "acer"]).reset_index(drop=True)
    table.to_csv(table_path, index=False)

    pct = ["acer", "apcer", "bpcer", "accuracy", "eer", "bpcer_at_apcer_5"]
    shown = table.copy()
    shown[pct] = shown[pct].astype(float).map(lambda v: f"{v:.2%}")
    shown["auc"] = shown["auc"].astype(float).map(lambda v: f"{v:.4f}")
    md = ["| " + " | ".join(shown.columns) + " |", "|" + " --- |" * len(shown.columns)]
    md += ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    (args.fig_dir / "liveness_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(shown.to_string(index=False))
    for name, r in results.items():
        print(f"{name}: APCER per attack " + ", ".join(f"{a} {v:.2%}" for a, v in r["apcer_per_attack"].items()))


if __name__ == "__main__":
    main()
