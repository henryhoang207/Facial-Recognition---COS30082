"""Evaluate a liveness checkpoint on the test split and add it to the comparison table.

Usage (from the repo root):
    python -m src.liveness.evaluate --checkpoint MODEL.pt \\
        --manifest DATA/manifests/celeba_liveness.csv --data-root DATA

Uses the threshold stored in the checkpoint (picked on val), never one tuned on test.
Writes to --fig-dir:
    liveness_<model>_roc.png, liveness_<model>_scores.png, liveness_<model>_test.json
    liveness_results.csv / .md: one row per model and test source.
    Re-running replaces that model's rows.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from .checkpoint import load_checkpoint
from .data import load_manifests, make_loader
from .metrics import summarize
from .provenance import audit_images, file_sha256, verify_provenance
from .train import predict_scores

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TABLE_COLUMNS = [
    "model",
    "test_set",
    "n_live",
    "n_spoof",
    "acer",
    "apcer",
    "bpcer",
    "accuracy",
    "auc",
    "eer",
    "bpcer_at_apcer_5",
    "apcer_print",
    "apcer_replay",
    "threshold",
    "run_id",
    "worst_attack",
    "checkpoint",
]


def plot_roc(curves: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (labels, scores) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr * 100, (1 - tpr) * 100, label=name)
    ax.set(
        xlabel="APCER (%)",
        ylabel="BPCER (%)",
        xscale="symlog",
        yscale="symlog",
        xlim=(0, 100),
        ylim=(0, 100),
        title=title,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scores(df: pd.DataFrame, threshold: float, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for attack, group in df.groupby("attack"):
        ax.hist(
            group.score,
            bins=40,
            range=(0, 1),
            alpha=0.5,
            density=True,
            label=f"{attack} ({len(group)})",
        )
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold:.2f}")
    ax.set(xlabel="score = P(live)", ylabel="density", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(frame, root, size, scale, threshold, path):
    import cv2

    from src.common.face_crop import crop_face

    errors = frame[(frame.score >= threshold) != (frame.label == 1)]
    errors = errors.groupby("attack", sort=True).head(4)
    figure, axes = plt.subplots(3, 4, figsize=(10, 8), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, errors.itertuples(), strict=False):
        image = cv2.imread(str(root / row.path))
        crop = crop_face(
            image, (row.x1, row.y1, row.x2, row.y2), size=size, scale=scale
        )
        axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{row.attack}: P(live)={row.score:.3f}")
    figure.suptitle(
        "Representative errors" if len(errors) else "No errors at the frozen threshold"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fig-dir", type=Path, default=Path("reports/figures"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument(
        "--frozen", type=Path, help="optional frozen checkpoint decision JSON"
    )
    args = ap.parse_args(argv)
    if args.frozen:
        frozen = json.loads(args.frozen.read_text())
        if file_sha256(args.checkpoint) not in {
            item["sha256"] for item in frozen["candidates"]
        }:
            raise ValueError("checkpoint was not frozen before test evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, threshold = load_checkpoint(args.checkpoint, device)
    tag = cfg.get("tag") or cfg["model"]

    frame = audit_images(load_manifests(args.manifest), args.data_root)
    verify_provenance(frame, cfg.get("provenance"), args.split)
    df = frame[frame.split == args.split].reset_index(drop=True)

    loader = make_loader(
        df,
        args.data_root,
        cfg["img_size"],
        cfg["crop_scale"],
        args.batch_size,
        args.num_workers,
        train=False,
    )
    df["score"] = predict_scores(model, loader, device)

    results, curves = {}, {}
    for name, part in [("all", df), *df.groupby("source")]:
        if name != "all" and df.source.nunique() == 1:
            continue
        results[name] = summarize(part.score, part.label, part.attack, threshold)
        curves[name] = (part.label.to_numpy(), part.score.to_numpy())

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.fig_dir / f"liveness_{tag}_{args.split}_scores.csv", index=False)
    display = ConfusionMatrixDisplay.from_predictions(
        df.label,
        (df.score >= threshold).astype(int),
        labels=[0, 1],
        display_labels=["spoof", "live"],
        colorbar=False,
    )
    display.figure_.savefig(args.fig_dir / f"liveness_{tag}_confusion.png", dpi=150)
    plt.close(display.figure_)
    plot_errors(
        df,
        args.data_root,
        cfg["img_size"],
        cfg["crop_scale"],
        threshold,
        args.fig_dir / f"liveness_{tag}_errors.png",
    )
    plot_roc(
        curves,
        args.fig_dir / f"liveness_{tag}_roc.png",
        f"{tag}: APCER vs BPCER on {args.split}",
    )
    plot_scores(
        df,
        threshold,
        args.fig_dir / f"liveness_{tag}_scores.png",
        f"{tag}: test score distribution",
    )
    (args.fig_dir / f"liveness_{tag}_{args.split}.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "split": args.split,
                "operating_point": "frozen validation EER threshold",
                "roc_diagnostics": (
                    "EER and BPCER@pooled APCER5 describe the ROC; "
                    "they are not deployment calibration"
                ),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    for name, r in results.items():
        worst = (
            max(r["apcer_per_attack"], key=r["apcer_per_attack"].get)
            if r["apcer_per_attack"]
            else ""
        )
        rows.append(
            {
                **{k: r[k] for k in TABLE_COLUMNS if k in r},
                "model": tag,
                "test_set": name,
                "worst_attack": worst,
                "checkpoint": str(args.checkpoint),
                "run_id": cfg.get("run_id", "legacy"),
                "apcer_print": r["apcer_per_attack"].get("print"),
                "apcer_replay": r["apcer_per_attack"].get("replay"),
            }
        )
    table_path = args.fig_dir / "liveness_results.csv"
    table = (
        pd.read_csv(table_path)
        if table_path.exists()
        else pd.DataFrame(columns=TABLE_COLUMNS)
    )
    table = pd.concat(
        [table[table.model != tag], pd.DataFrame(rows)], ignore_index=True
    )[TABLE_COLUMNS]
    table = table.sort_values(["test_set", "acer"]).reset_index(drop=True)
    table.to_csv(table_path, index=False)

    pct = ["acer", "apcer", "bpcer", "accuracy", "eer", "bpcer_at_apcer_5"]
    shown = table.copy()
    shown[pct] = shown[pct].astype(float).map(lambda v: f"{v:.2%}")
    shown["auc"] = shown["auc"].astype(float).map(lambda v: f"{v:.4f}")
    md = ["| " + " | ".join(shown.columns) + " |", "|" + " --- |" * len(shown.columns)]
    md += [
        "| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)
    ]
    (args.fig_dir / "liveness_results.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(shown.to_string(index=False))
    for name, r in results.items():
        print(
            f"{name}: APCER per attack "
            + ", ".join(f"{a} {v:.2%}" for a, v in r["apcer_per_attack"].items())
        )


if __name__ == "__main__":
    main()
