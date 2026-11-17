"""Evaluate a mask checkpoint on the FMLD test split (and optionally your own photos).

Usage (from the repo root):
    python -m src.mask.evaluate --checkpoint model/Mask/mask_mobilenetv3s_<date>.pt
    python -m src.mask.evaluate --checkpoint ... --extra data/mask_webcam

--extra points to a folder with mask/ and no_mask/ subfolders of photos (for example
webcam snapshots of the team). Faces are found with the app's YuNet detector and
cropped exactly like in the app; the largest face per photo is used.

Writes reports/figures/mask_results.csv / .md (one row per model and test set) and
mask_<arch>_confusion.png.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import MaskFaces, make_loader, read_manifest
from .model import CLASSES, eval_transform, load_checkpoint
from .predictor import MaskPredictor
from .train import metrics, predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

YUNET = Path("app/weights/face_detection_yunet_2023mar.onnx")


def summarize(name: str, arch: str, prob: np.ndarray, target: np.ndarray) -> dict:
    predicted = np.where(prob >= 0.5, 0, 1)
    precision, recall, f1, support = precision_recall_fscore_support(
        target, predicted, labels=[0, 1], zero_division=0
    )
    row = {"model": arch, "test_set": name, "faces": int(len(target)), **metrics(prob, target)}
    if len(set(target)) < 2:
        row["macro_f1"] = float("nan")  # tập chỉ có một lớp: F1 trung bình không có nghĩa
    for i, label in enumerate(CLASSES):
        row[f"{label}_precision"] = float(precision[i])
        row[f"{label}_recall"] = float(recall[i])
        row[f"{label}_f1"] = float(f1[i])
    return row


def plot_confusion(prob, target, title: str, path: Path) -> None:
    matrix = confusion_matrix(target, np.where(prob >= 0.5, 0, 1), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value}", ha="center", va="center",
                color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def extra_faces(folder: Path, predictor: MaskPredictor) -> tuple[np.ndarray, np.ndarray]:
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 50)
    probs, targets = [], []
    for index, label in enumerate(CLASSES):
        for path in sorted((folder / label).glob("*")):
            image = cv2.imread(str(path))
            if image is None:
                continue
            detector.setInputSize((image.shape[1], image.shape[0]))
            _, faces = detector.detect(image)
            if faces is None:
                print(f"  no face found: {path}")
                continue
            x, y, w, h = max(faces[:, :4].tolist(), key=lambda f: f[2] * f[3])
            box = (int(x), int(y), int(x + w), int(y + h))
            probs.append(predictor.predict_frame(image, box)[1])
            targets.append(index)
    return np.array(probs), np.array(targets)


def to_markdown(table: pd.DataFrame) -> str:
    cells = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row] for row in table.values]
    lines = ["| " + " | ".join(table.columns) + " |", "|" + " --- |" * len(table.columns)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in cells]) + "\n"


def cpu_latency(checkpoint: Path, runs: int = 50) -> float:
    predictor = MaskPredictor(checkpoint, device="cpu")
    face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    for _ in range(5):
        predictor.predict(face)
    started = time.perf_counter()
    for _ in range(runs):
        predictor.predict(face)
    return (time.perf_counter() - started) / runs * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--faces", type=Path, default=Path("data/mask_faces"))
    parser.add_argument("--extra", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    arch = config["arch"]
    args.figures.mkdir(parents=True, exist_ok=True)

    test_set = MaskFaces(args.faces, "test", eval_transform(config["img_size"]), args.limit)
    loader = make_loader(test_set, 128, False, args.workers)
    prob, target = predict(model, loader, device, amp=device.type == "cuda")
    rows = [summarize("FMLD test", arch, prob, target)]
    plot_confusion(prob, target, f"{arch}, FMLD test", args.figures / f"mask_{arch}_confusion.png")

    # độ chính xác theo nguồn ảnh: MAFA (có khẩu trang + một phần không) và WIDER (không khẩu trang)
    sources = np.array([r["source"] for r in read_manifest(args.faces, "test")][: len(target)])
    for source in sorted(set(sources)):
        keep = sources == source
        rows.append(summarize(f"FMLD test, {source} only", arch, prob[keep], target[keep]))

    if args.extra:
        predictor = MaskPredictor(args.checkpoint, device=str(device))
        extra_prob, extra_target = extra_faces(args.extra, predictor)
        if len(extra_target):
            rows.append(summarize(f"own photos ({args.extra.name})", arch, extra_prob, extra_target))
            plot_confusion(extra_prob, extra_target, f"{arch}, own photos",
                           args.figures / f"mask_{arch}_confusion_own.png")

    latency = cpu_latency(args.checkpoint)
    for row in rows:
        row["cpu_ms_per_face"] = round(latency, 1)

    table = pd.DataFrame(rows)
    csv_path = args.figures / "mask_results.csv"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        table = pd.concat([old[old["model"] != arch], table], ignore_index=True)
    table.to_csv(csv_path, index=False)
    columns = ["model", "test_set", "faces", "accuracy", "macro_f1", "auc",
               "mask_recall", "no_mask_recall", "cpu_ms_per_face"]
    (args.figures / "mask_results.md").write_text(to_markdown(table[columns]), encoding="utf-8")
    print(table[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
