"""Build a anti-spoofing manifest CSV from a dataset folder.

A manifest has one row per image:
    path, label, attack, subject, source, split, x1, y1, x2, y2
`path` is relative to --data-root, `label` is 1 = live, 0 = spoof, and the box is the
face in original pixel coordinates. Training and evaluation only read manifests, so a
new dataset only needs a new builder here.

Usage (from the repo root):
    python -m src.antispoofing.manifest celeba --data-root DATA --root DATA/spoof_public/CelebA_Spoof \\
        --attacks print,replay --max-per-class 20 --out DATA/antispoofing/celeba.csv
    python -m src.antispoofing.manifest folder --data-root DATA --root DATA/spoof_recorded \\
        --source recorded --test-subjects anh,duc --out DATA/antispoofing/recorded.csv
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}
COLUMNS = ["path", "label", "attack", "subject", "source", "split", "x1", "y1", "x2", "y2"]

# CelebA-Spoof label vector: index 40 = spoof type, index 43 = live (0) / spoof (1)
CELEBA_SPOOF_TYPE = {
    0: "live",
    1: "print", 2: "print", 3: "print",                # photo, poster, A4
    4: "paper_mask", 5: "paper_mask", 6: "paper_mask",  # face / upper-body / region mask
    7: "replay", 8: "replay", 9: "replay",              # PC, pad, phone
    10: "3d_mask",
}


# ---------- CelebA-Spoof ----------

def read_celeba_bbox(img_path: Path) -> tuple[int, int, int, int] | None:
    """CelebA-Spoof ships `<name>_BB.txt` = "x y w h score" on a 224 x 224 scale.

    The official README does not document this file; check a few crops in the notebook.
    """
    bb_path = img_path.with_name(img_path.stem + "_BB.txt")
    if not bb_path.exists():
        return None
    x, y, w, h = map(float, bb_path.read_text().split()[:4])
    width, height = Image.open(img_path).size
    sx, sy = width / 224, height / 224
    return int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)


def load_celeba_labels(root: Path) -> dict[str, list[int]]:
    labels = {}
    for name in ("train_label.json", "test_label.json"):
        path = root / "metas" / "intra_test" / name
        if path.exists():
            labels.update(json.loads(path.read_text()))
    return labels


def build_celeba(root: Path, attacks: set[str] | None, max_per_class: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    labels = load_celeba_labels(root)
    if attacks and not labels:
        raise SystemExit("--attacks needs metas/intra_test/*_label.json; not found under " + str(root))

    rows, mismatches, checked = [], 0, 0
    for split in ("train", "test"):
        for subject_dir in sorted((root / "Data" / split).iterdir()):
            for cls in ("live", "spoof"):
                imgs = sorted(p for p in (subject_dir / cls).glob("*") if p.suffix.lower() in IMAGE_EXTS)
                candidates = []
                for img in imgs:
                    key = img.relative_to(root).as_posix()
                    attack = cls
                    if key in labels:
                        vec = labels[key]
                        attack = CELEBA_SPOOF_TYPE[vec[40]]
                        mismatches += int(vec[43] != (cls == "spoof"))
                        checked += 1
                    if attacks and attack != "live" and attack not in attacks:
                        continue
                    candidates.append((img, attack))
                rng.shuffle(candidates)
                for img, attack in candidates[:max_per_class]:
                    box = read_celeba_bbox(img)
                    if box is None:
                        continue
                    rows.append(dict(path=img, label=int(cls == "live"), attack=attack,
                                     subject=f"celeba_{subject_dir.name}",
                                     split="train" if split == "train" else "test",
                                     box=box))
    if checked and mismatches > 0.01 * checked:
        print(f"WARNING: {mismatches}/{checked} images where label[43] disagrees with the live/spoof folder; "
              "check the label indices against the CelebA-Spoof README")
    return rows


# ---------- Generic folder (our recordings, or any pre-sorted dataset) ----------

def extract_frames(video: Path, out_dir: Path, every: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap, frames, i = cv2.VideoCapture(str(video)), [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            path = out_dir / f"{video.stem}_{i:05d}.jpg"
            cv2.imwrite(str(path), frame)
            frames.append(path)
        i += 1
    cap.release()
    return frames


_haar = None


def detect_largest_face(img_path: Path) -> tuple[int, int, int, int] | None:
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    faces = _haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(x + w), int(y + h)


def build_folder(root: Path, detect: str, frame_every: int) -> list[dict]:
    """Layout: <root>/<subject>/<class>/<image or video>; class `live` is live, any other name is an attack type."""
    rows, no_face = [], 0
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for cls_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            files = []
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    files.append(f)
                elif f.suffix.lower() in VIDEO_EXTS:
                    files += extract_frames(f, root / "_frames" / subject_dir.name / cls_dir.name, frame_every)
            for img in files:
                if detect == "none":
                    w, h = Image.open(img).size
                    box = (0, 0, w, h)
                else:
                    box = detect_largest_face(img)
                    if box is None:
                        no_face += 1
                        continue
                attack = cls_dir.name.lower()
                rows.append(dict(path=img, label=int(attack == "live"), attack=attack,
                                 subject=subject_dir.name, split=None, box=box))
    if no_face:
        print(f"skipped {no_face} images with no face found")
    return rows


# ---------- Splits and output ----------

def assign_splits(rows: list[dict], val_frac: float, test_subjects: set[str], test_frac: float, seed: int):
    """Split by subject so nobody appears in two splits. Rows that already have `test` keep it."""
    subjects = sorted({r["subject"] for r in rows if r["split"] != "test"} - test_subjects)
    random.Random(seed).shuffle(subjects)
    n_test = 0 if test_subjects or any(r["split"] == "test" for r in rows) else round(len(subjects) * test_frac)
    n_val = max(1, round(len(subjects) * val_frac)) if len(subjects) > 1 else 0
    test = set(subjects[:n_test]) | test_subjects
    val = set(subjects[n_test:n_test + n_val])
    for r in rows:
        if r["split"] == "test" or r["subject"] in test:
            r["split"] = "test"
        else:
            r["split"] = "val" if r["subject"] in val else "train"


def to_frame(rows: list[dict], data_root: Path, source: str) -> pd.DataFrame:
    records = []
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        records.append(dict(path=Path(r["path"]).resolve().relative_to(data_root.resolve()).as_posix(),
                            label=r["label"], attack=r["attack"], subject=r["subject"], source=source,
                            split=r["split"], x1=x1, y1=y1, x2=x2, y2=y2))
    return pd.DataFrame(records, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["celeba", "folder"])
    ap.add_argument("--root", type=Path, required=True, help="dataset folder")
    ap.add_argument("--data-root", type=Path, required=True, help="manifest paths are relative to this")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default=None, help="name stored in the `source` column (default: kind)")
    ap.add_argument("--attacks", default=None, help="celeba: keep only these attack types, e.g. print,replay")
    ap.add_argument("--max-per-class", type=int, default=None, help="celeba: images per subject per live/spoof")
    ap.add_argument("--detect", choices=["haar", "none"], default="haar",
                    help="folder: find faces with OpenCV Haar, or `none` if images are already face crops")
    ap.add_argument("--frame-every", type=int, default=10, help="folder: keep every Nth video frame")
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of training subjects held out for val")
    ap.add_argument("--test-frac", type=float, default=0.2, help="folder: fraction of subjects for test")
    ap.add_argument("--test-subjects", default="", help="folder: comma-separated subjects to force into test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.kind == "celeba":
        attacks = set(args.attacks.split(",")) if args.attacks else None
        rows = build_celeba(args.root, attacks, args.max_per_class, args.seed)
    else:
        rows = build_folder(args.root, args.detect, args.frame_every)
    test_subjects = {s for s in args.test_subjects.split(",") if s}
    assign_splits(rows, args.val_frac, test_subjects, args.test_frac, args.seed)

    df = to_frame(rows, args.data_root, args.source or args.kind)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(df)} images")
    print(df.groupby(["split", "attack"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
