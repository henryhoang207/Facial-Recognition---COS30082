"""Crop every FMLD face into a small image so training never reads the full photos.

Usage (from the repo root, takes about 10-20 minutes on a laptop CPU):
    python -m src.mask.prepare

Inputs (see data/README.md):
    data/FMLD_annotations/{train,test}/*.xml     FMLD labels (PASCAL VOC)
    data/MAFA/{train,test}-images/images/*.jpg   MAFA photos
    data/WIDER/WIDER_{train,val}/images/...      WIDER FACE photos

Output:
    data/mask_faces/{train,val,test}/{mask,no_mask}/<photo>_<i>.jpg
    data/mask_faces/manifest.csv   one row per face (split, label, path, source, photo, size)

Two classes: masked_face and incorrectly_masked_face -> "mask", unmasked_face -> "no_mask".
val is 10% of the FMLD train photos, split by photo so faces of one photo stay together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.common.face_crop import crop_face

LABELS = {
    "masked_face": "mask",
    "incorrectly_masked_face": "mask",
    "unmasked_face": "no_mask",
}
CLASSES = ["mask", "no_mask"]


@dataclass(frozen=True)
class Options:
    data_root: Path
    out_dir: Path
    size: int
    scale: float
    min_face: int
    val_fraction: float


def resolve_image(data_root: Path, database: str, xml_path_field: str) -> Path:
    """Đường dẫn trong XML là tương đối theo bộ ảnh gốc; ghép lại theo thư mục trong data/."""
    if database == "MAFA":
        folder, name = xml_path_field.split("/", 1)  # "train-images/train_0001.jpg"
        return data_root / "MAFA" / folder / "images" / name
    return data_root / "WIDER" / xml_path_field  # "WIDER_train/images/0--Parade/x.jpg"


def split_for(xml_split: str, photo: str, val_fraction: float) -> str:
    if xml_split == "test":
        return "test"
    bucket = int(hashlib.md5(photo.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def process_xml(args: tuple[Path, str, Options]) -> tuple[list[dict], str | None]:
    xml_file, xml_split, opt = args
    root = ET.parse(xml_file).getroot()
    database = root.findtext("source/database", "")
    image_path = resolve_image(opt.data_root, database, root.findtext("path", ""))
    image = cv2.imread(str(image_path))
    if image is None:
        return [], str(image_path)
    photo = xml_file.stem
    split = split_for(xml_split, photo, opt.val_fraction)
    rows = []
    for index, obj in enumerate(root.iter("object")):
        label = LABELS.get(obj.findtext("name", ""))
        box = obj.find("bndbox")
        if label is None or box is None or obj.findtext("difficult", "0") == "1":
            continue
        x1, y1, x2, y2 = (
            int(float(box.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")
        )
        side = min(x2 - x1, y2 - y1)
        if side < opt.min_face:
            continue
        face = crop_face(image, (x1, y1, x2, y2), size=opt.size, scale=opt.scale)
        relative = Path(split) / label / f"{photo}_{index}.jpg"
        target = opt.out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), face, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(
            {
                "split": split,
                "label": label,
                "path": relative.as_posix(),
                "source": database,
                "photo": photo,
                "face_px": side,
            }
        )
    return rows, None


def prepare(opt: Options, workers: int) -> dict[str, dict[str, int]]:
    jobs = [
        (xml_file, xml_split, opt)
        for xml_split in ("train", "test")
        for xml_file in sorted((opt.data_root / "FMLD_annotations" / xml_split).glob("*.xml"))
    ]
    if not jobs:
        raise SystemExit(f"no FMLD xml files under {opt.data_root / 'FMLD_annotations'}")
    opt.out_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (face_rows, missing_path) in enumerate(
            pool.map(process_xml, jobs, chunksize=64), start=1
        ):
            rows.extend(face_rows)
            if missing_path:
                missing.append(missing_path)
            if done % 2000 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} photos, {len(rows)} faces", flush=True)

    with (opt.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["split"], {c: 0 for c in CLASSES})[row["label"]] += 1
    if missing:
        print(f"WARNING: {len(missing)} photos not found, e.g. {missing[:3]}")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("data/mask_faces"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--scale", type=float, default=1.3, help="crop side / face box side")
    parser.add_argument("--min-face", type=int, default=40, help="skip faces smaller than this (px)")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    opt = Options(args.data_root, args.out, args.size, args.scale, args.min_face, args.val_fraction)
    counts = prepare(opt, args.workers)
    for split in ("train", "val", "test"):
        print(f"{split:>5}: {counts.get(split)}")


if __name__ == "__main__":
    main()
