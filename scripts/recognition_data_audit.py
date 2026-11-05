"""Read-only VGGFace2 inventory, full decode/SHA256 audit, and seeded quality EDA.

Run: python scripts/recognition_data_audit.py --workers 8
DuckDB may be installed workspace-only at .venv/research-deps; no raw files or
catalog tables are written. All output goes to reports/recognition/research.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".venv" / "research-deps"))
import duckdb
import numpy as np
import pandas as pd
from PIL import Image


def describe(values):
    series = pd.Series(values, dtype=float)
    return {str(key): float(value) for key, value in series.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).items()}


def environment():
    # Isolate the known local OpenMP conflict from read-only dataset measurements.
    import platform
    from importlib.metadata import version, PackageNotFoundError
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for package in ["duckdb", "numpy", "pandas", "Pillow", "torch", "torchvision", "nbclient"]:
        try:
            result["packages"][package] = version(package)
        except PackageNotFoundError:
            result["packages"][package] = None
    import subprocess
    child_env = os.environ.copy()
    child_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    command = [sys.executable, "-c", "import json,torch;print(json.dumps(dict(cuda_available=torch.cuda.is_available(),cuda_runtime=torch.version.cuda,devices=[dict(name=torch.cuda.get_device_name(i),total_memory_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())])))"]
    child = subprocess.run(command, env=child_env, capture_output=True, text=True)
    result["torch_probe"] = json.loads(child.stdout) if child.returncode == 0 else {"error": child.stderr}
    result["torch_probe_workaround"] = "Child process only: KMP_DUPLICATE_LIB_OK=TRUE, matching existing notebook. Rebuild a clean environment before training."
    return result


def inspect_image(task):
    root, relative, split, identity, size, do_quality = task
    record = {"rel_path": relative, "split": split, "identity_id": identity, "file_size_bytes": size,
              "sha256": None, "width": None, "height": None, "mode": None, "format": None,
              "decoded": False, "error": None, "warnings": None, "quality_sample": do_quality}
    try:
        raw = (root / relative).read_bytes()
        record["sha256"] = hashlib.sha256(raw).hexdigest()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with Image.open(io.BytesIO(raw)) as image:
                record.update(width=image.width, height=image.height, mode=image.mode, format=image.format)
                image.load()  # Decode every pixel, not only the image header.
                record["decoded"] = True
                if do_quality:
                    gray = np.asarray(image.convert("L"), dtype=np.float32)
                    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
                    standardized = np.asarray(image.convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
                    center = standardized[1:-1, 1:-1]
                    laplace = standardized[:-2, 1:-1] + standardized[2:, 1:-1] + standardized[1:-1, :-2] + standardized[1:-1, 2:] - 4 * center
                    record.update(brightness=float(gray.mean()), contrast=float(gray.std()),
                                  red_mean=float(rgb[:, :, 0].mean()), green_mean=float(rgb[:, :, 1].mean()), blue_mean=float(rgb[:, :, 2].mean()),
                                  laplacian_variance_128=float(laplace.var()),
                                  dark_pixel_fraction=float((gray <= 10).mean()), bright_pixel_fraction=float((gray >= 245).mean()))
            if caught:
                record["warnings"] = "; ".join(str(item.message) for item in caught)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def plots(images, identities, quality, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    colors = {"train": "#246fa8", "val": "#dc862c"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for split, group in identities.groupby("split"):
        axes[0].hist(group.image_count, bins=25, alpha=.7, label=split, color=colors[split])
    axes[0].set(xlabel="Images per identity", ylabel="Identities", title="Identity distribution (full inventory)")
    axes[0].legend()
    axes[1].hist(images["width"] / images["height"], bins=50, color="#438d7d")
    axes[1].set(xlabel="Width / height", ylabel="Images", title="Aspect ratio (all images)")
    thresholds = [112, 160, 224]
    rows = [100 * ((images.width < size) | (images.height < size)).mean() for size in thresholds]
    axes[2].bar([str(size) for size in thresholds], rows, color="#9c647e")
    axes[2].set(xlabel="Target pixels", ylabel="% with either side below target", title="Resolution (all images)", ylim=(0, 100))
    for index, value in enumerate(rows):
        axes[2].text(index, value + 1, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output / "dataset_overview.png")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for split, group in quality.groupby("split"):
        for axis, column, label in zip(axes, ["brightness", "contrast", "laplacian_variance_128"], ["Grayscale mean [0,255]", "Grayscale standard deviation", "Laplacian variance at 128x128"]):
            data = np.log10(1 + group[column]) if "laplacian" in column else group[column]
            axis.hist(data, bins=30, alpha=.7, label=split, color=colors[split])
            axis.set(xlabel=("log10(1 + " + label + ")") if "laplacian" in column else label, ylabel="Sampled images")
            axis.legend()
    fig.suptitle(f"Quality indicators: seeded stratified sample, {len(quality):,} images; no automatic exclusion")
    fig.tight_layout()
    fig.savefig(output / "quality_sample.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "VGGFace2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "recognition" / "research")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality-per-identity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = args.dataset_root.resolve()
    database = root / "vggface2.duckdb"
    db_hash_before = hashlib.sha256(database.read_bytes()).hexdigest()
    with duckdb.connect(str(database), read_only=True) as connection:
        catalog = connection.execute("SELECT * FROM vggface2_images ORDER BY split, identity_id, rel_path").df()
        legacy_summary = connection.execute("SELECT * FROM v_split_summary ORDER BY split").df()
        catalog_identities = connection.execute("SELECT * FROM vggface2_identities ORDER BY split,identity_id").df()
    legacy_summary.to_csv(output / "legacy_split_summary.csv", index=False)
    catalog_identities.to_csv(output / "catalog_identity_counts.csv", index=False)
    inventory = []
    ignored = Counter()
    empty_directories = []
    rng = random.Random(args.seed)
    for split in ["train", "val"]:
        for identity_dir in sorted((root / split).iterdir()):
            if not identity_dir.is_dir():
                ignored[f"split_root:{identity_dir.suffix.lower()}"] += 1
                continue
            image_files = []
            with os.scandir(identity_dir) as entries:
                for entry in entries:
                    if entry.is_file() and Path(entry.name).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        image_files.append((entry.name, entry.stat().st_size))
                    else:
                        ignored[f"other:{Path(entry.name).suffix.lower() or '(none)'}"] += 1
            image_files.sort()
            selected = set(rng.sample(range(len(image_files)), min(args.quality_per_identity, len(image_files))))
            if not image_files:
                empty_directories.append(f"{split}/{identity_dir.name}")
            for index, (filename, size) in enumerate(image_files):
                inventory.append((root, f"{split}/{identity_dir.name}/{filename}", split, identity_dir.name, size, index in selected))
    print(f"Inventory: {len(inventory):,} images; start full decode and SHA256 with {args.workers} threads", flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(inspect_image, inventory), 1):
            records.append(record)
            if index % 10000 == 0:
                print(f"Scanned {index:,}/{len(inventory):,}; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    images = pd.DataFrame(records)
    images["aspect_ratio"] = images.width / images.height
    images["extension"] = images.rel_path.map(lambda value: Path(value).suffix.lower())
    identities = images.groupby(["split", "identity_id"]).agg(image_count=("rel_path", "size"), total_bytes=("file_size_bytes", "sum")).reset_index()
    identities.to_csv(output / "identity_counts.csv", index=False)
    images.to_csv(output / "image_audit.csv", index=False)
    quality = images[images.quality_sample & images.decoded].copy()
    quality.to_csv(output / "quality_sample.csv", index=False)
    images[~images.decoded].to_csv(output / "decode_errors.csv", index=False)
    catalog_paths, disk_paths = set(catalog.rel_path), set(images.rel_path)
    missing, unindexed = sorted(catalog_paths-disk_paths), sorted(disk_paths-catalog_paths)
    pd.DataFrame({"rel_path": missing}).to_csv(output / "catalog_missing_files.csv", index=False)
    pd.DataFrame({"rel_path": unindexed}).to_csv(output / "unindexed_files.csv", index=False)
    joined = catalog[["rel_path", "file_size_bytes"]].merge(images[["rel_path", "file_size_bytes"]], on="rel_path", suffixes=("_catalog", "_disk"))
    mismatched_sizes = joined[joined.file_size_bytes_catalog != joined.file_size_bytes_disk]
    mismatched_sizes.to_csv(output / "catalog_size_mismatches.csv", index=False)
    duplicate_rows = images[images.sha256.notna() & images.sha256.duplicated(keep=False)].sort_values(["sha256", "rel_path"])
    duplicate_rows.to_csv(output / "exact_duplicate_files.csv", index=False)
    duplicate_groups = []
    for digest, group in duplicate_rows.groupby("sha256"):
        duplicate_groups.append({"sha256": digest, "count": len(group), "cross_split": group.split.nunique() > 1,
                                 "cross_identity": group.identity_id.nunique() > 1, "paths": group.rel_path.tolist()})
    (output / "exact_duplicate_groups.json").write_text(json.dumps(duplicate_groups, indent=2), encoding="utf-8")
    train_ids = set(identities.loc[identities.split == "train", "identity_id"])
    val_ids = set(identities.loc[identities.split == "val", "identity_id"])
    split_summary = {}
    for split, group in identities.groupby("split"):
        subset = images[images.split == split]
        valid = subset[subset.decoded]
        split_summary[split] = {"identities": len(group), "images": len(subset), "bytes": int(group.total_bytes.sum()),
                                "images_per_identity": describe(group.image_count),
                                "identities_below_2_images": int((group.image_count < 2).sum()),
                                "identities_below_4_images": int((group.image_count < 4).sum()),
                                "below_resolution_fraction": {str(size): float(((valid.width < size) | (valid.height < size)).mean()) for size in [112, 160, 224]},
                                "quality_sample_size": int((quality.split == split).sum())}
    summary = {"dataset_root": str(root), "seed": args.seed, "workers": args.workers,
               "method": "Full directory inventory, SHA256 of compressed bytes, full Pillow pixel decode for every image; quality sample uniform 8 images per identity (identity-balanced, not image-weighted).",
               "split_summary": split_summary, "identity_overlap": sorted(train_ids & val_ids),
               "catalog": {"rows": len(catalog), "duplicate_rel_paths": int(catalog.rel_path.duplicated().sum()), "missing_files_count": len(missing), "unindexed_files_count": len(unindexed), "file_size_mismatch_count": len(mismatched_sizes), "sha256_before": db_hash_before, "sha256_after": hashlib.sha256(database.read_bytes()).hexdigest()},
               "files": {"total": len(images), "decoded": int(images.decoded.sum()), "decode_errors": int((~images.decoded).sum()), "image_warning_count": int(images.warnings.notna().sum()), "extensions": images.extension.value_counts().to_dict(), "modes": images.loc[images.decoded, "mode"].value_counts().to_dict(), "empty_identity_directories": empty_directories, "ignored_entries": dict(ignored)},
               "resolution": {"width": describe(images.width), "height": describe(images.height), "aspect_ratio": describe(images.aspect_ratio), "below_resolution_fraction": {str(size): float(((images.width < size) | (images.height < size)).mean()) for size in [112, 160, 224]}},
               "duplicates": {"groups": len(duplicate_groups), "files_in_groups": len(duplicate_rows), "excess_copies": sum(group["count"]-1 for group in duplicate_groups), "cross_split_groups": sum(group["cross_split"] for group in duplicate_groups), "cross_identity_groups": sum(group["cross_identity"] for group in duplicate_groups)},
               "quality": {"sample_images": len(quality), "identities": int(quality.identity_id.nunique()), "per_identity_target": args.quality_per_identity, "metrics": {column: describe(quality[column]) for column in ["brightness", "contrast", "red_mean", "green_mean", "blue_mean", "laplacian_variance_128", "dark_pixel_fraction", "bright_pixel_fraction"]}},
               "environment": environment(),
               "limitations": ["No perceptual or embedding-based near-duplicate search.", "No face-detector, landmark-alignment, identity-label, demographic, or pose audit; RGB headers alone do not establish aligned faces.", "Laplacian score is a relative quality indicator at standardized128x128; no universal blur threshold or automatic filtering was applied.", "Seeded quality sample is identity-balanced; its summary is not a pixel-weighted full-dataset mean.", "Catalog summary mean is legacy image-weighted; use this audit's arithmetic identity means."],
               "elapsed_seconds": round(time.perf_counter()-start, 2)}
    plots(images, identities, quality, output)
    summary["elapsed_seconds"] = round(time.perf_counter()-start, 2)
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
