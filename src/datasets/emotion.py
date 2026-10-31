"""Preserve DFEW benchmark folds, with clip-level validation inside training."""

import csv
from collections import Counter, defaultdict

import duckdb
from PIL import Image

from .catalog import (
    IMAGE_EXTENSIONS,
    holdout,
    ordered_parallel,
    stable_rng,
    write_csv,
    write_json,
)
from .config import PreparationConfig


class EmotionPreparer:
    def __init__(self, config: PreparationConfig):
        self.config = config

    def prepare(self) -> dict:
        config, source = self.config, self.config.emotion
        with duckdb.connect(str(source.root / source.database), read_only=True) as con:
            catalog = con.execute(
                "SELECT fold, split, clip_id, label, emotion, rel_path_16f "
                "FROM dfew_splits ORDER BY fold, clip_id"
            ).fetchall()
            source_clip_count = con.execute(
                "SELECT count(*) FROM dfew_clips"
            ).fetchone()[0]
        labels = sorted({(row[3], row[4]) for row in catalog})
        if [label for label, _ in labels] != list(range(1, 8)):
            raise ValueError(
                "DFEW requires the seven documented nonambiguous labels 1..7"
            )
        clips = {}
        for _, _, clip_id, label, emotion, path in catalog:
            item = (label, emotion, path)
            if clip_id in clips and clips[clip_id] != item:
                raise ValueError(f"inconsistent DFEW clip metadata: {clip_id}")
            clips[clip_id] = item

        def inspect(clip_id):
            label, emotion, relative = clips[clip_id]
            folder = source.root / relative
            try:
                paths = sorted(
                    (
                        path
                        for path in folder.iterdir()
                        if path.suffix.lower() in IMAGE_EXTENSIONS
                    ),
                    key=lambda path: int(path.stem),
                )
                if len(paths) != source.expected_frames or [
                    int(path.stem) for path in paths
                ] != list(range(1, source.expected_frames + 1)):
                    raise ValueError(
                        "frame count/order differs from configured DFEW clip layout"
                    )
                with Image.open(paths[0]) as image:
                    image.verify()
                return (
                    clip_id,
                    [
                        {
                            "path": path.relative_to(source.root).as_posix(),
                            "clip_id": clip_id,
                            "frame_index": int(path.stem),
                            "label": label - 1,
                            "emotion": emotion,
                        }
                        for path in paths
                    ],
                    None,
                )
            except (OSError, ValueError) as exc:
                return (
                    clip_id,
                    [],
                    {"clip_id": clip_id, "path": relative, "reason": str(exc)},
                )

        available, frames, rejected = set(), [], []
        for clip_id, clip_frames, error in ordered_parallel(
            inspect, sorted(clips), config.workers
        ):
            if error:
                rejected.append(error)
            else:
                available.add(clip_id)
                frames.extend(clip_frames)
        if rejected:
            write_csv(
                config.output_dir / "dfew_rejected.csv",
                rejected,
                ["clip_id", "path", "reason"],
            )
            raise ValueError(
                "DFEW benchmark clips are incomplete; "
                "inspect dfew_rejected.csv before exporting folds"
            )
        fold_counts = {}
        for fold in sorted({row[0] for row in catalog}):
            selected = [row for row in catalog if row[0] == fold]
            for split in ("train", "test"):
                csv_path = (
                    source.root
                    / source.splits_dir
                    / f"{split}(single-labeled)"
                    / f"set_{fold}.csv"
                )
                with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                    official = {
                        (int(row["video_name"]), int(row["label"]))
                        for row in csv.DictReader(stream)
                    }
                database = {(row[2], row[3]) for row in selected if row[1] == split}
                if official != database:
                    raise ValueError(
                        f"DuckDB and official DFEW CSV differ for fold {fold}/{split}"
                    )
            by_label = defaultdict(list)
            seen_ids = set()
            for _, split, clip_id, label, _, _ in selected:
                if clip_id in seen_ids:
                    raise ValueError(f"DFEW clip appears twice within fold {fold}")
                seen_ids.add(clip_id)
                if split == "train":
                    by_label[label].append(clip_id)
            validation = set()
            for label, clip_ids in sorted(by_label.items()):
                _, held_out = holdout(
                    clip_ids,
                    source.val_fraction,
                    stable_rng(config.seed, f"emotion:{fold}:{label}"),
                )
                validation.update(held_out)
            rows = [
                {
                    "path": path,
                    "clip_id": clip_id,
                    "label": label - 1,
                    "emotion": emotion,
                    "split": (
                        "test"
                        if split == "test"
                        else "val" if clip_id in validation else "train"
                    ),
                    "source_split": split,
                    "num_frames": source.expected_frames,
                }
                for _, split, clip_id, label, emotion, path in selected
                if clip_id in available
            ]
            write_csv(
                config.output_dir / f"dfew_fold{fold}.csv",
                rows,
                [
                    "path",
                    "clip_id",
                    "label",
                    "emotion",
                    "split",
                    "source_split",
                    "num_frames",
                ],
            )
            fold_counts[fold] = dict(Counter(row["split"] for row in rows))
        write_csv(
            config.output_dir / "dfew_frames.csv",
            frames,
            ["path", "clip_id", "frame_index", "label", "emotion"],
        )
        write_json(
            config.output_dir / "dfew_class_to_idx.json",
            {emotion: label - 1 for label, emotion in labels},
        )
        summary = {
            "annotated_source_clips": source_clip_count,
            "benchmark_clips": len(available),
            "exported_frames": len(frames),
            "folds": fold_counts,
            "policy": (
                "official test folds retained; validation stratified by emotion "
                "and held out by clip; subject/session metadata unavailable"
            ),
        }
        write_json(config.output_dir / "emotion_summary.json", summary)
        return summary
