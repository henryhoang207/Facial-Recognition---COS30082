"""Export the processed CelebA catalog to the existing liveness CSV contract."""

from collections import Counter

import duckdb

from src.liveness.manifest import CELEBA_SPOOF_TYPE, COLUMNS, read_celeba_bbox

from .catalog import holdout, ordered_parallel, stable_rng, write_csv, write_json
from .config import PreparationConfig


class LivenessPreparer:
    def __init__(self, config: PreparationConfig):
        self.config = config

    def prepare(self) -> dict:
        config, source = self.config, self.config.liveness
        with duckdb.connect(str(source.root / source.database), read_only=True) as con:
            catalog = con.execute(
                "SELECT path, subject_id, split, spoof_type, is_spoof "
                "FROM celeba_labels ORDER BY path"
            ).fetchall()
        test_subjects = {row[1] for row in catalog if row[2] == "test"}
        source_train_subjects = {row[1] for row in catalog if row[2] == "train"}
        overlap = source_train_subjects & test_subjects
        eligible, rejected = [], []
        excluded_attacks = 0
        for relative, subject, split, spoof_type, is_spoof in catalog:
            if split not in {"train", "test"} or is_spoof not in {0, 1}:
                raise ValueError("invalid CelebA source split or binary label")
            attack = CELEBA_SPOOF_TYPE[int(spoof_type)]
            if (attack == "live") != (is_spoof == 0):
                raise ValueError(f"CelebA label disagrees with attack: {relative}")
            if split == "train" and subject in test_subjects:
                rejected.append(
                    {"path": relative, "reason": "subject_also_in_official_test"}
                )
            elif attack != "live" and attack not in source.attacks:
                excluded_attacks += 1
            else:
                eligible.append((relative, subject, split, attack, 1 - is_spoof))
        training_subjects = sorted({row[1] for row in eligible if row[2] == "train"})
        _, validation_subjects = holdout(
            training_subjects,
            source.val_fraction,
            stable_rng(config.seed, "liveness-validation-subjects"),
        )
        validation_subjects = set(validation_subjects)

        def inspect(row):
            relative, subject, split, attack, label = row
            path = source.root / relative
            try:
                box = read_celeba_bbox(path)
                if box is None or box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError("missing or nonpositive face bbox")
                return {
                    "path": relative,
                    "label": label,
                    "attack": attack,
                    "subject": f"celeba_{subject}",
                    "source": "celeba_processed",
                    "split": (
                        "test"
                        if split == "test"
                        else "val" if subject in validation_subjects else "train"
                    ),
                    **dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
                }, None
            except (OSError, ValueError) as exc:
                return None, {"path": relative, "reason": str(exc)}

        rows = []
        print(
            f"liveness: validating images/bboxes for {len(eligible)} rows", flush=True
        )
        for index, (row, error) in enumerate(
            ordered_parallel(inspect, eligible, config.workers), 1
        ):
            if error is not None:
                rejected.append(error)
            else:
                rows.append(row)
            if index % 20000 == 0:
                print(f"liveness: checked {index}/{len(eligible)}", flush=True)
        subjects = {
            split: {row["subject"] for row in rows if row["split"] == split}
            for split in ("train", "val", "test")
        }
        if any(
            subjects[first] & subjects[second]
            for first, second in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise ValueError("liveness subjects leaked between splits")
        for split in subjects:
            if {row["label"] for row in rows if row["split"] == split} != {0, 1}:
                raise ValueError(
                    f"liveness {split} requires both live and spoof images"
                )
        write_csv(config.output_dir / "celeba_liveness.csv", rows, COLUMNS)
        write_csv(
            config.output_dir / "celeba_rejected.csv", rejected, ["path", "reason"]
        )
        write_json(
            config.output_dir / "celeba_label_mapping.json",
            {
                "live": 1,
                "spoof": 0,
                "spoof_type_to_attack": CELEBA_SPOOF_TYPE,
                "selected_attacks": source.attacks,
            },
        )
        summary = {
            "source_images": len(catalog),
            "exported_images": len(rows),
            "splits": dict(Counter(row["split"] for row in rows)),
            "split_attacks": dict(
                Counter(f"{row['split']}:{row['attack']}" for row in rows)
            ),
            "subject_overlap_in_source": len(overlap),
            "rejected_images": len(rejected),
            "excluded_unselected_attacks": excluded_attacks,
            "policy": (
                "official test rows retained; training rows with official test "
                "subjects excluded; validation held out by subject"
            ),
        }
        write_json(config.output_dir / "liveness_summary.json", summary)
        return summary
