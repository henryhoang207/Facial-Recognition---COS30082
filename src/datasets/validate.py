"""Audit generated manifests independently of the builders."""

import csv
import json
from collections import Counter, defaultdict

from .catalog import write_json
from .config import PreparationConfig
from .prepare import load_config


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def disjoint_groups(records: list[dict], key: str) -> None:
    assigned = {}
    for record in records:
        group, split = record[key], record["split"]
        require(
            group not in assigned or assigned[group] == split,
            f"{key} leaks across splits: {group}",
        )
        assigned[group] = split


class ManifestValidator:
    def __init__(self, config: PreparationConfig):
        self.config = config
        self.directory = config.output_dir

    def recognition(self) -> dict:
        classification = rows(self.directory / "vggface2_classification.csv")
        mapping = json.loads(
            (self.directory / "vggface2_class_to_idx.json").read_text()
        )
        require(
            set(mapping.values()) == set(range(len(mapping))),
            "class indices must be contiguous",
        )
        require(
            len({row["path"] for row in classification}) == len(classification),
            "duplicate classification path",
        )
        for split in ("train", "val", "test"):
            require(
                {row["identity"] for row in classification if row["split"] == split}
                == set(mapping),
                f"classification {split} class set differs",
            )
        for row in classification:
            require(
                row["split"] in {"train", "val", "test"}, "invalid classification split"
            )
            require(
                int(row["label"]) == mapping[row["identity"]],
                "classification label differs from class mapping",
            )
        disjoint_groups(classification, "sha256")
        disjoint_groups(classification, "group")
        verification = rows(self.directory / "vggface2_verification.csv")
        disjoint_groups(verification, "identity")
        disjoint_groups(verification, "sha256")
        require(
            not set(mapping) & {row["identity"] for row in verification},
            "verification identities were used for classification",
        )
        require(
            not {row["sha256"] for row in classification}
            & {row["sha256"] for row in verification},
            "verification contains bytes used in classification",
        )
        catalog = {row["path"]: row for row in verification}
        pairs_summary = {}
        for split in ("val", "test"):
            pairs = rows(self.directory / f"verification_pairs_{split}.csv")
            seen = set()
            for pair in pairs:
                first, second = catalog[pair["path1"]], catalog[pair["path2"]]
                require(
                    first["split"] == second["split"] == split,
                    "pair contains an image from another verification split",
                )
                require(
                    first["sha256"] != second["sha256"],
                    "pair uses identical image bytes",
                )
                require(
                    int(pair["label"]) == int(first["identity"] == second["identity"]),
                    "pair label disagrees with identity mapping",
                )
                key = tuple(sorted((pair["path1"], pair["path2"])))
                require(key not in seen, "duplicate verification pair")
                seen.add(key)
            counts = dict(Counter(pair["label"] for pair in pairs))
            require(
                set(counts) == {"0", "1"},
                "verification requires positive and negative pairs",
            )
            pairs_summary[split] = counts
        summary = {
            "classification_rows": len(classification),
            "classes": len(mapping),
            "pairs": pairs_summary,
        }
        box_file = self.directory / "bbox_manifest.csv"
        if box_file.exists():
            box_rows = rows(box_file)
            boxes = {row["path"]: row for row in box_rows}
            require(len(boxes) == len(box_rows), "duplicate bbox path")
            keys = ("x1", "y1", "x2", "y2")
            for row in box_rows:
                x1, y1, x2, y2 = (int(row[key]) for key in keys)
                require(
                    0 <= x1 < x2 <= int(row["width"])
                    and 0 <= y1 < y2 <= int(row["height"]),
                    f"bbox outside original image: {row['path']}",
                )
            for row in classification + verification:
                require(row["path"] in boxes, "recognition image has no bbox")
                require(
                    [int(row[key]) for key in keys]
                    == [int(boxes[row["path"]][key]) for key in keys],
                    "recognition and verification bbox catalogs disagree",
                )
            summary["bbox_rows"] = len(box_rows)
        return summary

    def liveness(self) -> dict:
        records = rows(self.directory / "celeba_liveness.csv")
        disjoint_groups(records, "subject")
        require(
            len({row["path"] for row in records}) == len(records),
            "duplicate liveness path",
        )
        for row in records:
            require(
                int(row["label"]) == int(row["attack"] == "live"),
                "live/spoof label is inverted",
            )
            require(
                int(row["x2"]) > int(row["x1"]) and int(row["y2"]) > int(row["y1"]),
                "invalid liveness bbox",
            )
        for split in ("train", "val", "test"):
            require(
                {row["label"] for row in records if row["split"] == split}
                == {"0", "1"},
                f"liveness {split} is missing a class",
            )
        return {
            "rows": len(records),
            "splits": dict(Counter(row["split"] for row in records)),
        }

    def emotion(self) -> dict:
        frames = rows(self.directory / "dfew_frames.csv")
        by_clip = defaultdict(list)
        for frame in frames:
            by_clip[frame["clip_id"]].append(frame)
        folds = {}
        for path in sorted(self.directory.glob("dfew_fold*.csv")):
            records = rows(path)
            require(
                len({row["clip_id"] for row in records}) == len(records),
                "DFEW clip leaks within a fold",
            )
            for record in records:
                clip_frames = by_clip[record["clip_id"]]
                require(
                    len(clip_frames) == self.config.emotion.expected_frames,
                    "DFEW frame count differs",
                )
                require(
                    {row["label"] for row in clip_frames} == {record["label"]},
                    "DFEW frame and clip labels differ",
                )
                require(
                    record["source_split"] != "test" or record["split"] == "test",
                    "official DFEW test clip moved into training",
                )
            for split in ("train", "val", "test"):
                require(
                    {int(row["label"]) for row in records if row["split"] == split}
                    == set(range(7)),
                    f"DFEW {path.name}/{split} is missing a class",
                )
            folds[path.stem] = dict(Counter(row["split"] for row in records))
        require(bool(folds), "DFEW fold manifests are missing")
        return {"frames": len(frames), "folds": folds}

    def team(self) -> dict:
        records = rows(self.directory / "team_gallery.csv")
        mapping = json.loads((self.directory / "team_person_to_idx.json").read_text())
        for record in records:
            require(record["role"] == "gallery", "team photo was assigned to training")
            require(
                mapping[record["person_id"]] == int(record["person_index"]),
                "gallery person index differs",
            )
            if record["status"] == "ready":
                area = (int(record["x2"]) - int(record["x1"])) * (
                    int(record["y2"]) - int(record["y1"])
                )
                require(
                    area
                    >= int(record["width"])
                    * int(record["height"])
                    * self.config.team.min_face_area_fraction,
                    "gallery bbox may be a small background face",
                )
                require(
                    0 <= int(record["x1"]) < int(record["x2"]) <= int(record["width"]),
                    "gallery x bbox is invalid",
                )
                require(
                    0 <= int(record["y1"]) < int(record["y2"]) <= int(record["height"]),
                    "gallery y bbox is invalid",
                )
        return {
            "images": len(records),
            "people": len(mapping),
            "status": dict(Counter(row["status"] for row in records)),
        }

    def validate(self, selected: list[str]) -> dict:
        report = {name: getattr(self, name)() for name in selected}
        write_json(self.directory / "validation_report.json", report)
        return report


def main(argv: list[str] | None = None) -> None:
    config, selected = load_config(argv)
    print(
        json.dumps(
            ManifestValidator(config).validate(selected), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
