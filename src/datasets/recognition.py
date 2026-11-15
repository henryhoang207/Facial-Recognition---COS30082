"""Closed-set classification and unseen-identity verification from VGGFace2."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb

from .catalog import FingerprintCache, holdout, stable_rng, write_csv, write_json
from .config import PreparationConfig


@dataclass(frozen=True, slots=True)
class RecognitionImage:
    path: str
    identity: str
    source_split: str
    source_index: int
    sha256: str


def classification_groups(
    images: list[RecognitionImage],
) -> list[list[RecognitionImage]]:
    """Keep equal bytes and filename prefixes together, without claiming sessions."""
    parent = list(range(len(images)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, image in enumerate(images):
        keys = [("hash", image.sha256), ("prefix", Path(image.path).stem.split("_")[0])]
        for key in keys:
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index, image in enumerate(images):
        groups[find(index)].append(image)
    return sorted(groups.values(), key=lambda group: group[0].path)


def generate_pairs(
    images: list[RecognitionImage], positive: int, negative: int, rng
) -> list[dict]:
    by_identity = defaultdict(list)
    for image in images:
        by_identity[image.identity].append(image.path)
    by_identity = {key: sorted(set(values)) for key, values in by_identity.items()}
    eligible = sorted(key for key, values in by_identity.items() if len(values) > 1)
    identities = sorted(by_identity)
    if len(identities) < 2 or not eligible:
        raise ValueError("verification needs two identities and positive pairs")
    capacities = {key: math.comb(len(by_identity[key]), 2) for key in eligible}
    if sum(capacities.values()) < positive:
        raise ValueError("not enough distinct positive pairs for requested count")
    negative_capacity = (
        sum(len(values) for values in by_identity.values()) ** 2
        - sum(len(values) ** 2 for values in by_identity.values())
    ) // 2
    if negative_capacity < negative:
        raise ValueError("not enough distinct negative pairs for requested count")
    pairs: dict[tuple[str, str], int] = {}
    positive_counts = Counter()
    while len(pairs) < positive:
        available = [key for key in eligible if positive_counts[key] < capacities[key]]
        minimum = min(positive_counts[key] for key in available)
        identity = rng.choice(
            [key for key in available if positive_counts[key] == minimum]
        )
        first, second = sorted(rng.sample(by_identity[identity], 2))
        if (first, second) not in pairs:
            pairs[(first, second)] = 1
            positive_counts[identity] += 1
    while len(pairs) < positive + negative:
        first_id, second_id = rng.sample(identities, 2)
        first, second = sorted(
            (rng.choice(by_identity[first_id]), rng.choice(by_identity[second_id]))
        )
        pairs[(first, second)] = 0
    return [
        {"path1": first, "path2": second, "label": label}
        for (first, second), label in sorted(pairs.items())
    ]


class RecognitionPreparer:
    def __init__(self, config: PreparationConfig):
        self.config = config

    def prepare(self) -> dict:
        config, source = self.config, self.config.recognition
        with duckdb.connect(str(source.root / source.database), read_only=True) as con:
            catalog = con.execute(
                "SELECT rel_path, identity_id, split, identity_idx "
                "FROM vggface2_images ORDER BY rel_path"
            ).fetchall()
        if {row[2] for row in catalog} != {"train", "val"}:
            raise ValueError(
                "VGGFace2 source must contain train and val identity partitions"
            )
        train_ids = {row[1] for row in catalog if row[2] == "train"}
        unseen_ids = {row[1] for row in catalog if row[2] == "val"}
        if train_ids & unseen_ids:
            raise ValueError("source VGGFace2 identities overlap between train and val")
        source_count = len(catalog)
        print(f"fingerprints: validating {source_count} images", flush=True)
        paths = (source.root / row[0] for row in catalog)
        fingerprints = FingerprintCache(config.output_dir).iter_inspect(
            paths, config.workers
        )
        hash_owner, conflicting = {}, set()
        by_identity = defaultdict(list)
        rejected = []
        dimensions = Counter()
        for row, entry in zip(catalog, fingerprints, strict=True):
            relative, identity, split, index = row
            if entry["error"]:
                rejected.append(
                    {"path": relative, "identity": identity, "reason": entry["error"]}
                )
            else:
                sha = entry["sha256"]
                if hash_owner.setdefault(sha, identity) != identity:
                    conflicting.add(sha)
                dimensions[f"{entry['width']}x{entry['height']}"] += 1
                by_identity[identity].append(
                    RecognitionImage(relative, identity, split, index, entry["sha256"])
                )
        del catalog, paths, fingerprints, hash_owner
        if conflicting:
            for identity, images in by_identity.items():
                retained = []
                for image in images:
                    if image.sha256 in conflicting:
                        rejected.append(
                            {
                                "path": image.path,
                                "identity": identity,
                                "reason": "identical_bytes_have_conflicting_identities",
                            }
                        )
                    else:
                        retained.append(image)
                by_identity[identity] = retained
        mapping = {identity: index for index, identity in enumerate(sorted(train_ids))}
        split_counts = Counter()

        def classification_rows():
            for identity in sorted(train_ids):
                groups = classification_groups(by_identity[identity])
                if len(groups) < 3:
                    raise ValueError(
                        f"identity {identity} has fewer than three independent groups"
                    )
                rng = stable_rng(config.seed, f"classification:{identity}")
                rng.shuffle(groups)
                n_val = max(1, round(len(groups) * source.val_fraction))
                n_test = max(1, round(len(groups) * source.test_fraction))
                if n_val + n_test >= len(groups):
                    raise ValueError(
                        f"split fractions leave no training groups for {identity}"
                    )
                rows = []
                for group_index, group in enumerate(groups):
                    split = (
                        "val"
                        if group_index < n_val
                        else "test" if group_index < n_val + n_test else "train"
                    )
                    group_name = f"{identity}:{min(item.path for item in group)}"
                    for image in group:
                        rows.append(
                            {
                                "path": image.path,
                                "identity": identity,
                                "label": mapping[identity],
                                "split": split,
                                "session": "",
                                "source_split": image.source_split,
                                "source_identity_idx": image.source_index,
                                "sha256": image.sha256,
                                "group": group_name,
                            }
                        )
                    split_counts[split] += len(group)
                yield from sorted(rows, key=lambda row: row["path"])

        verification_test_ids, verification_val_ids = holdout(
            sorted(unseen_ids),
            source.verification_val_fraction,
            stable_rng(config.seed, "verification-identities"),
        )
        if len(verification_test_ids) < 2 or len(verification_val_ids) < 2:
            raise ValueError(
                "verification val and test each need at least two identities"
            )
        verification_catalog = []
        pair_counts = {}
        for split, identities in (
            ("val", verification_val_ids),
            ("test", verification_test_ids),
        ):
            selected = []
            for identity in identities:
                unique = {}
                for image in by_identity[identity]:
                    unique.setdefault(image.sha256, image)
                selected.extend(unique.values())
            pairs = generate_pairs(
                selected,
                source.positive_pairs,
                source.negative_pairs,
                stable_rng(config.seed, f"verification-pairs:{split}"),
            )
            write_csv(
                config.output_dir / f"verification_pairs_{split}.csv",
                pairs,
                ["path1", "path2", "label"],
            )
            pair_counts[split] = {"pairs": len(pairs), "identities": len(identities)}
            verification_catalog.extend(
                {
                    "path": image.path,
                    "identity": image.identity,
                    "split": split,
                    "sha256": image.sha256,
                }
                for image in sorted(selected, key=lambda image: image.path)
            )
        write_csv(
            config.output_dir / "vggface2_classification.csv",
            classification_rows(),
            [
                "path",
                "identity",
                "label",
                "split",
                "session",
                "source_split",
                "source_identity_idx",
                "sha256",
                "group",
            ],
        )
        write_json(config.output_dir / "vggface2_class_to_idx.json", mapping)
        write_csv(
            config.output_dir / "vggface2_verification.csv",
            verification_catalog,
            ["path", "identity", "split", "sha256"],
        )
        write_csv(
            config.output_dir / "vggface2_rejected.csv",
            sorted(rejected, key=lambda row: row["path"]),
            ["path", "identity", "reason"],
        )
        summary = {
            "source_images": source_count,
            "verified_image_dimensions": dict(dimensions),
            "classification_images": sum(split_counts.values()),
            "classification_classes": len(mapping),
            "classification_splits": dict(split_counts),
            "verification_images": len(verification_catalog),
            "verification_duplicate_rows": (
                source_count
                - sum(split_counts.values())
                - len(verification_catalog)
                - len(rejected)
            ),
            "verification": pair_counts,
            "rejected_images": len(rejected),
            "conflicting_hashes": len(conflicting),
            "source_val_role": "unseen-identity verification only",
            "classification_split_unit": (
                "filename-prefix and exact-content connected groups; "
                "session metadata unavailable"
            ),
        }
        write_json(config.output_dir / "recognition_summary.json", summary)
        return summary
