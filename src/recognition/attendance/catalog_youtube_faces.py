"""Catalog YouTube Faces With Facial Keypoints for attendance experiments."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

from src.datasets.catalog import holdout, stable_rng, write_csv, write_json
from src.recognition.preprocessing.prepare import digest

from .prepare import rows


class YouTubeFacesCatalog:
    def __init__(self, root: Path, metadata: Path):
        self.root = root.resolve(strict=True)
        self.metadata = metadata

    def read_clips(self):
        archives = {}
        for path in self.root.rglob("*.npz"):
            path.resolve(strict=True).relative_to(self.root)
            if path.stem in archives:
                raise ValueError(f"duplicate clip archive: {path.stem}")
            archives[path.stem] = path

        people = defaultdict(list)
        seen = set()
        for row in rows(self.metadata):
            clip, identity = row["videoID"], row["personName"].strip()
            if not identity or clip in seen or clip not in archives:
                raise ValueError(f"invalid, duplicate, or missing clip: {clip}")
            seen.add(clip)
            path = archives[clip]
            height, width, frames = (
                positive_integer(row[key])
                for key in ("imageHeight", "imageWidth", "videoDuration")
            )
            check_archive(path, height, width, frames)
            people[identity].append(
                {
                    "clip_id": clip,
                    "identity": identity,
                    "path": path.relative_to(self.root).as_posix(),
                    "frame_count": frames,
                    "height": height,
                    "width": width,
                }
            )
        if extra := archives.keys() - seen:
            raise ValueError(f"archives absent from metadata: {sorted(extra)[:5]}")
        return people

    def export(
        self, output: Path, seed: int, dev_fraction: float, test_fraction: float
    ):
        if output.exists():
            raise FileExistsError(output)
        if min(dev_fraction, test_fraction) <= 0 or dev_fraction + test_fraction >= 1:
            raise ValueError(
                "dev and test fractions must be positive and sum to less than one"
            )
        people = self.read_clips()
        train_dev, test = holdout(
            list(people), test_fraction, stable_rng(seed, "youtube-faces-test")
        )
        train, dev = holdout(
            train_dev,
            dev_fraction / (1 - test_fraction),
            stable_rng(seed, "youtube-faces-dev"),
        )
        splits = {"train": train, "dev": dev, "test": test}
        mapping = {identity: index for index, identity in enumerate(sorted(people))}
        records = []
        counts = {}
        for split, names in splits.items():
            roles = Counter()
            for name in names:
                clips = sorted(people[name], key=lambda row: row["clip_id"])
                enrollment = stable_rng(seed, f"youtube-faces-enroll:{name}").choice(
                    clips
                )
                for clip in clips:
                    if len(clips) == 1:
                        role = "unknown_only"
                    else:
                        role = "enrollment" if clip is enrollment else "probe"
                    roles[role] += 1
                    records.append(
                        dict(clip, identity_index=mapping[name], split=split, role=role)
                    )
            counts[split] = {
                "identities": len(names),
                "clips": sum(roles.values()),
                "roles": dict(roles),
            }

        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temporary:
            staged = Path(temporary) / "catalog"
            staged.mkdir()
            write_csv(staged / "clips.csv", records, list(records[0]))
            write_json(staged / "identity_to_idx.json", mapping)
            write_json(
                staged / "dataset.json",
                {
                    "source": "youtube_faces_with_facial_keypoints",
                    "metadata_sha256": digest(self.metadata),
                    "frame_array": "colorImages",
                    "frame_layout": "HWCT",
                    "bbox_array": "boundingBox",
                    "frame_index_base": 0,
                    "seed": seed,
                    "dev_fraction": dev_fraction,
                    "test_fraction": test_fraction,
                    "splits": counts,
                    "clips": len(records),
                    "identities": len(mapping),
                    "frames": sum(row["frame_count"] for row in records),
                    "sha256": {
                        name: digest(staged / name)
                        for name in ("clips.csv", "identity_to_idx.json")
                    },
                },
            )
            staged.rename(output)
        return output


def positive_integer(raw: str) -> int:
    number = float(raw)
    if number <= 0 or not number.is_integer():
        raise ValueError(f"expected a positive integer, got {raw!r}")
    return int(number)


def check_archive(path: Path, height: int, width: int, frames: int) -> None:
    with ZipFile(path) as archive:
        for key, expected in (
            ("colorImages", (height, width, 3, frames)),
            ("boundingBox", (4, 2, frames)),
        ):
            with archive.open(f"{key}.npy") as stream:
                version = read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY version {version}: {path}:{key}")
                expected_type = (
                    dtype == np.uint8 if key == "colorImages" else dtype.kind == "f"
                )
                if shape != expected or not expected_type:
                    raise ValueError(
                        f"unexpected array format: {path}:{key} {shape} {dtype}"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--metadata", required=True, type=Path, help="CSV relative to root"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dev-fraction", default=0.2, type=float)
    parser.add_argument("--test-fraction", default=0.2, type=float)
    args = parser.parse_args()
    catalog = YouTubeFacesCatalog(args.root, args.root / args.metadata)
    print(catalog.export(args.output, args.seed, args.dev_fraction, args.test_fraction))
