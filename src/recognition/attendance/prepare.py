"""Build identity-disjoint attendance attempts from VGGFace2 still images."""

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import batched, islice
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from src.common.face_crop import crop_face
from src.datasets.catalog import (
    holdout,
    ordered_parallel,
    stable_rng,
    write_csv,
    write_json,
)
from src.recognition.classification.augmentation import FaceBatchTransform
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.prepare import digest
from src.training.config import resolve_device

from .config import AttendanceConfig, load_config
from .features import FEATURES, attempt_features, gallery_template, unit_rows


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream)


class AttemptPreparer:
    def __init__(self, config: AttendanceConfig):
        self.cfg = config

    def partition(
        self, training_identities: set[str]
    ) -> dict[str, dict[str, list[str]]]:
        if self.cfg.catalog is None or self.cfg.selection_pairs is None:
            raise ValueError(
                "VGGFace2 preparation requires catalog and selection_pairs"
            )
        used = {
            row[key]
            for row in rows(self.cfg.selection_pairs)
            for key in ("path1", "path2")
        }
        pools = {"val": defaultdict(list), "test": defaultdict(list)}
        identities, paths, hashes = {}, set(), set()
        for row in rows(self.cfg.catalog):
            identity, split, path, sha = (
                row["identity"],
                row["split"],
                row["path"],
                row["sha256"],
            )
            relative = Path(path)
            if (
                split not in pools
                or not identity
                or not relative.parts
                or relative.anchor
                or ".." in relative.parts
                or not sha
                or path in paths
                or sha in hashes
                or identity in training_identities
                or identities.setdefault(identity, split) != split
                or (path in used and split != "val")
            ):
                raise ValueError(
                    f"unsafe, duplicate, or leaking verification row: {path}"
                )
            paths.add(path)
            hashes.add(sha)
            if path not in used:
                pools[split][identity].append(path)
        if used - paths:
            raise ValueError(
                "checkpoint selection pairs must come from the verification val pool"
            )
        train, dev = holdout(
            list(pools["val"]),
            self.cfg.dev_fraction,
            stable_rng(self.cfg.training.seed, "attendance-dev"),
        )
        return {
            "train": {name: pools["val"][name] for name in train},
            "dev": {name: pools["val"][name] for name in dev},
            "test": dict(pools["test"]),
        }

    def sample(self, pools):
        enrollment, attempts = {}, []
        cfg = self.cfg
        count = cfg.enrollment_count + cfg.windows_per_identity * cfg.frame_count
        for split, people in pools.items():
            names = sorted(people)
            if len(names) <= cfg.gallery_size:
                raise ValueError(f"{split} needs more identities than the gallery size")
            for identity in names:
                rng = stable_rng(cfg.training.seed, f"attendance:{split}:{identity}")
                images = rng.sample(sorted(people[identity]), count)
                enrollment[identity] = images[: cfg.enrollment_count]
                alternatives = [name for name in names if name != identity]
                for index, window in enumerate(
                    batched(images[cfg.enrollment_count :], cfg.frame_count)
                ):
                    enrolled = index % 2 == 0
                    gallery = rng.sample(alternatives, cfg.gallery_size - int(enrolled))
                    if enrolled:
                        gallery.append(identity)
                    attempts.append(
                        {
                            "attempt_id": f"{split}:{identity}:{index}",
                            "split": split,
                            "identity": identity,
                            "frame_paths": list(window),
                            "gallery": sorted(gallery),
                            "is_enrolled": int(enrolled),
                            "gallery_sampling": (
                                "hard"
                                if split == "train" and index % 4 == 0
                                else "random"
                            ),
                        }
                    )
        return enrollment, attempts

    @torch.inference_mode()
    def embed(
        self,
        encoder,
        preprocess: PreprocessConfig,
        paths,
        device,
        face_box: Callable | None = None,
    ):
        size = preprocess.size
        cv2.setNumThreads(1)

        def decode(relative):
            path = self.cfg.root / relative
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"cannot decode {path}")
            h, w = frame.shape[:2]
            box = (0, 0, w, h) if face_box is None else face_box(frame)
            if box is None:
                return relative, None
            face = crop_face(frame, box, size, preprocess.crop_scale)
            return relative, cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        decoded = ordered_parallel(decode, paths, self.cfg.embedding_workers)
        with tqdm(
            decoded, total=len(paths), desc="Model A crops", unit="image"
        ) as crops:
            return self.embed_crops(encoder, preprocess, crops, device)

    @torch.inference_mode()
    def embed_crops(self, encoder, preprocess, crops, device):
        transform = FaceBatchTransform(preprocess).to(device).eval()
        vectors = {}
        usable = ((key, crop) for key, crop in crops if crop is not None)
        for batch in batched(usable, self.cfg.embedding_batch_size):
            names, images = zip(*batch, strict=True)
            inputs = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(device)
            vectors.update(
                zip(names, encoder.embed(transform(inputs)).cpu().numpy(), strict=True)
            )
        return vectors

    def prepare(self) -> Path:
        cfg = self.cfg
        if cfg.dataset_dir.exists():
            raise FileExistsError(f"dataset already exists: {cfg.dataset_dir}")
        device = resolve_device(cfg.embedding_device)
        task, saved = load_task(cfg.encoder_checkpoint, device)
        metadata = saved["metadata"]
        if cfg.catalog is None or cfg.selection_pairs is None:
            raise ValueError(
                "VGGFace2 preparation requires catalog and selection_pairs"
            )
        pair_digest = digest(cfg.selection_pairs)
        if (metadata.get("verification_inputs") or {}).get(
            "pairs_sha256"
        ) != pair_digest:
            raise ValueError("selection_pairs must match the Model A checkpoint")
        pools = self.partition(set(metadata["class_to_idx"]))
        enrollment, attempts = self.sample(pools)
        paths = sorted(
            {path for images in enrollment.values() for path in images}
            | {path for attempt in attempts for path in attempt["frame_paths"]}
        )
        preprocess = PreprocessConfig.model_validate(metadata["preprocess_config"])
        vectors = self.embed(task.model.encoder, preprocess, paths, device)
        return self.export(
            pools,
            enrollment,
            attempts,
            vectors,
            saved["epoch"],
            {
                "catalog_sha256": digest(cfg.catalog),
                "selection_pairs_sha256": pair_digest,
                "source": "vggface2_still_image_simulation",
            },
        )

    def export(self, pools, enrollment, attempts, vectors, encoder_epoch, source):
        cfg = self.cfg
        templates = {
            name: gallery_template(np.stack([vectors[path] for path in images]))
            for name, images in enrollment.items()
        }
        gallery_names = {
            split: sorted(set(people) & templates.keys())
            for split, people in pools.items()
        }
        gallery_templates = {
            split: np.stack([templates[name] for name in names])
            for split, names in gallery_names.items()
        }
        used_images = {path for images in enrollment.values() for path in images}
        counts = defaultdict(Counter)
        for row in attempts:
            gallery = row["gallery"]
            used_images.update(row["frame_paths"])
            frames = np.stack([vectors[path] for path in row["frame_paths"]])
            if row["gallery_sampling"] == "hard":
                split = row["split"]
                names = gallery_names[split]
                similarities = (
                    unit_rows(frames).mean(axis=0) @ gallery_templates[split].T
                )
                nearest = (
                    names[index]
                    for index in np.argsort(-similarities, kind="stable")
                    if names[index] != row["identity"]
                )
                enrolled = [row["identity"]] if row["is_enrolled"] else []
                gallery = sorted(
                    [*enrolled, *islice(nearest, cfg.gallery_size - len(enrolled))]
                )
            winner, features = attempt_features(
                frames,
                np.stack([templates[name] for name in gallery]),
            )
            candidate = gallery[winner]
            correct = candidate == row["identity"]
            kind = (
                "correct"
                if correct
                else "wrong_known" if row["is_enrolled"] else "unknown"
            )
            counts[row["split"]][kind] += 1
            row.update(candidate=candidate, label=int(correct), kind=kind)
            row.update(zip(FEATURES, map(float, features), strict=True))
            row["frame_paths"] = json.dumps(row["frame_paths"])
            row["gallery"] = json.dumps(gallery)
        for split in pools:
            values = counts[split]
            if not values["correct"] or not values["unknown"]:
                raise ValueError(
                    f"{split} must contain correct and incorrect candidates"
                )
        mapping = {
            name: index
            for index, name in enumerate(
                sorted(name for people in pools.values() for name in people)
            )
        }
        cfg.dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=cfg.dataset_dir.parent) as temporary:
            output = Path(temporary) / "dataset"
            output.mkdir()
            write_csv(output / "attempts.csv", attempts, list(attempts[0]))
            write_csv(
                output / "enrollment.csv",
                (
                    {
                        "split": split,
                        "identity": name,
                        "identity_index": mapping[name],
                        "path": path,
                    }
                    for split, people in pools.items()
                    for name in people
                    for path in enrollment.get(name, ())
                ),
                ["split", "identity", "identity_index", "path"],
            )
            write_json(output / "identity_to_idx.json", mapping)
            write_json(
                output / "dataset.json",
                {
                    "encoder_sha256": digest(cfg.encoder_checkpoint),
                    "encoder_epoch": encoder_epoch,
                    **source,
                    "feature_names": list(FEATURES),
                    "frame_count": cfg.frame_count,
                    "gallery_size": cfg.gallery_size,
                    "enrollment_count": cfg.enrollment_count,
                    "windows_per_identity": cfg.windows_per_identity,
                    "seed": cfg.training.seed,
                    "gallery_sampling": "nearest distractors for hard attempts",
                    "identities": {
                        split: len(people) for split, people in pools.items()
                    },
                    "attempt_counts": counts,
                    "unique_images": len(used_images),
                    "sha256": {
                        name: digest(output / name)
                        for name in (
                            "attempts.csv",
                            "enrollment.csv",
                            "identity_to_idx.json",
                        )
                    },
                },
            )
            output.rename(cfg.dataset_dir)
        return cfg.dataset_dir


if __name__ == "__main__":
    print(AttemptPreparer(load_config(__doc__)).prepare())
