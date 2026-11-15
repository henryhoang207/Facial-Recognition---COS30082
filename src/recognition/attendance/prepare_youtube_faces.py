"""Prepare clip-disjoint attendance attempts from the YouTube Faces NPZ catalog."""

import json
from collections import defaultdict
from itertools import batched, chain

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.common.face_crop import crop_face
from src.datasets.catalog import ordered_parallel, stable_rng
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.prepare import digest
from src.training.config import resolve_device

from .config import YouTubeFacesConfig, load_config
from .prepare import AttemptPreparer, rows


class YouTubeFacesPreparer(AttemptPreparer):
    cfg: YouTubeFacesConfig

    def sample_clips(self, catalog):
        cfg = self.cfg
        pools = {split: {} for split in ("train", "dev", "test")}
        for row in catalog:
            pools[row["split"]].setdefault(row["identity"], []).append(row)
        enrollment, attempts, selected = {}, [], defaultdict(set)
        for split, people in pools.items():
            eligible = sorted(
                name
                for name, clips in people.items()
                if any(clip["role"] == "enrollment" for clip in clips)
            )
            if len(eligible) <= cfg.gallery_size:
                raise ValueError(f"{split} has too few gallery identities")
            for name, clips in people.items():
                rng = stable_rng(cfg.training.seed, f"youtube-faces:{split}:{name}")
                windows = []
                for clip in clips:
                    path, count = clip["path"], int(clip["frame_count"])
                    if clip["role"] == "enrollment":
                        indices = np.linspace(0, count - 1, cfg.enrollment_count + 2)[
                            1:-1
                        ]
                        indices = [round(index) for index in indices]
                        selected[path].update(indices)
                        enrollment[name] = [f"{path}#{index}" for index in indices]
                        continue
                    indices = range(0, count, cfg.frame_stride)
                    windows.extend(
                        (clip, window)
                        for window in batched(indices, cfg.frame_count)
                        if len(window) == cfg.frame_count
                    )
                rng.shuffle(windows)
                alternatives = [other for other in eligible if other != name]
                for index, (clip, window) in enumerate(
                    windows[: cfg.windows_per_identity]
                ):
                    path = clip["path"]
                    selected[path].update(window)
                    for enrolled in ((False, True) if name in enrollment else (False,)):
                        gallery = rng.sample(
                            alternatives, cfg.gallery_size - int(enrolled)
                        )
                        if enrolled:
                            gallery.append(name)
                        attempts.append(
                            {
                                "attempt_id": f"{split}:{name}:{index}:{int(enrolled)}",
                                "split": split,
                                "identity": name,
                                "clip": clip["clip_id"],
                                "frame_paths": [f"{path}#{frame}" for frame in window],
                                "gallery": sorted(gallery),
                                "is_enrolled": int(enrolled),
                                "gallery_sampling": (
                                    "hard" if index % 2 == 0 else "random"
                                ),
                            }
                        )
        return pools, enrollment, attempts, selected

    def clip_crops(self, item, preprocess):
        relative, indices = item
        path = (self.cfg.root / relative).resolve(strict=True)
        path.relative_to(self.cfg.root)
        with np.load(path, allow_pickle=False) as archive:
            frames = archive["colorImages"]
            corners = archive["boundingBox"]
        crops = []
        for index in sorted(indices):
            key = f"{relative}#{index}"
            points = corners[:, :, index]
            if not np.isfinite(points).all():
                raise ValueError(f"invalid face annotation: {key}")
            left, top = np.floor(points.min(axis=0)).astype(int)
            right, bottom = np.ceil(points.max(axis=0)).astype(int)
            height, width = frames.shape[:2]
            box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"empty face crop: {key}")
            frame = np.ascontiguousarray(frames[..., index])
            crops.append(
                (key, crop_face(frame, box, preprocess.size, preprocess.crop_scale))
            )
        return crops

    def prepare(self):
        cfg = self.cfg
        if cfg.dataset_dir.exists():
            raise FileExistsError(cfg.dataset_dir)
        if cfg.catalog is None:
            raise ValueError("YouTube Faces preparation requires the clip catalog")
        catalog_meta = json.loads(
            cfg.catalog.with_name("dataset.json").read_text(encoding="utf-8")
        )
        catalog_hash = digest(cfg.catalog)
        if catalog_hash != catalog_meta["sha256"][cfg.catalog.name]:
            raise ValueError("YouTube Faces clip catalog checksum mismatch")
        pools, enrollment, attempts, selected = self.sample_clips(
            list(rows(cfg.catalog))
        )
        device = resolve_device(cfg.embedding_device)
        task, saved = load_task(cfg.encoder_checkpoint, device)
        preprocess = PreprocessConfig.model_validate(
            saved["metadata"]["preprocess_config"]
        )
        cv2.setNumThreads(1)
        clips = ordered_parallel(
            lambda item: self.clip_crops(item, preprocess),
            sorted(selected.items()),
            cfg.embedding_workers,
        )
        with tqdm(
            clips, total=len(selected), desc="YouTube Faces clips", unit="clip"
        ) as progress:
            vectors = self.embed_crops(
                task.model.encoder, preprocess, chain.from_iterable(progress), device
            )
        return self.export(
            pools,
            enrollment,
            attempts,
            vectors,
            saved["epoch"],
            {
                "source": "youtube_faces_with_facial_keypoints",
                "catalog_sha256": catalog_hash,
                "frame_stride": cfg.frame_stride,
                "crop": "annotated_bbox",
                "frame_paths_format": "relative_archive_path#zero_based_frame_index",
                "gallery_sampling": "alternating nearest and random distractors",
            },
        )


if __name__ == "__main__":
    config = load_config(__doc__)
    if not isinstance(config, YouTubeFacesConfig):
        raise SystemExit("A YouTube Faces YAML configuration is required.")
    print(YouTubeFacesPreparer(config).prepare())
