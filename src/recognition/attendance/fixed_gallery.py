"""Build and score fixed-roster attendance trials for video datasets."""

import argparse
import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from threading import local

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.catalog import ordered_parallel, stable_rng, write_csv, write_json
from src.recognition.classification.config import PreprocessConfig
from src.recognition.classification.predictor import load_task
from src.recognition.preprocessing.config import DetectorConfig
from src.recognition.preprocessing.detector import YuNetDetector, ensure_model
from src.recognition.preprocessing.prepare import digest
from src.training.config import read_yaml, resolve_device

from .config import VidTIMITConfig, YouTubeFacesConfig, read_config
from .features import FEATURES, attempt_features, gallery_template
from .model import AttendancePredictor
from .prepare import AttemptPreparer, rows
from .prepare_youtube_faces import YouTubeFacesPreparer
from .train import load_dataset, select_threshold


class FixedGalleryPreparer(AttemptPreparer):
    def embed_references(self, paths, encoder, preprocess, device):
        cfg = self.cfg
        if isinstance(cfg, YouTubeFacesConfig):
            selected = defaultdict(set)
            for reference in paths:
                path, index = reference.rsplit("#", 1)
                selected[path].add(int(index))
            reader = YouTubeFacesPreparer(cfg)
            clips = ordered_parallel(
                lambda item: reader.clip_crops(item, preprocess),
                sorted(selected.items()),
                cfg.embedding_workers,
            )
            with tqdm(
                clips, total=len(selected), desc="YouTube Faces clips"
            ) as progress:
                return self.embed_crops(
                    encoder, preprocess, chain.from_iterable(progress), device
                )
        if not isinstance(cfg, VidTIMITConfig):
            raise ValueError("fixed galleries require a video dataset config")
        detector_path = cfg.vidtimit.detector_config
        raw = read_yaml(detector_path)["detector"]
        raw["model"] = (detector_path.parent / raw["model"]).resolve()
        config = DetectorConfig.model_validate(raw)
        ensure_model(config)
        worker = local()

        def box(frame):
            if not hasattr(worker, "detector"):
                worker.detector = YuNetDetector(config)
            boxes = worker.detector.detect(frame)
            return boxes[0] if len(boxes) == 1 else None

        return self.embed(encoder, preprocess, sorted(paths), device, box)

    def build(self, gallery_count, *, splits=("dev", "test"), roster_seed=None):
        cfg = self.cfg
        directory = cfg.dataset_dir
        encoder_hash = digest(cfg.encoder_checkpoint)
        metadata, _ = load_dataset(directory, cfg, encoder_hash)
        if digest(directory / "enrollment.csv") != metadata["sha256"]["enrollment.csv"]:
            raise ValueError("enrollment manifest checksum mismatch")
        enroll = {split: defaultdict(list) for split in ("dev", "test")}
        probes = {split: defaultdict(set) for split in enroll}
        for row in rows(directory / "enrollment.csv"):
            if row["split"] in enroll:
                enroll[row["split"]][row["identity"]].append(row["path"])
        train_names = set()
        for row in rows(directory / "attempts.csv"):
            split, identity = row["split"], row["identity"]
            if split == "train":
                train_names.add(identity)
            else:
                probes[split][identity].add(tuple(json.loads(row["frame_paths"])))
        identities = [train_names, *[set(enroll[s]) | set(probes[s]) for s in enroll]]
        if len(set.union(*identities)) != sum(map(len, identities)):
            raise ValueError("identity appears in multiple splits")
        enrollment_paths = {
            p for split in enroll.values() for refs in split.values() for p in refs
        }
        probe_paths = {
            p
            for split in probes.values()
            for windows in split.values()
            for window in windows
            for p in window
        }
        if enrollment_paths & probe_paths:
            raise ValueError("enrollment and probe frames overlap")
        device = resolve_device(cfg.embedding_device)
        task, saved = load_task(cfg.encoder_checkpoint, device)
        preprocess = PreprocessConfig.model_validate(
            saved["metadata"]["preprocess_config"]
        )
        paths = enrollment_paths | probe_paths
        if set(splits) != enroll.keys():
            paths = {
                p for split in splits for refs in enroll[split].values() for p in refs
            } | {
                p
                for split in splits
                for windows in probes[split].values()
                for window in windows
                for p in window
            }
        vectors = self.embed_references(paths, task.model.encoder, preprocess, device)
        if vectors.keys() != paths:
            raise ValueError("fixed-gallery embeddings are incomplete")

        for split in splits:
            people = enroll[split]
            templates = {
                name: gallery_template(np.stack([vectors[path] for path in refs]))
                for name, refs in people.items()
                if name in probes[split]
            }
            seed = cfg.training.seed if roster_seed is None else roster_seed
            rng = stable_rng(seed, f"fixed-gallery:{metadata['source']}:{split}")
            prefix = f"{metadata['source']}:{split}"
            if roster_seed is not None:
                prefix += f":seed-{seed}"
            for index, (mode, gallery, visitors) in enumerate(
                fixed_rosters(
                    templates, probes[split], cfg.gallery_size, gallery_count, rng
                )
            ):
                matrix = np.stack([templates[name] for name in gallery])
                for identity in (*gallery, *visitors):
                    for window in sorted(probes[split][identity]):
                        winner, features = attempt_features(
                            np.stack([vectors[path] for path in window]), matrix
                        )
                        candidate = gallery[winner]
                        kind = (
                            "correct"
                            if candidate == identity
                            else ("wrong_known" if identity in gallery else "unknown")
                        )
                        yield {
                            "source": metadata["source"],
                            "split": split,
                            "gallery_id": f"{prefix}:{index}",
                            "gallery_mode": mode,
                            "gallery": json.dumps(gallery),
                            "identity": identity,
                            "frame_paths": json.dumps(window),
                            "candidate": candidate,
                            "kind": kind,
                            **dict(zip(FEATURES, map(float, features), strict=True)),
                        }


def fixed_rosters(templates, probes, size, count, rng):
    names = sorted(templates)
    if len(names) < size or len(probes) <= size:
        raise ValueError("fixed rosters require enrolled and visitor identities")
    similarity = np.stack([templates[name] for name in names])
    similarity = similarity @ similarity.T
    anchors = rng.sample(range(len(names)), len(names))
    for mode in ("random", "hard"):
        seen = set()
        for anchor in anchors:
            choices = [i for i in range(len(names)) if i != anchor]
            if mode == "random":
                others = rng.sample(choices, size - 1)
            else:
                others = sorted(choices, key=lambda i: -similarity[anchor, i])[
                    : size - 1
                ]
            gallery = tuple(sorted(names[i] for i in [anchor, *others]))
            if gallery in seen:
                continue
            seen.add(gallery)
            visitors = sorted(probes.keys() - set(gallery))
            if mode == "hard":
                visitor_indices = {name: i for i, name in enumerate(names)}
                enrolled_indices = [visitor_indices[name] for name in gallery]
                visitors.sort(
                    key=lambda name: (
                        -max(similarity[visitor_indices[name], enrolled_indices])
                        if name in visitor_indices
                        else np.inf
                    )
                )
                visitors = visitors[:size]
            else:
                visitors = rng.sample(visitors, min(size, len(visitors)))
            yield mode, gallery, visitors
            if len(seen) == count:
                break


def evaluate_rosters(records, config, checkpoints, *, calibration_modes=None):
    values = np.asarray(
        [[float(row[key]) for key in FEATURES] for row in records], dtype=np.float32
    )
    kinds, split, source, mode, galleries = (
        np.asarray([row[key] for row in records])
        for key in ("kind", "split", "source", "gallery_mode", "gallery_id")
    )
    correct = kinds == "correct"
    enrolled = kinds != "unknown"
    sources = sorted(set(source))
    dev = (split == "dev") & np.isin(mode, calibration_modes or ("random",))
    calibration_labels = correct if calibration_modes is None else enrolled
    strata = {
        (name, scenario): (source[dev] == name) & (mode[dev] == scenario)
        for name in sources
        for scenario in calibration_modes or ("random",)
    }
    if any(not (mask & ~calibration_labels[dev]).any() for mask in strata.values()):
        raise ValueError(
            "each calibration source and gallery mode needs negative trials"
        )
    cosine, consensus = (
        values[:, FEATURES.index(name)] for name in ("cosine_mean", "winner_fraction")
    )
    methods = {
        "cosine_consensus": np.where(
            consensus >= config.minimum_consensus, cosine, -2.0
        )
    }
    encoder_hash = digest(config.encoder_checkpoint)
    for path in checkpoints:
        predictor = AttendancePredictor(path, encoder_sha256=encoder_hash)
        model_values = values
        if predictor.feature_names != FEATURES:
            model_values = np.asarray(
                [
                    [
                        np.log2(len(json.loads(row["gallery"])))
                        if key == "log2_gallery_size" else float(row[key])
                        for key in predictor.feature_names
                    ]
                    for row in records
                ],
                dtype=np.float32,
            )
        with torch.inference_mode():
            methods[path.parent.name] = (
                predictor.model(torch.from_numpy(model_values)).sigmoid().numpy()
            )
    report = {}
    for name, scores in methods.items():
        scores = scores.astype(np.float64)
        threshold = select_threshold(
            calibration_labels[dev], scores[dev], config.target_false_accept, strata
        )
        accept = scores >= threshold
        if name == "cosine_consensus":
            accept &= consensus >= config.minimum_consensus
        groups = {}
        for phase in ("dev", "test"):
            for scenario in ("random", "hard"):
                for data_source in sources:
                    mask = (
                        (split == phase) & (mode == scenario) & (source == data_source)
                    )
                    if not mask.any():
                        continue
                    by_gallery = {}
                    for gallery in sorted(set(galleries[mask])):
                        member = mask & (galleries == gallery)
                        by_gallery[gallery] = outcome_counts(
                            accept, member, correct, enrolled
                        )
                    groups[f"{phase}/{scenario}/{data_source}"] = {
                        **outcome_counts(accept, mask, correct, enrolled),
                        "galleries": by_gallery,
                    }
        report[name] = {"dev_threshold": threshold, "groups": groups}
    return report


def outcome_counts(accept, mask, correct, enrolled):
    return {
        "enrolled": int((mask & enrolled).sum()),
        "unknown": int((mask & ~enrolled).sum()),
        "correct_accepted": int((mask & accept & correct).sum()),
        "wrong_enrolled_accepted": int((mask & accept & enrolled & ~correct).sum()),
        "unknown_accepted": int((mask & accept & ~enrolled).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gallery-count", type=int, default=28)
    parser.add_argument("--build-split", choices=("dev", "test"), nargs="+")
    parser.add_argument("--roster-seed", type=int)
    parser.add_argument("--calibration-modes", choices=("random", "hard"), nargs="+")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    configs = [read_config(path) for path in args.configs]
    if args.gallery_count < 1:
        parser.error("--gallery-count must be positive")
    config = configs[0]
    protocol = ("encoder_checkpoint", "frame_count", "gallery_size", "enrollment_count")
    if any(
        any(getattr(item, key) != getattr(config, key) for key in protocol)
        for item in configs
    ):
        parser.error("all sources must use the same encoder and gallery protocol")
    if args.report:
        write_json(
            args.report,
            evaluate_rosters(
                list(rows(args.manifest)),
                config,
                args.checkpoint,
                calibration_modes=args.calibration_modes,
            ),
        )
        print(args.report)
    else:
        if args.manifest.exists():
            raise FileExistsError(args.manifest)
        records = [
            row
            for cfg in configs
            for row in FixedGalleryPreparer(cfg).build(
                args.gallery_count,
                splits=tuple(dict.fromkeys(args.build_split or ("dev", "test"))),
                roster_seed=args.roster_seed,
            )
        ]
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.manifest, records, list(records[0]))
        print(f"{args.manifest}: {len(records)} attempts")


if __name__ == "__main__":
    main()
