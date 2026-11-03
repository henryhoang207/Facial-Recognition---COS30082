"""Detect faces once, retain audit data, and publish separate bbox-aware manifests."""

import argparse
import csv
import hashlib
import json
import sqlite3
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import yaml
from tqdm.auto import tqdm

from src.datasets.catalog import stable_rng, write_csv, write_json
from src.datasets.recognition import RecognitionImage, generate_pairs
from src.training.config import override_config, read_yaml

from .config import FacePreparationConfig
from .detector import YuNetDetector, ensure_model, inspect_face


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class FaceManifestPreparer:
    def __init__(
        self, config: FacePreparationConfig, *, detector_factory=YuNetDetector
    ):
        self.config = config
        self.detector_factory = detector_factory
        self.local = threading.local()
        policy = config.detector.model_dump(mode="json", exclude={"model", "url"})
        self.policy = {"version": 1, "opencv": cv2.__version__, **policy}
        self.policy_sha = hashlib.sha256(
            json.dumps(self.policy, sort_keys=True).encode()
        ).hexdigest()

    def _signature(self, row):
        path = self.config.root / row["path"]
        try:
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            stamp = (-1, -1)
        value = (self.policy_sha, row["sha256"], *stamp)
        return hashlib.sha256(repr(value).encode()).hexdigest()

    def _inspect(self, row):
        if not hasattr(self.local, "detector"):
            self.local.detector = self.detector_factory(self.config.detector)
        frame = cv2.imread(str(self.config.root / row["path"]))
        if frame is None:
            return {
                "status": "invalid_image",
                "width": 0,
                "height": 0,
                "box": None,
                "confidence": None,
                "candidates": [],
            }
        return inspect_face(
            frame, self.local.detector.candidates(frame), self.config.detector
        )

    def run(self, *, scan_limit: int | None = None) -> dict:
        config = self.config
        config.output_dir.mkdir(parents=True, exist_ok=True)
        ensure_model(config.detector)
        cv2.setNumThreads(1)
        classification, verification = read_rows(config.classification), read_rows(
            config.verification
        )
        by_path = {}
        for row in classification + verification:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("source image paths must stay within the dataset root")
            if row["path"] in by_path:
                raise ValueError(f"duplicate source image: {row['path']}")
            by_path[row["path"]] = row
        records = [by_path[path] for path in sorted(by_path)]
        if scan_limit is not None:
            records = records[:scan_limit]
        signatures = {row["path"]: self._signature(row) for row in records}
        with sqlite3.connect(config.output_dir / "detections.sqlite3") as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS detections "
                "(path TEXT PRIMARY KEY, signature TEXT NOT NULL, "
                "payload TEXT NOT NULL)"
            )
            cached = {
                path: (signature, payload)
                for path, signature, payload in connection.execute(
                    "SELECT path,signature,payload FROM detections"
                )
            }
            results, pending = {}, []
            for row in records:
                saved = cached.get(row["path"])
                if saved and saved[0] == signatures[row["path"]]:
                    results[row["path"]] = json.loads(saved[1])
                else:
                    pending.append(row)
            reused = len(results)
            print(
                json.dumps(
                    {
                        "source_images": len(by_path),
                        "scan_images": len(records),
                        "reused": reused,
                        "pending": len(pending),
                    }
                ),
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=config.workers) as pool:
                with tqdm(
                    total=len(pending),
                    desc="YuNet bbox scan",
                    mininterval=5,
                    disable=not config.progress,
                ) as progress:
                    for start in range(0, len(pending), config.chunk_size):
                        chunk = pending[start : start + config.chunk_size]
                        for row, result in zip(
                            chunk, pool.map(self._inspect, chunk), strict=True
                        ):
                            path = row["path"]
                            results[path] = result
                            connection.execute(
                                "INSERT OR REPLACE INTO detections VALUES (?,?,?)",
                                (path, signatures[path], json.dumps(result)),
                            )
                            progress.update(1)
                        connection.commit()
        report = {
            "complete": len(records) == len(by_path),
            "source_images": len(by_path),
            "scanned_images": len(records),
            "reused": reused,
            "detected_now": len(pending),
            "detector_policy": self.policy,
            "statuses": dict(Counter(item["status"] for item in results.values())),
        }
        write_json(config.output_dir / "scan_report.json", report)
        if not report["complete"]:
            return report
        self._apply_overrides(results)
        generation = self._generation()
        directory = config.output_dir / f"protocol_{generation[:12]}"
        directory.mkdir(parents=True, exist_ok=True)
        report.update(self._publish(directory, classification, verification, results))
        report["output_dir"] = str(directory)
        report["protocol_sha256"] = generation
        report["statuses_after_overrides"] = dict(
            Counter(item["status"] for item in results.values())
        )
        write_json(directory / "summary.json", report)
        write_json(directory / "resolved_config.json", config.model_dump(mode="json"))
        write_json(config.output_dir / "current.json", report)
        return report

    def _generation(self):
        config = self.config
        payload = {
            "classification": digest(config.classification),
            "verification": digest(config.verification),
            "policy": self.policy_sha,
            "overrides": digest(config.overrides) if config.overrides else None,
            "seed": config.seed,
            "positive_pairs": config.positive_pairs,
            "negative_pairs": config.negative_pairs,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _apply_overrides(self, results):
        if self.config.overrides is None:
            return
        seen = set()
        for row in read_rows(self.config.overrides):
            path = row["path"]
            if path in seen or path not in results:
                raise ValueError(f"override path must exist exactly once: {path}")
            seen.add(path)
            if row["decision"] == "reject":
                results[path] = {
                    **results[path],
                    "status": "manual_reject",
                    "box": None,
                }
            elif row["decision"] == "accept":
                box = [int(row[key]) for key in ("x1", "y1", "x2", "y2")]
                width, height = results[path]["width"], results[path]["height"]
                if not (
                    0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height
                ):
                    raise ValueError(f"override bbox outside original image: {path}")
                results[path] = {**results[path], "status": "manual_accept", "box": box}
            else:
                raise ValueError("override decision must be accept or reject")

    def _publish(self, directory, classification, verification, results):
        config = self.config

        def accepted(row):
            return results[row["path"]]["status"] in {"accepted", "manual_accept"}

        def with_box(row):
            result = results[row["path"]]
            return {
                **row,
                **dict(zip(("x1", "y1", "x2", "y2"), result["box"], strict=True)),
                "detector_status": result["status"],
                "detector_sha256": config.detector.sha256,
            }

        retained = [row for row in classification if accepted(row)]
        splits = defaultdict(set)
        for row in retained:
            splits[row["identity"]].add(row["split"])
        eligible = {
            identity
            for identity, values in splits.items()
            if values == {"train", "val", "test"}
        }
        dropped = sorted({row["identity"] for row in classification} - eligible)
        mapping = {identity: index for index, identity in enumerate(sorted(eligible))}
        if len(mapping) < 2:
            raise ValueError(
                "face filtering leaves fewer than two complete "
                "classification identities"
            )
        retained = [
            {
                **with_box(row),
                "source_label": row["label"],
                "label": mapping[row["identity"]],
            }
            for row in retained
            if row["identity"] in eligible
        ]
        verification_rows = [with_box(row) for row in verification if accepted(row)]
        pair_counts = {}
        for split in ("val", "test"):
            selected = [
                RecognitionImage(row["path"], row["identity"], "val", 0, row["sha256"])
                for row in verification_rows
                if row["split"] == split
            ]
            # One representative per hash prevents identical-content positive pairs.
            unique = {image.sha256: image for image in reversed(selected)}
            pairs = generate_pairs(
                sorted(unique.values(), key=lambda image: image.path),
                config.positive_pairs,
                config.negative_pairs,
                stable_rng(config.seed, f"face-bbox-pairs:{split}"),
            )
            write_csv(
                directory / f"verification_pairs_{split}.csv",
                pairs,
                ["path1", "path2", "label"],
            )
            pair_counts[split] = {
                "pairs": len(pairs),
                "identities": len({image.identity for image in selected}),
                "labels": dict(Counter(str(row["label"]) for row in pairs)),
            }
        box_columns = ["x1", "y1", "x2", "y2", "detector_status", "detector_sha256"]
        write_csv(
            directory / "vggface2_classification.csv",
            retained,
            list(classification[0]) + ["source_label"] + box_columns,
        )
        write_json(directory / "vggface2_class_to_idx.json", mapping)
        write_csv(
            directory / "vggface2_verification.csv",
            verification_rows,
            list(verification[0]) + box_columns,
        )
        audit = []
        for row in sorted(classification + verification, key=lambda item: item["path"]):
            result = results[row["path"]]
            box = result["box"] or ["", "", "", ""]
            audit.append(
                {
                    "path": row["path"],
                    "identity": row["identity"],
                    "split": row["split"],
                    "status": result["status"],
                    "width": result["width"],
                    "height": result["height"],
                    "confidence": result["confidence"],
                    "faces": len(result["candidates"]),
                    **dict(zip(("x1", "y1", "x2", "y2"), box, strict=True)),
                    "candidates": json.dumps(
                        result["candidates"], separators=(",", ":")
                    ),
                }
            )
        write_csv(directory / "detector_catalog.csv", audit, list(audit[0]))
        review = [
            row for row in audit if row["status"] not in {"accepted", "manual_accept"}
        ]
        write_csv(directory / "review.csv", review, list(audit[0]))
        boxes = [
            {
                **with_box(row),
                "width": results[row["path"]]["width"],
                "height": results[row["path"]]["height"],
            }
            for row in classification + verification
            if accepted(row)
        ]
        write_csv(
            directory / "bbox_manifest.csv",
            boxes,
            ["path", "identity", "split", "width", "height"] + box_columns,
        )
        return {
            "classification_images": len(retained),
            "classification_classes": len(mapping),
            "classification_splits": dict(Counter(row["split"] for row in retained)),
            "dropped_classification_identities": dropped,
            "verification_images": len(verification_rows),
            "verification": pair_counts,
            "review_images": len(review),
            "image_content_changed": False,
            "landmark_alignment_applied": False,
        }


def load_config(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.scan_limit is not None and args.scan_limit <= 0:
        parser.error("--scan-limit must be positive")
    raw = read_yaml(args.config)
    for item in args.set:
        key, separator, value = item.partition("=")
        if not separator:
            parser.error("--set expects KEY=VALUE")
        override_config(raw, key, yaml.safe_load(value))
    config = FacePreparationConfig.model_validate(raw)
    base = args.config.resolve().parent
    paths = {
        name: (base / getattr(config, name)).resolve()
        for name in ("root", "classification", "verification", "output_dir")
    }
    paths["overrides"] = (
        (base / config.overrides).resolve() if config.overrides else None
    )
    paths["detector"] = config.detector.model_copy(
        update={"model": (base / config.detector.model).resolve()}
    )
    return config.model_copy(update=paths), args.scan_limit


def main():
    config, limit = load_config()
    print(
        json.dumps(FaceManifestPreparer(config).run(scan_limit=limit), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
