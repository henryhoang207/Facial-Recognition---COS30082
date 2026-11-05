"""Pair-based verification. Positive means same identity; higher score is better."""

import csv
import hashlib
import os
import shlex
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm.auto import tqdm

from src.common.face_crop import crop_face
from src.common.interfaces import Embedder
from src.training.config import Settings


class VerificationConfig(Settings):
    root: Path
    pairs: Path
    pairs_format: Literal["whitespace", "csv"]
    target_fpr: float = Field(ge=0, lt=1)
    selection_metric: Literal["cosine", "euclidean"]
    batch_size: int = Field(default=64, gt=0)
    cache: Literal["auto", "none"] = "auto"
    bbox_manifest: Path | None = None


def score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    target_fpr: float,
    threshold: float | None = None,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if (
        labels.ndim != 1
        or labels.shape != scores.shape
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError(
            "verification requires one score per pair and both labels 0 and 1"
        )
    if not np.isfinite(scores).all() or not 0 <= target_fpr < 1:
        raise ValueError("scores must be finite and target_fpr must be in [0,1)")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    if threshold is None:
        # The first sklearn threshold is +inf. Use a finite reject-all boundary.
        thresholds[0] = np.nextafter(scores.max(), np.inf)
        eligible = np.flatnonzero(fpr <= target_fpr)
        best_tpr = tpr[eligible].max()
        selected = eligible[tpr[eligible] == best_tpr][0]
        threshold = float(thresholds[selected])
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    accepted = scores >= threshold
    positives, negatives = labels == 1, labels == 0
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "score_threshold": threshold,
        "tar": float(accepted[positives].mean()),
        "fpr": float(accepted[negatives].mean()),
        "accuracy": float((accepted == labels).mean()),
        "positive_pairs": int(positives.sum()),
        "negative_pairs": int(negatives.sum()),
    }, (fpr, tpr)


class PairEvaluator:
    def __init__(
        self,
        config: VerificationConfig,
        *,
        size: int,
        crop_scale: float,
        progress: bool = True,
    ):
        self.config = config
        self.size = size
        self.crop_scale = crop_scale
        self.progress = progress
        with config.pairs.open("rb") as stream:
            self.pairs_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.pairs = self._read_pairs()
        self.paths = tuple(
            sorted(
                {path for first, second, _ in self.pairs for path in (first, second)}
            )
        )
        self.boxes = {}
        self.bbox_manifest_sha256 = None
        if config.bbox_manifest is not None:
            with config.bbox_manifest.open("rb") as stream:
                self.bbox_manifest_sha256 = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            self.boxes = self._read_boxes()

    def _read_boxes(self):
        wanted = set(self.paths)
        boxes = {}
        with self.config.bbox_manifest.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            keys = ("x1", "y1", "x2", "y2")
            if not {"path", *keys}.issubset(reader.fieldnames or []):
                raise ValueError("bbox manifest requires path,x1,y1,x2,y2")
            for row in reader:
                path = Path(os.path.abspath(self.config.root / row["path"]))
                if path not in wanted:
                    continue
                if path in boxes:
                    raise ValueError(f"duplicate verification bbox: {path}")
                box = tuple(int(row[key]) for key in keys)
                if not (0 <= box[0] < box[2] and 0 <= box[1] < box[3]):
                    raise ValueError(f"invalid verification bbox: {path}")
                if row.get("width") and row.get("height"):
                    if box[2] > int(row["width"]) or box[3] > int(row["height"]):
                        raise ValueError(f"verification bbox outside image: {path}")
                boxes[path] = box
        if wanted - boxes.keys():
            raise ValueError("bbox manifest must cover every verification image")
        return boxes

    def _read_pairs(self) -> list[tuple[Path, Path, int]]:
        with self.config.pairs.open(encoding="utf-8-sig", newline="") as stream:
            if self.config.pairs_format == "csv":
                reader = csv.DictReader(stream)
                if not {"path1", "path2", "label"}.issubset(reader.fieldnames or []):
                    raise ValueError("pairs CSV requires path1,path2,label columns")
                rows = [(row["path1"], row["path2"], row["label"]) for row in reader]
            else:
                rows = [
                    shlex.split(line, comments=True, posix=True)
                    for line in stream
                    if line.strip() and not line.lstrip().startswith("#")
                ]
        pairs, seen = [], set()
        for row in rows:
            if len(row) != 3 or row[2] not in {"0", "1"}:
                raise ValueError(
                    "each pair must be: relative/path1 relative/path2 label(0|1)"
                )
            first, second = [(self.config.root / value).resolve() for value in row[:2]]
            if not first.is_file() or not second.is_file():
                raise ValueError(f"pair image is missing: {first}, {second}")
            key = tuple(sorted((str(first), str(second))))
            if key in seen:
                raise ValueError(f"duplicate or reversed pair: {key}")
            seen.add(key)
            pairs.append((first, second, int(row[2])))
        if {pair[2] for pair in pairs} != {0, 1}:
            raise ValueError(
                "pairs must contain both same-person (1) and different-person (0)"
            )
        return pairs

    def read_face(self, path: Path) -> np.ndarray:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"cannot decode verification image: {path}")
        height, width = frame.shape[:2]
        box = self.boxes.get(path, (0, 0, width, height))
        if box[2] > width or box[3] > height:
            raise ValueError(f"verification bbox outside decoded image: {path}")
        return crop_face(frame, box, self.size, self.crop_scale)

    def score(self, embedder: Embedder) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        vectors = (
            embedder.embed(self.read_face(path))
            for path in tqdm(
                self.paths, desc="verification embeddings", disable=not self.progress
            )
        )
        return self._score_vectors(vectors)

    def _score_vectors(self, vectors) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        embeddings = {}
        dimension = None
        for path, value in zip(self.paths, vectors, strict=True):
            vector = np.asarray(value, dtype=np.float64)
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError("embed must return a finite, nonempty 1-D vector")
            norm = np.linalg.norm(vector)
            if not np.isclose(norm, 1, atol=1e-4):
                raise ValueError("embed must return an L2-normalized vector")
            if dimension is not None and len(vector) != dimension:
                raise ValueError("embedding dimensions differ between images")
            dimension = len(vector)
            embeddings[path] = vector / norm
        cosine = np.empty(len(self.pairs), dtype=np.float64)
        euclidean = np.empty(len(self.pairs), dtype=np.float64)
        # Avoid two dense pairs-by-features matrices for the raw-pixel baseline.
        for index, (first_path, second_path, _) in enumerate(self.pairs):
            first, second = embeddings[first_path], embeddings[second_path]
            cosine[index] = np.dot(first, second).clip(-1, 1)
            euclidean[index] = -np.linalg.norm(first - second)
        return np.array([pair[2] for pair in self.pairs]), {
            "cosine": cosine,
            "euclidean": euclidean,
        }

    def evaluate(self, embedder: Embedder, *, thresholds: dict | None = None):
        labels, scores = self.score(embedder)
        return self._summarize(labels, scores, thresholds)

    def evaluate_embeddings(
        self, vectors: np.ndarray, *, thresholds: dict | None = None
    ):
        """Score fresh embeddings in self.paths order using the shared protocol."""
        if vectors.ndim != 2 or vectors.shape[0] != len(self.paths):
            raise ValueError("provide one embedding row per verification image")
        labels, scores = self._score_vectors(vectors)
        return self._summarize(labels, scores, thresholds)

    def _summarize(self, labels, scores, thresholds):
        results, curves = {}, {}
        for metric, values in scores.items():
            results[metric], curves[metric] = score_metrics(
                labels,
                values,
                target_fpr=self.config.target_fpr,
                threshold=thresholds[metric] if thresholds is not None else None,
            )
            if metric == "cosine":
                results[metric]["similarity_min"] = results[metric]["score_threshold"]
            else:
                results[metric]["distance_max"] = -results[metric]["score_threshold"]
        return results, curves, labels, scores
