"""Reproducible identity splits, verification pairs, and recognition input batches.

The original image directories and optional DuckDB catalog are read-only. Shared
manifests are versioned by source statistics, configuration, and artifact hashes.
Exact duplicates are removed logically; near-duplicate or identity-alias checks
remain a separate data review task.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import uuid
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from src.common.face_crop import crop_face

MANIFEST_VERSION = 1
IMAGE_COLUMNS = ["rel_path", "identity_id", "label", "split", "sha256"]
PAIR_COLUMNS = ["path_a", "path_b", "is_same", "identity_a", "identity_b"]
TRIPLET_COLUMNS = ["anchor", "positive", "negative"]
BBOX_COLUMNS = ["x1", "y1", "x2", "y2"]
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_ARTIFACTS = [
    "train.csv",
    "dev.csv",
    "test.csv",
    "dev_pairs.csv",
    "test_pairs.csv",
    "probe_triplets.csv",
    "train_labels.json",
]


@dataclass
class PreparedData:
    """Shared data artifacts used unchanged by all objectives and training seeds."""

    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    dev_pairs: pd.DataFrame
    test_pairs: pd.DataFrame
    probe_triplets: pd.DataFrame
    train_labels: dict[str, int]
    metadata: dict


def _derived_seed(seed: int, namespace: str) -> int:
    payload = f"{int(seed)}:{namespace}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_path(root: Path, rel_path: str) -> Path:
    """Reject absolute paths, traversal, and symlinks escaping the dataset root."""
    relative = PurePosixPath(str(rel_path).replace("\\", "/"))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or ":" in relative.parts[0]
    ):
        raise ValueError(f"Unsafe dataset relative path: {rel_path!r}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes its root: {rel_path!r}")
    return path


def _inventory(data_root: Path) -> tuple[pd.DataFrame, str]:
    records = []
    for split in ("train", "val"):
        directory = data_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected original dataset split: {directory}")
        for identity in sorted(directory.iterdir()):
            if not identity.is_dir():
                continue
            files = sorted(
                path
                for path in identity.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if not files:
                raise ValueError(f"Identity contains no supported images: {identity}")
            for path in files:
                relative = f"{split}/{identity.name}/{path.name}"
                safe = _safe_path(data_root, relative)
                stat = safe.stat()
                records.append(
                    {
                        "rel_path": relative,
                        "identity_id": identity.name,
                        "original_split": split,
                        "file_size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                )
    if not records:
        raise ValueError(f"No images found under {data_root}")
    frame = pd.DataFrame(records).sort_values("rel_path").reset_index(drop=True)
    overlap = frame.groupby("identity_id").original_split.nunique()
    if (overlap > 1).any():
        raise ValueError(
            "Identity leakage across original train/val splits: "
            f"{overlap[overlap > 1].index.tolist()[:10]}"
        )
    catalog_source = "filesystem"
    catalog_path = data_root / "vggface2.duckdb"
    if catalog_path.is_file():
        try:
            import duckdb
        except ImportError:
            catalog_source = "filesystem (optional duckdb package unavailable)"
        else:
            with duckdb.connect(str(catalog_path), read_only=True) as connection:
                catalog = connection.execute(
                    "SELECT rel_path, identity_id, split, file_size_bytes "
                    "FROM vggface2_images"
                ).df()
            catalog["rel_path"] = catalog.rel_path.astype(str).str.replace(
                "\\", "/", regex=False
            )
            if catalog.rel_path.duplicated().any():
                raise ValueError("DuckDB catalog contains duplicate relative paths.")
            for relative in catalog.rel_path:
                if not _safe_path(data_root, relative).is_file():
                    raise FileNotFoundError(
                        f"DuckDB catalog references a missing image: {relative}"
                    )
            if set(catalog.rel_path) != set(frame.rel_path):
                raise ValueError(
                    "DuckDB catalog and filesystem inventory differ; rebuild or "
                    "review the catalog before preparing manifests."
                )
            compared = frame.merge(
                catalog, on="rel_path", suffixes=("_disk", "_catalog")
            )
            mismatch = (
                (compared.identity_id_disk != compared.identity_id_catalog)
                | (compared.original_split != compared.split)
                | (compared.file_size_bytes_disk != compared.file_size_bytes_catalog)
            )
            if mismatch.any():
                raise ValueError(
                    "DuckDB catalog identity/split/file-size metadata "
                    "differs from disk."
                )
            catalog_source = "read-only DuckDB validated against filesystem"
    return frame, catalog_source


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _hash_inventory(
    frame: pd.DataFrame, data_root: Path, output_dir: Path
) -> tuple[pd.DataFrame, dict]:
    """Reuse hashes only when root, cache digest, path, size, and mtime match."""
    cache_path = output_dir / "hash_cache.csv"
    provenance_path = output_dir / "hash_cache_metadata.json"
    cached = {}
    reason = "no previous verified cache"
    if cache_path.is_file() and provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance.get("version") != MANIFEST_VERSION or provenance.get(
                "data_root"
            ) != str(data_root):
                reason = "cache root or schema changed"
            elif provenance.get("cache_sha256") != _sha256_file(cache_path):
                reason = "cache checksum mismatch"
            else:
                previous = pd.read_csv(
                    cache_path, dtype={"rel_path": str, "sha256": str}
                )
                required = {"rel_path", "file_size_bytes", "mtime_ns", "sha256"}
                if (
                    not required.issubset(previous.columns)
                    or previous.rel_path.duplicated().any()
                ):
                    reason = "invalid cache columns or duplicate paths"
                else:
                    cached = previous.set_index("rel_path").to_dict("index")
                    reason = None
        except (OSError, ValueError, KeyError, pd.errors.ParserError):
            reason = "unreadable cache provenance"
    hashes = {}
    pending = []
    for row in frame.itertuples(index=False):
        old = cached.get(row.rel_path)
        digest = str(old.get("sha256", "")) if old else ""
        valid_digest = len(digest) == 64 and all(
            character in "0123456789abcdef" for character in digest
        )
        if (
            old
            and valid_digest
            and int(old["file_size_bytes"]) == row.file_size_bytes
            and int(old["mtime_ns"]) == row.mtime_ns
        ):
            hashes[row.rel_path] = digest
        else:
            pending.append(row)
    paths = [_safe_path(data_root, row.rel_path) for row in pending]
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
        for row, path, digest in zip(
            pending, paths, executor.map(_sha256_file, paths), strict=True
        ):
            stat = path.stat()
            if stat.st_size != row.file_size_bytes or stat.st_mtime_ns != row.mtime_ns:
                raise RuntimeError(
                    f"Image changed while computing SHA256: {row.rel_path}"
                )
            hashes[row.rel_path] = digest
    result = frame.copy()
    result["sha256"] = result.rel_path.map(hashes)
    cache = result[["rel_path", "file_size_bytes", "mtime_ns", "sha256"]]
    _atomic_csv(cache_path, cache)
    _atomic_json(
        provenance_path,
        {
            "version": MANIFEST_VERSION,
            "data_root": str(data_root),
            "cache_sha256": _sha256_file(cache_path),
            "validation": "relative path + size bytes + mtime_ns",
        },
    )
    return result, {
        "reused": len(frame) - len(pending),
        "computed": len(pending),
        "cache_invalidation_reason": reason,
    }


def _make_pairs(
    frame: pd.DataFrame, genuine: int, impostor: int, seed: int
) -> pd.DataFrame:
    rng = random.Random(seed)
    groups = {
        identity: sorted(group.rel_path.tolist())
        for identity, group in frame.groupby("identity_id", sort=True)
    }
    rows = []
    for identity, paths in groups.items():
        count = len(paths)
        possible = count * (count - 1) // 2
        if genuine > possible:
            raise ValueError(
                f"Identity {identity} has only {possible} distinct genuine pairs, "
                f"but {genuine} were requested."
            )
        offsets = [index * (2 * count - index - 1) // 2 for index in range(count)]
        for ordinal in rng.sample(range(possible), genuine):
            left = bisect_right(offsets, ordinal) - 1
            right = left + 1 + ordinal - offsets[left]
            rows.append((paths[left], paths[right], 1, identity, identity))
    if impostor and len(groups) < 2:
        raise ValueError(
            "Impostor pair sampling requires at least two evaluation identities."
        )
    for identity_a, identity_b in itertools.combinations(groups, 2):
        paths_a, paths_b = groups[identity_a], groups[identity_b]
        possible = len(paths_a) * len(paths_b)
        if impostor > possible:
            raise ValueError(
                f"Identity pair {identity_a}/{identity_b} has only {possible} "
                f"unique impostor pairs; {impostor} requested."
            )
        for ordinal in rng.sample(range(possible), impostor):
            rows.append(
                (
                    paths_a[ordinal // len(paths_b)],
                    paths_b[ordinal % len(paths_b)],
                    0,
                    identity_a,
                    identity_b,
                )
            )
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def _make_probe(frame: pd.DataFrame, per_identity: int, seed: int) -> pd.DataFrame:
    groups = {
        identity: sorted(group.rel_path.tolist())
        for identity, group in frame.groupby("identity_id", sort=True)
    }
    rng = random.Random(seed)
    rows = []
    if per_identity and len(groups) < 2:
        raise ValueError(
            "A training triplet probe requires at least two train identities."
        )
    for identity, paths in groups.items():
        others = [name for name in groups if name != identity]
        possible = (
            len(paths) * (len(paths) - 1) * sum(len(groups[name]) for name in others)
        )
        if per_identity > possible:
            raise ValueError(
                f"Identity {identity} cannot supply {per_identity} "
                "unique valid probe triplets."
            )
        selected = set()
        while len(selected) < per_identity:
            anchor, positive = rng.sample(paths, 2)
            negative_identity = rng.choice(others)
            negative = rng.choice(groups[negative_identity])
            selected.add((anchor, positive, negative))
        rows.extend(sorted(selected))
    return pd.DataFrame(rows, columns=TRIPLET_COLUMNS)


def _load_prepared(output_dir: Path, metadata: dict) -> PreparedData:
    image_frames = [
        pd.read_csv(
            output_dir / f"{split}.csv",
            dtype={"rel_path": str, "identity_id": str, "sha256": str},
        )
        for split in ("train", "dev", "test")
    ]
    pair_frames = [
        pd.read_csv(
            output_dir / f"{split}_pairs.csv",
            dtype={"path_a": str, "path_b": str, "identity_a": str, "identity_b": str},
        )
        for split in ("dev", "test")
    ]
    probe = pd.read_csv(output_dir / "probe_triplets.csv", dtype=str)
    labels = json.loads((output_dir / "train_labels.json").read_text(encoding="utf-8"))
    return PreparedData(*image_frames, *pair_frames, probe, labels, metadata)


def prepare_manifests(
    data_root: Path,
    output_dir: Path,
    split_seed: int = 42,
    dev_identities: int = 48,
    genuine_per_identity: int = 200,
    impostor_per_identity_pair: int = 100,
    probe_per_identity: int = 4,
    force: bool = False,
) -> PreparedData:
    """Build or verify shared identity-disjoint manifests without modifying images.

    Original ``train`` identities are partitioned into train/dev; original ``val``
    becomes final test. Train labels are contiguous from zero, followed by disjoint
    dev/test labels. Cached manifests are rejected when roots, configuration,
    source size/mtime, or output hashes change; pass ``force=True`` to regenerate.
    SHA256 cache validation uses size and nanosecond mtime, not an adversarial
    tamper-proof guarantee when both file metadata fields are deliberately reset.
    """
    for name, value in (
        ("dev_identities", dev_identities),
        ("genuine_per_identity", genuine_per_identity),
        ("impostor_per_identity_pair", impostor_per_identity_pair),
        ("probe_per_identity", probe_per_identity),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")
    data_root = Path(data_root).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    if output_dir == data_root or output_dir.is_relative_to(data_root):
        raise ValueError("Manifest output must be outside the read-only dataset root.")
    inventory, source = _inventory(data_root)
    fingerprint = _json_digest(inventory.to_dict("records"))
    config = {
        "split_seed": int(split_seed),
        "dev_identities": dev_identities,
        "genuine_per_identity": genuine_per_identity,
        "impostor_per_identity_pair": impostor_per_identity_pair,
        "probe_per_identity": probe_per_identity,
    }
    metadata_path = output_dir / "metadata.json"
    if metadata_path.is_file() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("version") != MANIFEST_VERSION
            or metadata.get("data_root") != str(data_root)
            or metadata.get("config") != config
            or metadata.get("source_fingerprint") != fingerprint
        ):
            raise ValueError(
                "Stale recognition manifests: dataset root, source inventory, "
                "or configuration changed. Use force=True after review."
            )
        for name in _ARTIFACTS:
            path = output_dir / name
            if not path.is_file() or metadata.get("artifact_sha256", {}).get(
                name
            ) != _sha256_file(path):
                raise ValueError(
                    f"Recognition manifest is missing or modified: {name}. "
                    "Use force=True after review."
                )
        return _load_prepared(output_dir, metadata)
    if (
        not force
        and output_dir.exists()
        and any((output_dir / name).exists() for name in _ARTIFACTS)
    ):
        raise ValueError(
            "Incomplete recognition manifests without valid metadata; "
            "use force=True after review."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    hashed, hash_stats = _hash_inventory(inventory, data_root, output_dir)
    duplicates = hashed[hashed.sha256.duplicated(keep=False)]
    for digest, group in duplicates.groupby("sha256"):
        if group.identity_id.nunique() > 1 or group.original_split.nunique() > 1:
            raise ValueError(
                "Exact-duplicate leakage across identities or original splits "
                f"for SHA256 {digest}: {group.rel_path.tolist()[:6]}"
            )
    unique = hashed.drop_duplicates("sha256", keep="first").copy()
    original_train_ids = sorted(
        unique.loc[unique.original_split == "train", "identity_id"].unique()
    )
    test_ids = sorted(
        unique.loc[unique.original_split == "val", "identity_id"].unique()
    )
    if not 0 < dev_identities < len(original_train_ids):
        raise ValueError(
            f"dev_identities must be between 1 and {len(original_train_ids)-1}; "
            f"received {dev_identities}."
        )
    dev_ids = sorted(
        random.Random(int(split_seed)).sample(original_train_ids, dev_identities)
    )
    dev_set = set(dev_ids)
    train_ids = [identity for identity in original_train_ids if identity not in dev_set]
    identity_labels = {
        identity: index for index, identity in enumerate(train_ids + dev_ids + test_ids)
    }
    train_labels = {identity: identity_labels[identity] for identity in train_ids}
    frames = {}
    for split, identities in (
        ("train", train_ids),
        ("dev", dev_ids),
        ("test", test_ids),
    ):
        frame = unique[unique.identity_id.isin(identities)].copy()
        frame["label"] = frame.identity_id.map(identity_labels).astype(np.int64)
        frame["split"] = split
        frames[split] = (
            frame[IMAGE_COLUMNS]
            .sort_values(["identity_id", "rel_path"])
            .reset_index(drop=True)
        )
    dev_pairs = _make_pairs(
        frames["dev"],
        genuine_per_identity,
        impostor_per_identity_pair,
        _derived_seed(split_seed, "dev-pairs"),
    )
    test_pairs = _make_pairs(
        frames["test"],
        genuine_per_identity,
        impostor_per_identity_pair,
        _derived_seed(split_seed, "test-pairs"),
    )
    probe = _make_probe(
        frames["train"], probe_per_identity, _derived_seed(split_seed, "train-probe")
    )
    for split, frame in frames.items():
        _atomic_csv(output_dir / f"{split}.csv", frame)
    _atomic_csv(output_dir / "dev_pairs.csv", dev_pairs)
    _atomic_csv(output_dir / "test_pairs.csv", test_pairs)
    _atomic_csv(output_dir / "probe_triplets.csv", probe)
    _atomic_json(output_dir / "train_labels.json", train_labels)
    metadata = {
        "version": MANIFEST_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "data_root": str(data_root),
        "source": source,
        "source_fingerprint": fingerprint,
        "config": config,
        "hash_cache": hash_stats,
        "original_images": len(inventory),
        "unique_images": len(unique),
        "duplicates_removed": len(hashed) - len(unique),
        "duplicate_groups": int(duplicates.sha256.nunique()),
        "identity_labels": identity_labels,
        "splits": {
            split: {"images": len(frame), "identities": frame.identity_id.nunique()}
            for split, frame in frames.items()
        },
        "pairs": {
            "dev": {
                "genuine": int(dev_pairs.is_same.sum()),
                "impostor": int((dev_pairs.is_same == 0).sum()),
            },
            "test": {
                "genuine": int(test_pairs.is_same.sum()),
                "impostor": int((test_pairs.is_same == 0).sum()),
            },
        },
        "probe_triplets": len(probe),
        "artifact_sha256": {
            name: _sha256_file(output_dir / name) for name in _ARTIFACTS
        },
        "limitations": [
            "Byte-level deduplication does not detect perceptual near-duplicates, "
            "mislabeled people, or identity aliases.",
            "Hash cache reuse assumes file size and mtime_ns change "
            "when image bytes change.",
        ],
    }
    _atomic_json(metadata_path, metadata)
    return PreparedData(
        frames["train"],
        frames["dev"],
        frames["test"],
        dev_pairs,
        test_pairs,
        probe,
        train_labels,
        metadata,
    )


def bbox_array(frame: pd.DataFrame, *, required: bool = False) -> np.ndarray | None:
    """Validate complete original-image boxes before workers start decoding."""
    present = set(BBOX_COLUMNS).intersection(frame.columns)
    if not present and not required:
        return None
    if present != set(BBOX_COLUMNS):
        raise ValueError("Face bboxes require x1, y1, x2, y2 for every image.")
    boxes = frame[BBOX_COLUMNS].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(boxes).all()
        or (boxes < 0).any()
        or (boxes > np.iinfo(np.int32).max).any()
        or not np.equal(boxes, np.floor(boxes)).all()
        or (boxes[:, 2:] <= boxes[:, :2]).any()
    ):
        raise ValueError(
            "Face bboxes must be finite positive-area integer coordinates."
        )
    return boxes.astype(np.int32)


class FaceDataset(Dataset):
    """Decode/crop RGB faces on CPU with per-draw augmentation parameters.

    ``(index, draw_seed)`` indices supplied by :class:`PKBatchSampler` make image
    augmentation independent of worker scheduling, prefetch, or objective RNG.
    Plain integer indices use a stable per-image seed as a debug fallback.
    Training samples retain uint8 crops for transfer and device augmentation;
    evaluation samples retain deterministic normalized float32 inputs.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: Path,
        train: bool = False,
        seed: int = 42,
        require_bboxes: bool = False,
    ):
        required = {"rel_path", "identity_id", "label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"FaceDataset requires columns {sorted(required)}.")
        self.data_root = Path(data_root).resolve(strict=True)
        self.train, self.seed = bool(train), int(seed)
        # Windows spawn copies worker state. Keep compact metadata arrays only;
        # do not copy a full DataFrame or retain any decoded image cache.
        self.rel_paths = np.asarray(frame.rel_path, dtype=str)
        codes, identities = pd.factorize(frame.identity_id.astype(str), sort=True)
        self.identity_codes = codes.astype(np.int32)
        self.identity_names = np.asarray(identities, dtype=str)
        self.labels = frame.label.to_numpy(dtype=np.int64, copy=True)
        self.boxes = bbox_array(frame, required=require_bboxes)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, token):
        if isinstance(token, tuple):
            index, draw_seed = token
            index, draw_seed = int(index), int(draw_seed)
        else:
            index = int(token)
            draw_seed = _derived_seed(self.seed, f"image-{index}")
        relative = str(self.rel_paths[index])
        path = _safe_path(self.data_root, relative)
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        box = (
            tuple(int(value) for value in self.boxes[index])
            if self.boxes is not None
            else (0, 0, width, height)
        )
        if box[2] > width or box[3] > height:
            raise ValueError(f"Face bbox lies outside the original image: {relative}")
        cropped = crop_face(bgr, box, size=112, scale=1.0)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        values = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        tensor = torch.from_numpy(values)
        sample = {
            "image": tensor if self.train else tensor.float().div_(127.5).sub_(1),
            "label": int(self.labels[index]),
            "rel_path": relative,
            "identity_id": str(self.identity_names[self.identity_codes[index]]),
        }
        if self.train:
            rng = random.Random(draw_seed)
            flip = float(rng.random() < 0.5)
            brightness, contrast = (
                (rng.uniform(0.9, 1.1), rng.uniform(0.9, 1.1))
                if rng.random() < 0.5
                else (1.0, 1.0)
            )
            sample["augmentation_params"] = torch.tensor(
                [flip, brightness, contrast], dtype=torch.float32
            )
        return sample


class PKBatchSampler(Sampler):
    """Uniform identities and K distinct image indices per identity in full batches."""

    def __init__(
        self,
        labels: Sequence,
        P: int = 8,
        K: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if not isinstance(P, int) or not isinstance(K, int) or P < 2 or K < 2:
            raise ValueError("PKBatchSampler requires integer P >= 2 and K >= 2.")
        self.P, self.K, self.seed, self.epoch = P, K, int(seed), 0
        self.groups = {}
        for index, label in enumerate(labels):
            if pd.isna(label):
                raise ValueError("Sampler labels cannot be missing.")
            self.groups.setdefault(label, []).append(index)
        self.identities = list(self.groups)
        if len(self.identities) < P:
            raise ValueError(
                f"Sampler needs at least P={P} identities; "
                f"found {len(self.identities)}."
            )
        short = [label for label, indices in self.groups.items() if len(indices) < K]
        if short:
            raise ValueError(
                f"Each identity needs at least K={K} distinct images; "
                f"insufficient labels: {short[:10]}."
            )
        count = sum(len(indices) for indices in self.groups.values())
        self.batches_per_epoch = (
            math.ceil(count / (P * K))
            if batches_per_epoch is None
            else batches_per_epoch
        )
        if not isinstance(self.batches_per_epoch, int) or self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be a positive integer.")

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be nonnegative.")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        image_rng = random.Random(
            _derived_seed(self.seed, f"sampling-epoch-{self.epoch}")
        )
        draw_rng = random.Random(
            _derived_seed(self.seed, f"augmentation-epoch-{self.epoch}")
        )
        for _ in range(self.batches_per_epoch):
            batch = []
            for label in image_rng.sample(self.identities, self.P):
                for index in image_rng.sample(self.groups[label], self.K):
                    batch.append((index, draw_rng.getrandbits(63)))
            yield batch


def seed_worker(worker_id: int) -> None:
    """Seed non-Torch worker RNGs; draw-token augmentation itself uses a local RNG."""
    del worker_id
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(
    frame: pd.DataFrame,
    data_root: Path,
    *,
    train: bool = False,
    batch_size: int = 128,
    batch_sampler: PKBatchSampler | None = None,
    num_workers: int | None = None,
    notebook: bool = False,
    seed: int = 42,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    require_bboxes: bool = False,
) -> DataLoader:
    """Stream CPU crops with four spawn workers and a bounded prefetch queue.

    Scripts using workers must call this under a main guard (and may call
    ``multiprocessing.freeze_support()``). No live database connection is retained.
    Notebook workers import this module; ``notebook`` no longer disables them.
    """
    del notebook
    workers = 4 if num_workers is None else int(num_workers)
    if workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive.")
    if train and batch_sampler is None:
        raise ValueError("Training requires an explicit PKBatchSampler.")
    arguments = {
        "dataset": FaceDataset(
            frame, data_root, train=train, seed=seed, require_bboxes=require_bboxes
        ),
        "num_workers": workers,
        "pin_memory": pin_memory and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(int(seed)),
    }
    if workers:
        arguments.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context="spawn",
        )
    if batch_sampler is not None:
        arguments["batch_sampler"] = batch_sampler
    else:
        arguments.update(batch_size=batch_size, shuffle=False)
    return DataLoader(**arguments)
