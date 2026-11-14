"""Adapt published recognition manifests to streaming B0/B1 runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    BBOX_COLUMNS,
    IMAGE_COLUMNS,
    PAIR_COLUMNS,
    PreparedData,
    _derived_seed,
    _make_probe,
    _sha256_file,
    bbox_array,
)


def load_face_protocol(
    data_root: Path,
    protocol: Path,
    *,
    seed: int = 42,
    images_per_identity: int = 4,
    probe_per_identity: int = 4,
) -> PreparedData:
    """Read completed bbox manifests; never decode or preload dataset images.

    Training uses only the classification TRAIN rows. Verification val/test
    become identity-disjoint dev/test, using the published finite pair lists.
    Classification val/test rows stay excluded. No physical val/ folder is needed.
    ``protocol`` accepts a generation directory or its parent's current.json.
    """
    if images_per_identity < 2 or probe_per_identity < 1:
        raise ValueError(
            "images_per_identity >= 2 and probe_per_identity >= 1 required."
        )
    data_root = Path(data_root).resolve(strict=True)
    protocol = Path(protocol).resolve()
    if protocol.name == "current.json":
        if not protocol.is_file():
            raise FileNotFoundError(
                f"Missing completed face protocol: {protocol}. "
                "Run src.datasets.prepare "
                "--only recognition, then src.recognition.preprocessing.prepare "
                "as described in docs/recognition-face-preparation.md."
            )
        pointer = json.loads(protocol.read_text(encoding="utf-8"))
        signature = pointer.get("protocol_sha256", "")
        if (
            not pointer.get("complete")
            or len(signature) != 64
            or any(char not in "0123456789abcdef" for char in signature)
        ):
            raise ValueError("current.json must reference a completed face protocol.")
        # Resolve next to the pointer so a copied dataset remains portable.
        protocol = protocol.parent / f"protocol_{signature[:12]}"
    summary = json.loads((protocol / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("Face detection is incomplete; no training protocol is ready.")
    return _load_protocol(
        data_root,
        protocol,
        seed=seed,
        images_per_identity=images_per_identity,
        probe_per_identity=probe_per_identity,
        require_bboxes=True,
        marker_names=["summary.json"],
    )


def load_image_protocol(
    data_root: Path,
    protocol_dir: Path,
    *,
    seed: int = 42,
    images_per_identity: int = 4,
    probe_per_identity: int = 4,
) -> PreparedData:
    """Read completed base manifests and use the full images without detection.

    Classification TRAIN supplies the learning identities; verification val/test
    supply unseen dev/test identities and their existing finite pair lists.
    Both preparation summaries must agree before a base generation is consumed.
    Only metadata is loaded: image decoding stays in the streaming DataLoader.
    """
    if images_per_identity < 2 or probe_per_identity < 1:
        raise ValueError(
            "images_per_identity >= 2 and probe_per_identity >= 1 required."
        )
    data_root = Path(data_root).resolve(strict=True)
    protocol_dir = Path(protocol_dir).resolve()
    marker_names = ["dataset_report.json", "recognition_summary.json"]
    if not all((protocol_dir / name).is_file() for name in marker_names):
        raise FileNotFoundError(
            f"Missing completed base recognition manifests: {protocol_dir}. "
            "Run src.datasets.prepare --only recognition."
        )
    report = json.loads((protocol_dir / marker_names[0]).read_text(encoding="utf-8"))
    summary = json.loads((protocol_dir / marker_names[1]).read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or not summary
        or not isinstance(report, dict)
        or report.get("recognition") != summary
        or not isinstance(summary.get("classification_splits"), dict)
        or not isinstance(summary.get("verification"), dict)
    ):
        raise ValueError(
            "Base recognition preparation is incomplete: completion summaries disagree."
        )
    return _load_protocol(
        data_root,
        protocol_dir,
        seed=seed,
        images_per_identity=images_per_identity,
        probe_per_identity=probe_per_identity,
        require_bboxes=False,
        marker_names=marker_names,
        expected_summary=summary,
    )


def _load_protocol(
    data_root: Path,
    protocol: Path,
    *,
    seed: int,
    images_per_identity: int,
    probe_per_identity: int,
    require_bboxes: bool,
    marker_names: list[str],
    expected_summary: dict | None = None,
) -> PreparedData:
    names = [
        "vggface2_classification.csv",
        "vggface2_verification.csv",
        "verification_pairs_val.csv",
        "verification_pairs_test.csv",
        *marker_names,
    ]
    hashes = {name: _sha256_file(protocol / name) for name in names}
    extra_columns = BBOX_COLUMNS if require_bboxes else []
    columns = ["path", "identity", "split", "sha256", *extra_columns]

    def read_images(name, selected_splits):
        chunks = []
        for chunk in pd.read_csv(
            protocol / name,
            usecols=columns,
            dtype={"path": str, "identity": str, "split": str, "sha256": str},
            chunksize=100_000,
        ):
            chunks.append(chunk[chunk.split.isin(selected_splits)].copy())
        frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"Empty or incomplete recognition manifest: {name}")
        frame = frame.rename(columns={"path": "rel_path", "identity": "identity_id"})
        # Vectorized lexical check; FaceDataset also resolves symlinks on access.
        frame["rel_path"] = frame.rel_path.str.replace("\\", "/", regex=False)
        if frame.rel_path.duplicated().any():
            raise ValueError(f"Duplicate image paths in {name}")
        unsafe = frame.rel_path.str.contains(r"(?:^/|:|(?:^|/)\.\.(?:/|$))", regex=True)
        if unsafe.any():
            raise ValueError(f"Unsafe dataset relative path in {name}")
        if not frame.sha256.str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid SHA256 in {name}")
        if require_bboxes:
            bbox_array(frame, required=True)
        return frame

    train = read_images("vggface2_classification.csv", {"train"})
    verification = read_images("vggface2_verification.csv", {"val", "test"})
    dev = verification[verification.split == "val"].copy()
    test = verification[verification.split == "test"].copy()
    del verification
    if expected_summary is not None:
        if len(train) != expected_summary["classification_splits"].get("train"):
            raise ValueError("Base classification train count disagrees with summary.")
        for name, frame in (("val", dev), ("test", test)):
            details = expected_summary["verification"].get(name, {})
            if not isinstance(
                details, dict
            ) or frame.identity_id.nunique() != details.get("identities"):
                raise ValueError(
                    f"Base verification {name} identity count disagrees with summary."
                )
    frames = [train, dev, test]
    identities, paths, digests = set(), set(), set()
    for frame in frames:
        if frame.identity_id.nunique() < 2:
            raise ValueError("Each recognition split needs at least two identities.")
        new_ids = set(frame.identity_id)
        new_paths, new_hashes = set(frame.rel_path), set(frame.sha256)
        if identities & new_ids or paths & new_paths or digests & new_hashes:
            raise ValueError(
                "Identity, image path, or exact-duplicate leakage across splits."
            )
        conflicts = frame.groupby("sha256", sort=False).identity_id.nunique()
        if (conflicts > 1).any():
            raise ValueError("Exact-duplicate images have conflicting identity labels.")
        identities.update(new_ids)
        paths.update(new_paths)
        digests.update(new_hashes)
    del identities, paths, digests
    # P-by-K positives must contain distinct image content, not duplicate files.
    train = train.drop_duplicates("sha256").copy()
    counts = train.groupby("identity_id").size()
    dropped = sorted(counts[counts < images_per_identity].index)
    train = train[~train.identity_id.isin(dropped)].copy()
    if train.identity_id.nunique() < 2:
        raise ValueError("Too few train identities remain for P-by-K sampling.")
    labels = {}
    for split, frame in zip(("train", "dev", "test"), (train, dev, test), strict=True):
        for identity in sorted(frame.identity_id.unique()):
            labels[identity] = len(labels)
        frame["label"] = frame.identity_id.map(labels).astype(np.int64)
        frame["split"] = split
    train_labels = {
        identity: labels[identity] for identity in sorted(train.identity_id.unique())
    }

    def read_pairs(name, frame):
        pairs = pd.read_csv(protocol / name, dtype={"path1": str, "path2": str})
        if not {"path1", "path2", "label"}.issubset(pairs.columns) or pairs.empty:
            raise ValueError(f"Incomplete verification pairs: {name}")
        pairs = pairs.rename(
            columns={"path1": "path_a", "path2": "path_b", "label": "is_same"}
        )
        for column in ("path_a", "path_b"):
            pairs[column] = pairs[column].str.replace("\\", "/", regex=False)
        known = frame.set_index("rel_path")
        pairs["identity_a"] = pairs.path_a.map(known.identity_id)
        pairs["identity_b"] = pairs.path_b.map(known.identity_id)
        if pairs[["identity_a", "identity_b"]].isna().any().any():
            raise ValueError(f"Verification pair references a missing image: {name}")
        if (
            not pairs.is_same.isin([0, 1]).all()
            or set(pairs.is_same) != {0, 1}
            or not np.array_equal(
                pairs.is_same.astype(bool), pairs.identity_a == pairs.identity_b
            )
            or (pairs.path_a == pairs.path_b).any()
            or (pairs.path_a.map(known.sha256) == pairs.path_b.map(known.sha256)).any()
        ):
            raise ValueError(
                f"Invalid verification pair labels or duplicate content: {name}"
            )
        canonical = np.sort(pairs[["path_a", "path_b"]].to_numpy(), axis=1)
        if pd.DataFrame(canonical).duplicated().any():
            raise ValueError(f"Duplicate verification pairs: {name}")
        return pairs[PAIR_COLUMNS]

    dev_pairs = read_pairs("verification_pairs_val.csv", dev)
    test_pairs = read_pairs("verification_pairs_test.csv", test)
    if expected_summary is not None:
        for name, pairs in (("val", dev_pairs), ("test", test_pairs)):
            if len(pairs) != expected_summary["verification"][name].get("pairs"):
                raise ValueError(
                    f"Base verification {name} pair count disagrees with summary."
                )
    frames = [
        frame[IMAGE_COLUMNS + extra_columns].reset_index(drop=True)
        for frame in (train, dev, test)
    ]
    probe = _make_probe(
        frames[0], probe_per_identity, _derived_seed(seed, "train-probe")
    )
    return PreparedData(
        *frames,
        dev_pairs,
        test_pairs,
        probe,
        train_labels,
        {
            "version": (
                "bbox-identification-v1"
                if require_bboxes
                else "full-image-identification-v1"
            ),
            "data_root": str(data_root),
            "protocol_dir": str(protocol),
            "artifact_sha256": hashes,
            "seed": seed,
            "images_per_identity": images_per_identity,
            "probe_per_identity": probe_per_identity,
            "dropped_short_train_identities": dropped,
            "splits": {
                name: {"images": len(frame), "identities": frame.identity_id.nunique()}
                for name, frame in zip(("train", "dev", "test"), frames, strict=True)
            },
            "crop": (
                "detector bbox -> square crop 112; no landmark alignment"
                if require_bboxes
                else "full image -> square pad/resize 112; no detection or alignment"
            ),
        },
    )
