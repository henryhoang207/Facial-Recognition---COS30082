"""Data integrity and reproducibility tests without model training."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from src.recognition.identification import data


@pytest.fixture
def tiny_faces(tmp_path):
    root = tmp_path / "faces"
    for split, count, offset in (("train", 6, 0), ("val", 3, 100)):
        for identity in range(count):
            directory = root / split / f"id{identity + offset:03d}"
            directory.mkdir(parents=True)
            for image_index in range(6):
                pixels = np.zeros((18, 26, 3), dtype=np.uint8)
                pixels[:, :, 0] = identity + offset + 10
                pixels[:, :, 1] = image_index * 30 + 10
                pixels[:, :, 2] = np.arange(26, dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f"{image_index:02d}.png")
    shutil.copyfile(root / "train/id000/00.png", root / "train/id000/99.png")
    return root, tmp_path / "manifests"


def prepare(root, output, **overrides):
    options = dict(
        dev_identities=2,
        genuine_per_identity=3,
        impostor_per_identity_pair=4,
        probe_per_identity=2,
    )
    options.update(overrides)
    return data.prepare_manifests(root, output, **options)


def test_shared_manifests_are_disjoint_deduplicated_and_reloadable(tiny_faces):
    root, output = tiny_faces
    prepared = prepare(root, output)
    frames = [prepared.train, prepared.dev, prepared.test]
    identity_sets = [set(frame.identity_id) for frame in frames]
    assert [len(ids) for ids in identity_sets] == [4, 2, 3]
    assert not any(
        identity_sets[left] & identity_sets[right]
        for left in range(3)
        for right in range(left)
    )
    combined = pd.concat(frames)
    assert len(combined) == 54
    assert combined.sha256.nunique() == 54
    assert prepared.metadata["duplicates_removed"] == 1
    assert len(list(root.rglob("*.png"))) == 55
    assert set(prepared.train_labels.values()) == set(range(4))
    assert set(prepared.dev.label).isdisjoint(prepared.train.label)
    assert set(prepared.test.label).isdisjoint(pd.concat(frames[:2]).label)
    assert set(prepared.train.rel_path.str.split("/").str[0]) == {"train"}
    assert set(prepared.dev.rel_path.str.split("/").str[0]) == {"train"}
    assert set(prepared.test.rel_path.str.split("/").str[0]) == {"val"}
    for filename, expected in prepared.metadata["artifact_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    reloaded = prepare(root, output)
    for name in ("train", "dev", "test", "dev_pairs", "test_pairs", "probe_triplets"):
        pd.testing.assert_frame_equal(getattr(prepared, name), getattr(reloaded, name))
    assert prepared.train_labels == reloaded.train_labels


def test_pairs_are_balanced_unique_and_triplets_use_train_only(tiny_faces):
    root, output = tiny_faces
    prepared = prepare(root, output)
    for frame, pairs in (
        (prepared.dev, prepared.dev_pairs),
        (prepared.test, prepared.test_pairs),
    ):
        positives = pairs[pairs.is_same == 1]
        negatives = pairs[pairs.is_same == 0]
        assert set(positives.groupby("identity_a").size()) == {3}
        assert set(negatives.groupby(["identity_a", "identity_b"]).size()) == {4}
        assert (positives.identity_a == positives.identity_b).all()
        assert (negatives.identity_a != negatives.identity_b).all()
        assert (pairs.path_a != pairs.path_b).all()
        assert (
            not pairs.apply(lambda row: tuple(sorted((row.path_a, row.path_b))), axis=1)
            .duplicated()
            .any()
        )
        identities = frame.set_index("rel_path").identity_id.to_dict()
        assert set(pairs.path_a) | set(pairs.path_b) <= set(identities)
        assert all(
            identities[row.path_a] == row.identity_a
            and identities[row.path_b] == row.identity_b
            for row in pairs.itertuples()
        )
    identities = prepared.train.set_index("rel_path").identity_id.to_dict()
    assert len(prepared.probe_triplets) == 8
    assert not prepared.probe_triplets.duplicated().any()
    for row in prepared.probe_triplets.itertuples():
        assert row.anchor != row.positive
        assert identities[row.anchor] == identities[row.positive]
        assert identities[row.anchor] != identities[row.negative]
    anchors = prepared.probe_triplets.anchor.map(identities)
    assert set(anchors.value_counts()) == {2}


@pytest.mark.parametrize("destination", ["train/id001/00.png", "val/id100/00.png"])
def test_exact_duplicate_leakage_is_rejected(tiny_faces, destination):
    root, output = tiny_faces
    shutil.copyfile(root / "train/id000/00.png", root / destination)
    with pytest.raises(ValueError, match="Exact-duplicate leakage"):
        prepare(root, output)


def test_identity_name_overlap_is_rejected(tiny_faces):
    root, output = tiny_faces
    (root / "val/id100").rename(root / "val/id000")
    with pytest.raises(ValueError, match="Identity leakage"):
        prepare(root, output)


def test_saved_manifest_config_root_and_artifact_changes_are_rejected(
    tiny_faces, tmp_path
):
    root, output = tiny_faces
    prepare(root, output)
    with pytest.raises(ValueError, match="Stale"):
        prepare(root, output, split_seed=7)
    moved_root = tmp_path / "other_faces"
    shutil.copytree(root, moved_root)
    with pytest.raises(ValueError, match="Stale"):
        prepare(moved_root, output)
    with (output / "dev_pairs.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="missing or modified"):
        prepare(root, output)


def test_hash_cache_reuses_only_matching_file_statistics(tiny_faces, monkeypatch):
    root, output = tiny_faces
    first = prepare(root, output)
    calls = []
    original_hash = data._sha256_file

    def counted_hash(path):
        if Path(path).suffix == ".png":
            calls.append(Path(path))
        return original_hash(path)

    monkeypatch.setattr(data, "_sha256_file", counted_hash)
    cached = prepare(root, output, force=True)
    assert calls == []
    assert cached.metadata["hash_cache"]["reused"] == 55
    assert cached.metadata["artifact_sha256"] == first.metadata["artifact_sha256"]
    modified = root / "train/id002/03.png"
    previous_stat = modified.stat()
    with Image.open(modified) as image:
        pixels = np.array(image)
    pixels[3, 3, 0] = 240
    Image.fromarray(pixels).save(modified)
    os.utime(
        modified, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns + 1_000_000)
    )
    with pytest.raises(ValueError, match="Stale"):
        prepare(root, output)
    changed = prepare(root, output, force=True)
    assert calls == [modified.resolve()]
    assert changed.metadata["hash_cache"]["computed"] == 1
    assert changed.metadata["hash_cache"]["reused"] == 54


def test_modified_hash_cache_is_not_trusted(tiny_faces):
    root, output = tiny_faces
    prepare(root, output)
    with (output / "hash_cache.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    rebuilt = prepare(root, output, force=True)
    assert rebuilt.metadata["hash_cache"]["computed"] == 55
    assert (
        rebuilt.metadata["hash_cache"]["cache_invalidation_reason"]
        == "cache checksum mismatch"
    )


def test_pair_request_cannot_silently_become_unbalanced(tiny_faces):
    root, output = tiny_faces
    with pytest.raises(ValueError, match="distinct genuine pairs"):
        prepare(root, output, genuine_per_identity=16)


def test_catalog_missing_image_is_rejected(tiny_faces):
    duckdb = pytest.importorskip("duckdb")
    root, output = tiny_faces
    with duckdb.connect(str(root / "vggface2.duckdb")) as connection:
        connection.execute(
            "CREATE TABLE vggface2_images(rel_path VARCHAR, identity_id VARCHAR, "
            "split VARCHAR, file_size_bytes BIGINT)"
        )
        connection.execute(
            "INSERT INTO vggface2_images VALUES "
            "('train/id000/missing.png','id000','train',20)"
        )
    with pytest.raises(FileNotFoundError, match="catalog references a missing"):
        prepare(root, output)


def face_frame(path="train/person/face.png"):
    return pd.DataFrame({"rel_path": [path], "identity_id": ["person"], "label": [0]})


def test_rgb_square_crop_preserves_channels_and_black_padding(tmp_path):
    path = tmp_path / "train/person/face.png"
    path.parent.mkdir(parents=True)
    pixels = np.empty((34, 60, 3), dtype=np.uint8)
    pixels[:] = [208, 11, 39]
    Image.fromarray(pixels).save(path)
    sample = data.FaceDataset(face_frame(), tmp_path)[0]
    assert sample["image"].shape == (3, 112, 112)
    assert sample["image"].dtype == torch.float32
    assert sample["identity_id"] == "person" and sample["label"] == 0
    torch.testing.assert_close(
        sample["image"][:, 56, 56], torch.tensor([208.0, 11.0, 39.0]) / 127.5 - 1
    )
    assert torch.equal(sample["image"][:, 0, 56], torch.full((3,), -1.0))


def test_draw_token_augmentation_is_local_and_reproducible(tmp_path):
    path = tmp_path / "train/person/face.png"
    path.parent.mkdir(parents=True)
    pixels = np.zeros((40, 40, 3), dtype=np.uint8)
    pixels[:, :20] = [210, 70, 30]
    pixels[:, 20:] = [20, 50, 200]
    Image.fromarray(pixels).save(path)
    dataset = data.FaceDataset(face_frame(), tmp_path, train=True)
    random.seed(95)
    before = random.getstate()
    flipped = dataset[(0, 1)]["image"]
    assert random.getstate() == before
    plain = dataset[(0, 2)]["image"]
    torch.testing.assert_close(flipped, plain.flip(-1))
    random.seed(100)
    np.random.seed(200)
    torch.manual_seed(300)
    assert torch.equal(dataset[(0, 1)]["image"], flipped)
    assert torch.equal(dataset[0]["image"], dataset[0]["image"])
    evaluation = data.FaceDataset(face_frame(), tmp_path, train=False)
    assert torch.equal(evaluation[(0, 1)]["image"], evaluation[(0, 900)]["image"])


def test_dataset_fails_fast_on_missing_corrupt_or_unsafe_images(tmp_path):
    dataset = data.FaceDataset(face_frame("missing.png"), tmp_path)
    with pytest.raises(FileNotFoundError):
        dataset[0]
    (tmp_path / "broken.png").write_bytes(b"this is not an image")
    with pytest.raises(UnidentifiedImageError):
        data.FaceDataset(face_frame("broken.png"), tmp_path)[0]
    with pytest.raises(ValueError, match="Unsafe"):
        data.FaceDataset(face_frame("../outside.png"), tmp_path)[0]


def test_pk_batches_have_exact_counts_distinct_files_and_epoch_reproducibility():
    labels = np.repeat(np.arange(5), [5, 6, 7, 8, 9])
    sampler = data.PKBatchSampler(labels, P=3, K=2, seed=42)
    assert len(sampler) == 6
    epoch_zero = list(sampler)
    for batch in epoch_zero:
        indices, seeds = zip(*batch, strict=True)
        assert len(batch) == 6 and len(set(indices)) == 6
        assert set(Counter(labels[list(indices)]).values()) == {2}
        assert len(Counter(labels[list(indices)])) == 3
        assert len(set(seeds)) == len(batch)
    random.seed(900)
    torch.manual_seed(800)
    assert list(sampler) == epoch_zero
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero
    duplicate = data.PKBatchSampler(labels, P=3, K=2, seed=42)
    duplicate.set_epoch(1)
    assert list(duplicate) == list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch_zero


def test_pk_identity_draws_do_not_follow_image_frequency():
    labels = [0] * 4 + [1] * 8 + [2] * 16 + [3] * 32
    sampler = data.PKBatchSampler(labels, P=2, K=2, batches_per_epoch=2000, seed=42)
    counts = Counter()
    for batch in sampler:
        counts.update(set(labels[index] for index, _ in batch))
    assert max(counts.values()) / min(counts.values()) < 1.2


@pytest.mark.parametrize(
    "labels,P,K", [([0, 0], 2, 2), ([0, 0, 1], 2, 2), ([0, 0, 1, 1], 1, 2)]
)
def test_pk_invalid_small_batches_fail_clearly(labels, P, K):
    with pytest.raises(ValueError):
        data.PKBatchSampler(labels, P=P, K=K)


def test_notebook_loader_uses_seeded_pk_tokens_without_worker_options(tiny_faces):
    root, output = tiny_faces
    prepared = prepare(root, output)
    sampler = data.PKBatchSampler(prepared.train.label, P=2, K=2, batches_per_epoch=1)
    loader = data.make_loader(
        prepared.train, root, train=True, batch_sampler=sampler, notebook=True
    )
    assert loader.num_workers == 0 and not loader.persistent_workers
    assert loader.prefetch_factor is None
    batch = next(iter(loader))
    expected = [loader.dataset[token] for token in next(iter(sampler))]
    assert torch.equal(
        batch["image"], torch.stack([item["image"] for item in expected])
    )
    assert batch["rel_path"] == [item["rel_path"] for item in expected]


def test_spawn_workers_preserve_per_draw_augmentation_across_epochs(tiny_faces):
    root, output = tiny_faces
    prepared = prepare(root, output)
    serial_sampler = data.PKBatchSampler(
        prepared.train.label, P=2, K=2, batches_per_epoch=2, seed=72
    )
    spawn_sampler = data.PKBatchSampler(
        prepared.train.label, P=2, K=2, batches_per_epoch=2, seed=72
    )
    serial = data.make_loader(
        prepared.train,
        root,
        train=True,
        batch_sampler=serial_sampler,
        num_workers=0,
        seed=72,
    )
    parallel = data.make_loader(
        prepared.train,
        root,
        train=True,
        batch_sampler=spawn_sampler,
        num_workers=2,
        seed=72,
    )
    for epoch in range(2):
        serial_sampler.set_epoch(epoch)
        spawn_sampler.set_epoch(epoch)
        for first, second in zip(serial, parallel, strict=True):
            assert first["rel_path"] == second["rel_path"]
            assert torch.equal(first["label"], second["label"])
            assert torch.equal(first["image"], second["image"])
