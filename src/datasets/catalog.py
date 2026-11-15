"""Shared export and fingerprint helpers. All writes belong to the output directory."""

import csv
import hashlib
import io
import json
import os
import random
import sqlite3
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from uuid import uuid4

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_csv(path: Path, rows: Iterable[dict], columns: list[str]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def holdout(items: list, fraction: float, rng: random.Random) -> tuple[list, list]:
    if len(items) < 2:
        raise ValueError("a holdout needs at least two independent groups")
    items = sorted(items)
    rng.shuffle(items)
    count = min(len(items) - 1, max(1, round(len(items) * fraction)))
    return sorted(items[count:]), sorted(items[:count])


def ordered_parallel(function: Callable, items: list, workers: int):
    """Bound queued futures while preserving source order for deterministic output."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), 512):
            yield from pool.map(function, items[start : start + 512])


class FingerprintCache:
    """Cache full-file SHA256 and image-header checks by path, size and mtime."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "fingerprints.sqlite3"

    @staticmethod
    def _inspect(path: Path) -> dict:
        try:
            stat = path.stat()
            content = path.read_bytes()
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                image.verify()
            return {
                "path": str(path),
                "size": stat.st_size,
                "mtime": stat.st_mtime_ns,
                "sha256": hashlib.sha256(content).hexdigest(),
                "width": width,
                "height": height,
                "error": "",
            }
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            return {
                "path": str(path),
                "size": -1,
                "mtime": -1,
                "sha256": "",
                "width": 0,
                "height": 0,
                "error": str(exc),
            }

    def inspect(self, paths: list[Path], workers: int) -> dict[str, dict]:
        return {row["path"]: row for row in self.iter_inspect(paths, workers)}

    def iter_inspect(self, paths: Iterable[Path], workers: int):
        columns = ("path", "size", "mtime", "sha256", "width", "height", "error")
        with (
            sqlite3.connect(self.path) as connection,
            ThreadPoolExecutor(max_workers=workers) as pool,
        ):
            connection.execute(
                "CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, "
                "size INTEGER, mtime INTEGER, sha256 TEXT, "
                "width INTEGER, height INTEGER, error TEXT)"
            )

            def inspect_or_reuse(path, previous):
                if previous is not None:
                    try:
                        stat = path.stat()
                        if (
                            previous["size"] == stat.st_size
                            and previous["mtime"] == stat.st_mtime_ns
                        ):
                            return previous, False
                    except OSError:
                        pass
                return self._inspect(path), True

            checked = 0
            for batch in batched(paths, 512):
                keys = tuple(map(str, batch))
                placeholders = ",".join("?" for _ in keys)
                cached = {
                    row[0]: dict(zip(columns, row, strict=True))
                    for row in connection.execute(
                        f"SELECT * FROM images WHERE path IN ({placeholders})", keys
                    )
                }
                results = list(
                    pool.map(inspect_or_reuse, batch, (cached.get(key) for key in keys))
                )
                connection.executemany(
                    "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (tuple(record.values()) for record, changed in results if changed),
                )
                if checked // 20000 != (checked + len(batch)) // 20000:
                    connection.commit()
                    print(f"fingerprints: checked {checked + len(batch)}", flush=True)
                checked += len(batch)
                for record, _ in results:
                    yield record
            connection.commit()
