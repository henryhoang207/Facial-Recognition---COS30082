"""Stage just the eligible processed CelebA files from a large ZIP on Drive."""

import argparse
import hashlib
import json
import shutil
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import duckdb
from tqdm.auto import tqdm


def safe_name(info) -> str:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or ":" in name
        or stat.S_ISLNK(info.external_attr >> 16)
    ):
        raise ValueError(f"unsafe ZIP member: {info.filename}")
    return path.as_posix()


def _extract(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".extracting")
    try:
        with archive.open(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def stage_archive(
    source: Path, root: Path, *, reserve_bytes: int = 5 * 1024**3
) -> dict:
    source, root = Path(source).resolve(), Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_relative_to(source.parent) and str(source.parent).startswith(
        "/content/drive"
    ):
        raise ValueError("extract to local Colab storage, not mounted Drive")
    with ZipFile(source) as archive:
        members = {}
        for info in archive.infolist():
            name = safe_name(info)
            if name in members:
                raise ValueError(f"duplicate ZIP member: {name}")
            members[name] = info
        databases = [
            name
            for name in members
            if PurePosixPath(name).name == "celeba_spoof.duckdb"
        ]
        if len(databases) != 1:
            raise ValueError(
                "expected exactly one celeba_spoof.duckdb in processed archive"
            )
        database_name = databases[0]
        prefix = database_name.removesuffix("celeba_spoof.duckdb")
        database = root / "celeba_spoof.duckdb"
        if (
            members[database_name].file_size + reserve_bytes
            > shutil.disk_usage(root).free
        ):
            raise OSError("not enough local disk space to inspect the source database")
        _extract(archive, members[database_name], database)
        with duckdb.connect(str(database), read_only=True) as connection:
            paths = connection.execute(
                "SELECT path FROM celeba_labels WHERE split IN ('train','test') "
                "AND spoof_type IN (0,1,2,3,7,8,9) ORDER BY path"
            ).fetchall()
        names = {database_name}
        destinations = {database_name: database}
        for (relative,) in paths:
            relative = str(relative).replace("\\", "/")
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or ":" in relative:
                raise ValueError(f"unsafe catalog path: {relative}")
            for item in (relative, str(path.with_name(path.stem + "_BB.txt"))):
                name = prefix + item
                if name not in members:
                    raise FileNotFoundError(f"ZIP lacks required image/bbox: {name}")
                names.add(name)
                destinations[name] = root / item
        required = sum(members[name].file_size for name in names)
        identity = hashlib.sha256(
            json.dumps(
                [
                    (name, members[name].CRC, members[name].file_size)
                    for name in sorted(names)
                ]
            ).encode()
        ).hexdigest()
        marker = root / "staging.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous.get("identity") == identity and all(
                destinations[name].exists()
                and destinations[name].stat().st_size == members[name].file_size
                for name in names
            ):
                return previous
        free = shutil.disk_usage(root).free
        additional = sum(
            max(
                0,
                members[name].file_size
                - (
                    destinations[name].stat().st_size
                    if destinations[name].exists()
                    else 0
                ),
            )
            for name in names
        )
        largest = max(members[name].file_size for name in names)
        if additional + largest + reserve_bytes > free:
            raise OSError(
                f"selected data totals {required / 1024**3:.1f} GiB; "
                f"an additional {additional / 1024**3:.1f} GiB plus reserve "
                f"is needed, but only {free / 1024**3:.1f} GiB is free; "
                "no evaluation split was reduced"
            )
        copy_local = (
            source.stat().st_size + additional + largest + reserve_bytes <= free
        )
        local_archive = root.parent / "_celeba_source.zip"
        if copy_local and source != local_archive:
            shutil.copyfile(source, local_archive)
        try:
            with ZipFile(local_archive if copy_local else source) as payload:
                ordered = sorted(names, key=lambda name: members[name].header_offset)
                for name in tqdm(ordered, desc="Extract live / print / replay"):
                    if name != database_name:
                        _extract(payload, members[name], destinations[name])
        finally:
            if copy_local and source != local_archive:
                local_archive.unlink(missing_ok=True)
        result = {
            "identity": identity,
            "source": str(source),
            "archive_bytes": source.stat().st_size,
            "archive_uncompressed_bytes": sum(
                info.file_size for info in members.values()
            ),
            "selected_bytes": required,
            "selected_files": len(names),
            "copied_zip_locally": copy_local,
        }
        marker.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_archive(args.archive, args.root), indent=2))


if __name__ == "__main__":
    main()
