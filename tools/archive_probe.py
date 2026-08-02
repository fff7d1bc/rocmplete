#!/usr/bin/env python3
"""Inspect and hash ZIP members without extracting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence


CHUNK_SIZE = 1024 * 1024


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    while True:
        block = handle.read(CHUNK_SIZE)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and bool(path.parts)
        and "\\" not in name
        and not path.is_absolute()
        and not any(part in (".", "..") for part in path.parts)
    )


def _member_type(item: zipfile.ZipInfo) -> str:
    mode = item.external_attr >> 16
    if item.is_dir():
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_IFMT(mode) in (0, stat.S_IFREG):
        return "file"
    return "other"


def _selected_members(
    archive: zipfile.ZipFile, names: Sequence[str]
) -> Iterable[zipfile.ZipInfo]:
    members = archive.infolist()
    if not names:
        return members
    selected = []
    for name in names:
        matches = [item for item in members if item.filename == name]
        if len(matches) != 1:
            raise RuntimeError(
                "expected exactly one archive member {!r}; found {}".format(
                    name, len(matches)
                )
            )
        selected.append(matches[0])
    return selected


def probe(path: Path, names: Sequence[str] = ()) -> Mapping[str, object]:
    try:
        archive_size = path.stat().st_size
        archive_sha256 = _sha256_file(path)
        with zipfile.ZipFile(str(path)) as archive:
            all_names = [item.filename for item in archive.infolist()]
            name_counts = Counter(all_names)
            duplicates = sorted(
                name for name, count in name_counts.items() if count > 1
            )
            members = []
            for item in sorted(
                _selected_members(archive, names),
                key=lambda value: value.filename,
            ):
                kind = _member_type(item)
                summary: Dict[str, object] = {
                    "name": item.filename,
                    "type": kind,
                    "safe_path": _safe_member(item.filename),
                    "size": item.file_size,
                    "compressed_size": item.compress_size,
                    "sha256": None,
                }
                if kind == "file":
                    with archive.open(item) as handle:
                        summary["sha256"] = _sha256_stream(handle)
                members.append(summary)
    except (
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise RuntimeError("cannot inspect archive {}: {}".format(path, error))
    return {
        "path": str(path),
        "size": archive_size,
        "sha256": archive_sha256,
        "duplicate_names": duplicates,
        "members": members,
    }


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local ZIP and hash members without extraction."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        help="inspect one exact member; repeat to select several",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    arguments = parse_arguments(argv or sys.argv[1:])
    try:
        value = probe(arguments.archive, arguments.member)
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
