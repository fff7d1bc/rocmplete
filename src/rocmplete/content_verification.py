"""Durable verification receipts for immutable managed content."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Optional

from .errors import LauncherError
from .layout import StorageLayout


SCHEMA_VERSION = 1


def _signature(status: os.stat_result) -> Dict[str, int]:
    return {
        "st_dev": status.st_dev,
        "st_ino": status.st_ino,
        "st_mtime_ns": status.st_mtime_ns,
        "st_ctime_ns": status.st_ctime_ns,
    }


class VerificationStore:
    """Cache successful hashes while the exact filesystem object is stable."""

    def __init__(
        self,
        data_dir: Path,
        path: Path,
        records: Optional[Mapping[str, Mapping[str, object]]] = None,
    ) -> None:
        self.data_dir = data_dir
        self.path = path
        self.records: Dict[str, Dict[str, object]] = {
            key: dict(value) for key, value in (records or {}).items()
        }
        self.changed = False

    @classmethod
    def load(cls, data_dir: Path) -> "VerificationStore":
        path = StorageLayout(data_dir).content_verification
        try:
            status = path.lstat()
        except FileNotFoundError:
            return cls(data_dir, path)
        except OSError as error:
            raise LauncherError(
                "cannot inspect content verification receipt {}: {}".format(
                    path, error
                )
            )
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise LauncherError(
                "content verification receipt is not a regular file: {}".format(
                    path
                )
            )
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print(
                "WARNING: cannot read content verification receipt {}; "
                "managed content will be treated as unverified: {}".format(
                    path, error
                ),
                file=sys.stderr,
            )
            return cls(data_dir, path)
        if not isinstance(document, dict) or document.get("schema") != SCHEMA_VERSION:
            raise LauncherError(
                "unsupported content verification receipt schema in {}".format(path)
            )
        records = document.get("files")
        if not isinstance(records, dict):
            raise LauncherError(
                "invalid content verification receipt records in {}".format(path)
            )
        for key, record in records.items():
            if not isinstance(key, str) or not isinstance(record, dict):
                raise LauncherError(
                    "invalid content verification receipt record in {}".format(path)
                )
        return cls(data_dir, path, records)

    def _key(self, path: Path, strict: bool = True) -> str:
        try:
            root = self.data_dir.resolve(strict=strict)
            resolved = path.resolve(strict=strict)
            relative = resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise LauncherError(
                "managed content path is outside the data directory: {} ({})".format(
                    path, error
                )
            )
        if not relative.parts:
            raise LauncherError("managed content path cannot be the data directory")
        return relative.as_posix()

    def matches(self, path: Path, size: int, sha256: str) -> bool:
        try:
            status = path.lstat()
        except OSError:
            return False
        if not stat.S_ISREG(status.st_mode) or status.st_size != size:
            return False
        key = self._key(path)
        record = self.records.get(key)
        if record is None:
            return False
        expected = {
            "sha256": sha256,
            "size": size,
            **_signature(status),
        }
        return record == expected

    def record(
        self,
        path: Path,
        size: int,
        sha256: str,
        status: Optional[os.stat_result] = None,
    ) -> None:
        try:
            current = path.lstat() if status is None else status
        except OSError as error:
            raise LauncherError(
                "cannot record content verification for {}: {}".format(path, error)
            )
        if not stat.S_ISREG(current.st_mode) or current.st_size != size:
            raise LauncherError(
                "cannot record verification for unexpected content file: {}".format(
                    path
                )
            )
        key = self._key(path)
        record = {
            "sha256": sha256,
            "size": size,
            **_signature(current),
        }
        if self.records.get(key) != record:
            self.records[key] = record
            self.changed = True

    def save(self) -> None:
        if not self.changed:
            return
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            parent_status = parent.lstat()
        except OSError as error:
            raise LauncherError(
                "cannot prepare content verification directory {}: {}".format(
                    parent, error
                )
            )
        if stat.S_ISLNK(parent_status.st_mode) or not stat.S_ISDIR(
            parent_status.st_mode
        ):
            raise LauncherError(
                "content verification directory is not a directory: {}".format(
                    parent
                )
            )
        document = {"schema": SCHEMA_VERSION, "files": self.records}
        encoded = (
            json.dumps(document, sort_keys=True, indent=2, separators=(",", ": "))
            + "\n"
        ).encode("utf-8")
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".verification-", suffix=".tmp", dir=str(parent)
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, str(self.path))
            temporary = ""
            directory_fd = os.open(str(parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise LauncherError(
                "cannot write content verification receipt {}: {}".format(
                    self.path, error
                )
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
        self.changed = False
