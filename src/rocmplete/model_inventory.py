"""Read-only discovery of runnable llama.cpp GGUF models."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from .catalog import Artifact, Catalog
from .content_verification import VerificationStore
from .errors import LauncherError
from .layout import StorageLayout


_GGUF_SHARD = re.compile(
    r"^(?P<prefix>.+)-(?P<part>[0-9]{5})-of-(?P<total>[0-9]{5})\.gguf$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LlamaModel:
    """One model a user can select, or one incomplete model-shaped set."""

    path: Path
    size: int
    state: str
    source: str
    presets: Tuple[str, ...] = ()
    shard_count: int = 1
    expected_shards: int = 1


def _path_state(
    path: Path, expected_size: int = -1, managed: bool = False
) -> Tuple[str, int]:
    try:
        status = path.lstat() if managed else path.stat()
    except FileNotFoundError:
        return ("missing", 0)
    except OSError:
        return ("missing", 0)
    if managed and not stat.S_ISREG(status.st_mode):
        return ("user-file", status.st_size)
    if not managed and not path.is_file():
        return ("missing", 0)
    size = status.st_size
    if expected_size >= 0 and size != expected_size:
        return ("size-mismatch", size)
    return ("ready", size)


def _catalog_models(
    catalog: Catalog, root: Path, verification_store: VerificationStore
) -> Tuple[Tuple[LlamaModel, ...], Set[Path]]:
    presets_by_artifact: Dict[str, List[str]] = {}
    for identifier, preset in catalog.llama_presets.items():
        presets_by_artifact.setdefault(preset.artifact, []).append(identifier)

    models = []
    claimed: Set[Path] = set()
    for artifact_identifier, preset_identifiers in presets_by_artifact.items():
        primary = catalog.artifact(artifact_identifier)
        required: Dict[str, Artifact] = {}
        for preset_identifier in preset_identifiers:
            preset = catalog.llama_preset(preset_identifier)
            for artifact in catalog.bundle_artifacts(
                catalog.bundle(preset.bundle)
            ):
                if artifact.target == "llama-models":
                    required[artifact.identifier] = artifact

        states = []
        observed_size = 0
        expected_size = sum(artifact.size for artifact in required.values())
        for artifact in required.values():
            path = root / artifact.destination
            claimed.add(path)
            state, file_size = _path_state(
                path, artifact.size, managed=True
            )
            if state == "ready" and not verification_store.matches(
                path, artifact.size, artifact.sha256
            ):
                state = "unverified"
            states.append(state)
            observed_size += file_size
        if states and all(state == "missing" for state in states):
            state = "missing"
        elif "user-file" in states:
            state = "user-file"
        elif "size-mismatch" in states:
            state = "size-mismatch"
        elif "missing" in states:
            state = "partial"
        elif "unverified" in states:
            state = "unverified"
        else:
            state = "ready"
        models.append(
            LlamaModel(
                path=root / primary.destination,
                size=(expected_size if state == "missing" else observed_size),
                state=state,
                source="catalog",
                presets=tuple(sorted(preset_identifiers)),
                shard_count=len(required),
                expected_shards=len(required),
            )
        )
    return tuple(models), claimed


def _directory_ggufs(root: Path) -> Tuple[Path, ...]:
    found = []

    def fail(error: OSError) -> None:
        raise LauncherError(
            "cannot scan model directory {}: {}".format(root, error)
        )

    for directory, names, files in os.walk(
        str(root), topdown=True, onerror=fail, followlinks=False
    ):
        # Never cross a directory symlink while recursively inventorying a
        # user-selected tree. An explicitly selected symlink root is fine.
        names[:] = [
            name
            for name in names
            if not (Path(directory) / name).is_symlink()
        ]
        for name in files:
            if name.lower().endswith(".gguf"):
                found.append(Path(directory) / name)
    return tuple(found)


def _scan_path(value: Path) -> Tuple[Path, ...]:
    try:
        path = Path(os.path.abspath(str(value.expanduser())))
    except OSError as error:
        raise LauncherError("cannot resolve model scan path {}: {}".format(
            value, error
        ))
    if not path.exists():
        raise LauncherError("model scan path does not exist: {}".format(path))
    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise LauncherError(
                "model scan file is not a .gguf file: {}".format(path)
            )
        return (path,)
    if not path.is_dir():
        raise LauncherError(
            "model scan path is not a file or directory: {}".format(path)
        )
    return _directory_ggufs(path)


def _loose_models(paths: Iterable[Path]) -> Tuple[LlamaModel, ...]:
    standalone = []
    shard_groups: Dict[Tuple[Path, str, int], Dict[int, Path]] = {}
    for path in paths:
        match = _GGUF_SHARD.match(path.name)
        if match is None:
            state, size = _path_state(path)
            standalone.append(
                LlamaModel(
                    path=path,
                    size=size,
                    state="broken" if state == "missing" else state,
                    source="local",
                )
            )
            continue
        part = int(match.group("part"))
        total = int(match.group("total"))
        if total < 1 or part < 1 or part > total:
            state, size = _path_state(path)
            standalone.append(
                LlamaModel(
                    path=path,
                    size=size,
                    state="broken" if state == "missing" else state,
                    source="local",
                )
            )
            continue
        key = (path.parent, match.group("prefix"), total)
        shard_groups.setdefault(key, {})[part] = path

    models = list(standalone)
    for (_, _, total), parts in shard_groups.items():
        states_and_sizes = tuple(_path_state(path) for path in parts.values())
        complete = (
            set(parts) == set(range(1, total + 1))
            and all(state == "ready" for state, _ in states_and_sizes)
        )
        first = parts.get(1, parts[min(parts)])
        models.append(
            LlamaModel(
                path=first,
                size=sum(size for _, size in states_and_sizes),
                state="ready" if complete else "partial",
                source="local",
                shard_count=len(parts),
                expected_shards=total,
            )
        )
    return tuple(models)


def llama_models(
    catalog: Catalog,
    data_dir: Path,
    scan_paths: Sequence[Path] = (),
) -> Tuple[LlamaModel, ...]:
    """Inventory catalog presets and loose GGUFs without creating state."""
    root = StorageLayout(data_dir).llama_models
    catalog_models, claimed = _catalog_models(
        catalog, root, VerificationStore.load(data_dir)
    )

    paths: Dict[Path, Path] = {}

    def remember(path: Path) -> None:
        try:
            identity = path.resolve(strict=False)
        except OSError:
            identity = path
        paths.setdefault(identity, path)

    if root.is_dir():
        for path in _directory_ggufs(root):
            remember(path)
    elif root.exists():
        raise LauncherError(
            "llama.cpp model root is not a directory: {}".format(root)
        )
    for value in scan_paths:
        for path in _scan_path(value):
            remember(path)

    claimed_identities = set()
    for path in claimed:
        try:
            claimed_identities.add(path.resolve(strict=False))
        except OSError:
            claimed_identities.add(path)
    loose = _loose_models(
        path
        for identity, path in paths.items()
        if identity not in claimed_identities
    )
    return tuple(
        sorted(
            catalog_models + loose,
            key=lambda item: (
                item.source != "catalog",
                str(item.path).lower(),
            ),
        )
    )
