"""Validated Podman image archives for managed ROCmplete build outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import posixpath
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Mapping, Sequence, Tuple

from .config import (
    APPLICATIONS,
    BUILD_APPLICATIONS,
    CONTENT_TOOLS_IMAGE,
    ROCM_BASE_IMAGE,
    ROCM_RUNTIME_IMAGE,
)
from .errors import LauncherError


_MAX_MANIFEST_SIZE = 4 * 1024 * 1024
_MAX_CONFIG_SIZE = 16 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 10000
_MAX_MANIFEST_ENTRIES = 64
_ARCHITECTURES = {
    "aarch64": "arm64",
    "amd64": "amd64",
    "arm64": "arm64",
    "x86_64": "amd64",
}


@dataclass(frozen=True)
class ArchivedImage:
    reference: str
    image_id: str
    architecture: str
    operating_system: str


@dataclass(frozen=True)
class ImageArchive:
    path: Path
    size: int
    images: Tuple[ArchivedImage, ...]


def managed_image_references() -> Tuple[str, ...]:
    return (CONTENT_TOOLS_IMAGE, ROCM_RUNTIME_IMAGE, ROCM_BASE_IMAGE) + tuple(
        APPLICATIONS[name].image for name in BUILD_APPLICATIONS
    )


def selected_image_references(target: str) -> Tuple[str, ...]:
    if target == "all":
        return managed_image_references()
    if target == "base":
        return (CONTENT_TOOLS_IMAGE, ROCM_RUNTIME_IMAGE, ROCM_BASE_IMAGE)
    if target not in BUILD_APPLICATIONS:
        raise LauncherError("unknown image export target: {}".format(target))
    application = APPLICATIONS[target]
    if application.shared_pytorch_base:
        return (
            CONTENT_TOOLS_IMAGE,
            ROCM_RUNTIME_IMAGE,
            ROCM_BASE_IMAGE,
            application.image,
        )
    return (CONTENT_TOOLS_IMAGE, ROCM_RUNTIME_IMAGE, application.image)


def save_command(images: Sequence[str], output: Path) -> Tuple[str, ...]:
    command = [
        "podman",
        "save",
        "--format",
        "docker-archive",
        "--output",
        str(output),
    ]
    if len(images) > 1:
        command.append("--multi-image-archive")
    command.extend(images)
    return tuple(command)


def load_command(path: Path) -> Tuple[str, ...]:
    return ("podman", "load", "--input", str(path))


def _safe_archive_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and bool(path.parts)
        and "\\" not in name
        and not path.is_absolute()
        and ".." not in path.parts
    )


def _read_member(
    archive: tarfile.TarFile,
    members: Mapping[str, Tuple[tarfile.TarInfo, ...]],
    name: str,
    maximum_size: int,
) -> bytes:
    matches = members.get(name, ())
    if len(matches) != 1:
        raise LauncherError(
            "image archive must contain exactly one {!r}; found {}".format(
                name, len(matches)
            )
        )
    member = matches[0]
    if not member.isfile():
        raise LauncherError(
            "image archive member {!r} is not a regular file".format(name)
        )
    if not 0 < member.size <= maximum_size:
        raise LauncherError(
            "image archive member {!r} has an unsafe size".format(name)
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise LauncherError(
            "cannot read image archive member {!r}".format(name)
        )
    with handle:
        contents = handle.read(maximum_size + 1)
    if len(contents) != member.size:
        raise LauncherError(
            "image archive member {!r} is truncated".format(name)
        )
    return contents


def _valid_layer_link(
    member: tarfile.TarInfo,
    members: Mapping[str, Tuple[tarfile.TarInfo, ...]],
) -> bool:
    if (
        not (member.issym() or member.islnk())
        or not member.name.endswith("/layer.tar")
        or not member.linkname
        or "\\" in member.linkname
    ):
        return False
    if member.islnk():
        candidate = member.linkname
    else:
        candidate = posixpath.join(
            posixpath.dirname(member.name), member.linkname
        )
    target_name = posixpath.normpath(candidate)
    if (
        not _safe_archive_name(target_name)
        or not target_name.endswith(".tar")
    ):
        return False
    targets = members.get(target_name, ())
    return len(targets) == 1 and targets[0].isfile()


def inspect_archive(path: Path) -> ImageArchive:
    try:
        status = path.stat()
    except OSError as error:
        raise LauncherError(
            "cannot inspect image archive {}: {}".format(path, error)
        )
    if not path.is_file() or status.st_size <= 0:
        raise LauncherError(
            "image archive is not a non-empty regular file: {}".format(path)
        )
    try:
        with tarfile.open(str(path), mode="r:*") as archive:
            grouped: Dict[str, list] = {}
            for index, member in enumerate(archive, 1):
                if index > _MAX_ARCHIVE_MEMBERS:
                    raise LauncherError(
                        "image archive contains too many members"
                    )
                if not _safe_archive_name(member.name):
                    raise LauncherError(
                        "image archive contains unsafe member {!r}".format(
                            member.name
                        )
                    )
                grouped.setdefault(member.name, []).append(member)
            duplicates = sorted(
                name for name, items in grouped.items() if len(items) > 1
            )
            if duplicates:
                raise LauncherError(
                    "image archive repeats members: {}".format(
                        ", ".join(duplicates)
                    )
                )
            member_map: Dict[str, Tuple[tarfile.TarInfo, ...]] = {
                name: tuple(items) for name, items in grouped.items()
            }
            for name, matches in member_map.items():
                member = matches[0]
                if (
                    not member.isfile()
                    and not member.isdir()
                    and not _valid_layer_link(member, member_map)
                ):
                    raise LauncherError(
                        "image archive contains unsupported member type {!r}".format(
                            name
                        )
                    )
            manifest_contents = _read_member(
                archive,
                member_map,
                "manifest.json",
                _MAX_MANIFEST_SIZE,
            )
            try:
                manifest = json.loads(manifest_contents)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise LauncherError(
                    "image archive manifest is invalid JSON: {}".format(error)
                )
            if (
                not isinstance(manifest, list)
                or not manifest
                or len(manifest) > _MAX_MANIFEST_ENTRIES
            ):
                raise LauncherError(
                    "image archive manifest has an unsafe entry count"
                )
            images = []
            references = set()
            for entry in manifest:
                if not isinstance(entry, dict):
                    raise LauncherError(
                        "image archive manifest contains an invalid entry"
                    )
                config_name = entry.get("Config")
                tags = entry.get("RepoTags")
                layers = entry.get("Layers")
                if (
                    not isinstance(config_name, str)
                    or not _safe_archive_name(config_name)
                    or not isinstance(tags, list)
                    or not tags
                    or not all(
                        isinstance(tag, str) and tag for tag in tags
                    )
                    or not isinstance(layers, list)
                    or not all(
                        isinstance(layer, str)
                        and _safe_archive_name(layer)
                        for layer in layers
                    )
                ):
                    raise LauncherError(
                        "image archive manifest entry has invalid config, "
                        "tags, or layers"
                    )
                for layer in layers:
                    matches = member_map.get(layer, ())
                    if len(matches) != 1 or not (
                        matches[0].isfile()
                        or _valid_layer_link(matches[0], member_map)
                    ):
                        raise LauncherError(
                            "image archive layer {!r} is missing or invalid".format(
                                layer
                            )
                        )
                config_contents = _read_member(
                    archive,
                    member_map,
                    config_name,
                    _MAX_CONFIG_SIZE,
                )
                try:
                    config = json.loads(config_contents)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise LauncherError(
                        "image config {!r} is invalid JSON: {}".format(
                            config_name, error
                        )
                    )
                if not isinstance(config, dict):
                    raise LauncherError(
                        "image config {!r} must be an object".format(
                            config_name
                        )
                    )
                architecture = config.get("architecture")
                operating_system = config.get("os")
                if (
                    not isinstance(architecture, str)
                    or not architecture
                    or not isinstance(operating_system, str)
                    or not operating_system
                ):
                    raise LauncherError(
                        "image config {!r} has no OS or architecture".format(
                            config_name
                        )
                    )
                image_id = "sha256:{}".format(
                    hashlib.sha256(config_contents).hexdigest()
                )
                if config_name != "{}.json".format(image_id[7:]):
                    raise LauncherError(
                        "image config filename does not match its digest: {}".format(
                            config_name
                        )
                    )
                for reference in tags:
                    if reference in references:
                        raise LauncherError(
                            "image archive repeats tag {!r}".format(reference)
                        )
                    references.add(reference)
                    images.append(
                        ArchivedImage(
                            reference,
                            image_id,
                            architecture,
                            operating_system,
                        )
                    )
    except (OSError, tarfile.TarError) as error:
        raise LauncherError(
            "cannot read image archive {}: {}".format(path, error)
        )
    return ImageArchive(
        path=path,
        size=status.st_size,
        images=tuple(sorted(images, key=lambda image: image.reference)),
    )


def validate_managed_archive(
    archive: ImageArchive,
    expected: Sequence[str] = (),
) -> None:
    references = {image.reference for image in archive.images}
    managed = set(managed_image_references())
    unexpected = sorted(references - managed)
    if unexpected:
        raise LauncherError(
            "image archive contains unmanaged or obsolete tags: {}".format(
                ", ".join(unexpected)
            )
        )
    if not references:
        raise LauncherError("image archive contains no tagged images")
    if CONTENT_TOOLS_IMAGE not in references:
        raise LauncherError(
            "image archive is missing the managed content-tools tag"
        )
    runtime_dependents = {ROCM_BASE_IMAGE} | {
        application.image for application in APPLICATIONS.values()
    }
    if (
        references & runtime_dependents
        and ROCM_RUNTIME_IMAGE not in references
    ):
        raise LauncherError(
            "image archive is missing the managed ROCm runtime tag"
        )
    shared_applications = {
        application.image
        for application in APPLICATIONS.values()
        if application.shared_pytorch_base
    }
    if references & shared_applications and ROCM_BASE_IMAGE not in references:
        raise LauncherError(
            "image archive is missing the managed ROCm/PyTorch base tag"
        )
    if expected and references != set(expected):
        missing = sorted(set(expected) - references)
        extra = sorted(references - set(expected))
        details = []
        if missing:
            details.append("missing {}".format(", ".join(missing)))
        if extra:
            details.append("unexpected {}".format(", ".join(extra)))
        raise LauncherError(
            "exported image archive has the wrong tags ({})".format(
                "; ".join(details)
            )
        )
    host_architecture = _ARCHITECTURES.get(
        platform.machine().lower(), platform.machine().lower()
    )
    for image in archive.images:
        if image.operating_system != "linux":
            raise LauncherError(
                "image {} targets unsupported OS {}".format(
                    image.reference, image.operating_system
                )
            )
        if image.architecture != host_architecture:
            raise LauncherError(
                "image {} targets architecture {}, not host {}".format(
                    image.reference,
                    image.architecture,
                    host_architecture,
                )
            )
