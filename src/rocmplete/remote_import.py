"""Resolve supported model-page URLs into verified local content packs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .errors import LauncherError
from .project import PROJECT_ROOT


_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_SAFE_IDENTIFIER = re.compile(r"[^a-z0-9._-]+")
_WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth", ".bin")
_CIVITAI_KIND_CANDIDATES = {
    "checkpoint": (
        "comfyui:checkpoint",
        "comfyui:diffusion-model",
    ),
    "lora": ("comfyui:lora",),
    "locon": ("comfyui:lora",),
    "dora": ("comfyui:lora",),
    "controlnet": ("comfyui:controlnet",),
    "vae": ("comfyui:vae",),
    "upscaler": ("comfyui:upscaler",),
    "workflows": ("comfyui:workflow",),
    "workflow": ("comfyui:workflow",),
}
_UNSUPPORTED_CIVITAI_TYPES = {
    "aestheticgradient",
    "embedding",
    "hypernetwork",
    "motionmodule",
    "poses",
    "textualinversion",
    "wildcards",
}


@dataclass(frozen=True)
class ImportKind:
    identifier: str
    label: str
    application: str
    target: str
    destination_prefix: str
    suffixes: Tuple[str, ...]


IMPORT_KINDS: Mapping[str, ImportKind] = {
    item.identifier: item
    for item in (
        ImportKind(
            "comfyui:checkpoint",
            "ComfyUI checkpoint",
            "comfyui",
            "models",
            "checkpoints/imported",
            (".safetensors", ".ckpt"),
        ),
        ImportKind(
            "comfyui:diffusion-model",
            "ComfyUI diffusion model",
            "comfyui",
            "models",
            "diffusion_models/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:lora",
            "ComfyUI LoRA",
            "comfyui",
            "models",
            "loras/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:vae",
            "ComfyUI VAE",
            "comfyui",
            "models",
            "vae/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:text-encoder",
            "ComfyUI text encoder",
            "comfyui",
            "models",
            "text_encoders/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:controlnet",
            "ComfyUI ControlNet model",
            "comfyui",
            "models",
            "controlnet/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:upscaler",
            "ComfyUI upscaler",
            "comfyui",
            "models",
            "upscale_models/imported",
            _WEIGHT_SUFFIXES,
        ),
        ImportKind(
            "comfyui:workflow",
            "exact imported ComfyUI workflow",
            "comfyui",
            "workflows",
            "remote",
            (".json",),
        ),
        ImportKind(
            "llama-cpp:model",
            "llama.cpp GGUF model",
            "llama-cpp",
            "llama-models",
            "imported",
            (".gguf",),
        ),
    )
}


@dataclass(frozen=True)
class RemoteFile:
    identifier: str
    name: str
    size: int
    sha256: str
    primary: bool = False
    download_url: str = ""


@dataclass(frozen=True)
class RemoteDiscovery:
    provider: str
    source_url: str
    title: str
    repository: str
    revision: str
    files: Tuple[RemoteFile, ...]
    declared_license: str
    model_type: str = ""
    provider_host: str = ""
    model_id: int = 0
    model_version_id: int = 0
    requires_auth: bool = False


@dataclass(frozen=True)
class RemoteImportPlan:
    discovery: RemoteDiscovery
    file: RemoteFile
    kind: ImportKind
    artifact_identifier: str
    bundle_identifier: str
    destination: str
    pack: Mapping[str, object]


def _request_json(
    url: str, token: str, provider: str
) -> Mapping[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "ROCmplete-content-import/1",
    }
    request = urllib.request.Request(url, headers=headers)
    # Do not let urllib copy credentials if a provider redirects metadata to
    # another host. Download credentials use the same non-forwarding policy.
    if token:
        request.add_unredirected_header(
            "Authorization", "Bearer {}".format(token)
        )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.load(response)
    except (
        OSError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as error:
        raise LauncherError(
            "{} metadata request failed: {}".format(provider, error)
        )
    if not isinstance(value, dict):
        raise LauncherError("{} metadata response is not an object".format(
            provider
        ))
    return value


def _normalized_url(value: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError as error:
        raise LauncherError("invalid import URL: {}".format(error))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
    ):
        raise LauncherError(
            "content import requires a plain HTTPS provider URL"
        )
    return parsed


def remote_provider(url: str) -> str:
    parsed = _normalized_url(url)
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    if host in ("civitai.com", "civitai.red"):
        return "civitai"
    if host == "huggingface.co":
        return "huggingface"
    raise LauncherError(
        "unsupported import host {!r}; use civitai.com, civitai.red, "
        "or huggingface.co".format(parsed.hostname)
    )


def _civitai_identity(url: str) -> Tuple[str, int, int]:
    parsed = _normalized_url(url)
    host = parsed.hostname.lower()
    if host.startswith("www."):
        host = host[4:]
    path = urllib.parse.unquote(parsed.path)
    model_match = re.fullmatch(r"/models/([0-9]+)(?:/[^/]*)?/?", path)
    download_match = re.fullmatch(
        r"/api/download/models/([0-9]+)/?", path
    )
    query = urllib.parse.parse_qs(parsed.query)
    versions = query.get("modelVersionId", ())
    if len(versions) > 1:
        raise LauncherError("Civitai URL contains several modelVersionId values")
    version_id = 0
    if versions:
        try:
            version_id = int(versions[0])
        except ValueError:
            raise LauncherError("Civitai modelVersionId must be an integer")
    if download_match:
        path_version = int(download_match.group(1))
        if version_id and version_id != path_version:
            raise LauncherError("Civitai URL contains conflicting version IDs")
        return host, 0, path_version
    if not model_match:
        raise LauncherError(
            "Civitai import requires a model page or model download URL"
        )
    return host, int(model_match.group(1)), version_id


def civitai_version_choices(
    url: str, token: str = ""
) -> Tuple[Tuple[str, str], ...]:
    host, model_id, version_id = _civitai_identity(url)
    if version_id:
        return ((str(version_id), "version from URL"),)
    if not model_id:
        return ()
    model = _request_json(
        "https://{}/api/v1/models/{}".format(host, model_id),
        token,
        "Civitai",
    )
    versions = model.get("modelVersions")
    if not isinstance(versions, list):
        raise LauncherError("Civitai model has no version inventory")
    choices = []
    for value in versions:
        if not isinstance(value, dict) or not isinstance(value.get("id"), int):
            continue
        description = str(value.get("name") or "unnamed version")
        base = value.get("baseModel")
        if isinstance(base, str) and base:
            description = "{} — {}".format(description, base)
        choices.append((str(value["id"]), description))
    if not choices:
        raise LauncherError("Civitai model has no selectable versions")
    return tuple(choices)


def _exact_size_from_kib(value: object) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LauncherError("Civitai file is missing its exact size")
    size = float(value) * 1024.0
    rounded = round(size)
    if rounded <= 0 or abs(size - rounded) > 0.001:
        raise LauncherError("Civitai file size is not exact")
    return int(rounded)


def _safe_remote_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise LauncherError("remote file has no name")
    path = PurePosixPath(value)
    if path.name != value or value in (".", ".."):
        raise LauncherError("remote filename must not contain directories")
    return value


def _recognized_file(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith(_WEIGHT_SUFFIXES + (".gguf", ".json"))


def _civitai_file(value: Mapping[str, Any]) -> Optional[RemoteFile]:
    try:
        name = _safe_remote_name(value.get("name"))
    except LauncherError:
        return None
    hashes = value.get("hashes")
    if not isinstance(hashes, dict):
        return None
    digest = hashes.get("SHA256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        return None
    if not _recognized_file(name):
        return None
    identifier = value.get("id")
    if not isinstance(identifier, int) or identifier <= 0:
        return None
    download_url = value.get("downloadUrl")
    if not isinstance(download_url, str):
        return None
    return RemoteFile(
        identifier=str(identifier),
        name=name,
        size=_exact_size_from_kib(value.get("sizeKB")),
        sha256=digest.lower(),
        primary=value.get("primary") is True,
        download_url=download_url,
    )


def _discover_civitai(
    url: str, version_id: int, token: str
) -> RemoteDiscovery:
    host, model_id, url_version_id = _civitai_identity(url)
    if version_id and url_version_id and version_id != url_version_id:
        raise LauncherError("Civitai URL and --version select different versions")
    selected_version = version_id or url_version_id
    if not selected_version:
        raise LauncherError(
            "Civitai URL does not select a model version"
        )
    version = _request_json(
        "https://{}/api/v1/model-versions/{}".format(host, selected_version),
        token,
        "Civitai",
    )
    returned_version = version.get("id")
    returned_model = version.get("modelId")
    if returned_version != selected_version or not isinstance(returned_model, int):
        raise LauncherError("Civitai returned inconsistent version metadata")
    if model_id and returned_model != model_id:
        raise LauncherError(
            "Civitai modelVersionId does not belong to the URL's model"
        )
    model_id = returned_model
    model = _request_json(
        "https://{}/api/v1/models/{}".format(host, model_id),
        token,
        "Civitai",
    )
    values = version.get("files")
    files = tuple(
        item
        for item in (
            _civitai_file(value)
            for value in values
            if isinstance(value, dict)
        )
        if item is not None
    ) if isinstance(values, list) else ()
    if not files:
        raise LauncherError(
            "Civitai version has no supported file with exact size and SHA-256"
        )
    title = str(model.get("name") or "Civitai model")
    version_name = version.get("name")
    if isinstance(version_name, str) and version_name:
        title = "{} — {}".format(title, version_name)
    availability = version.get("availability")
    model_type = str(model.get("type") or "")
    source_url = "https://{}/models/{}?modelVersionId={}".format(
        host, model_id, selected_version
    )
    return RemoteDiscovery(
        provider="civitai",
        source_url=source_url,
        title=title,
        repository="{}/models/{}".format(host, model_id),
        revision=str(selected_version),
        files=files,
        declared_license="Civitai model-page permissions",
        model_type=model_type,
        provider_host=host,
        model_id=model_id,
        model_version_id=selected_version,
        requires_auth=(
            host == "civitai.red"
            or (
                isinstance(availability, str)
                and availability.lower() != "public"
            )
        ),
    )


def _huggingface_identity(
    url: str,
) -> Tuple[str, Optional[str], Optional[str]]:
    parsed = _normalized_url(url)
    parts = [
        urllib.parse.unquote(item)
        for item in parsed.path.split("/")
        if item
    ]
    if len(parts) < 2:
        raise LauncherError("Hugging Face URL must name a model repository")
    if parts[0] in ("datasets", "spaces"):
        raise LauncherError("only Hugging Face model repositories are supported")
    repository = "{}/{}".format(parts[0], parts[1])
    if len(parts) == 2:
        return repository, None, None
    if len(parts) < 5 or parts[2] not in ("blob", "resolve"):
        raise LauncherError(
            "Hugging Face import supports a model repository page or one "
            "blob/resolve file URL"
        )
    revision = parts[3]
    path = "/".join(parts[4:])
    if not revision or not path or ".." in PurePosixPath(path).parts:
        raise LauncherError("Hugging Face URL contains an unsafe file path")
    return repository, revision, path


def _discover_huggingface(url: str, token: str) -> RemoteDiscovery:
    repository, requested_revision, requested_path = _huggingface_identity(url)
    repository_path = urllib.parse.quote(repository, safe="/")
    if requested_revision is None:
        endpoint = "https://huggingface.co/api/models/{}?blobs=true".format(
            repository_path
        )
    else:
        endpoint = (
            "https://huggingface.co/api/models/{}/revision/{}?blobs=true"
        ).format(
            repository_path,
            urllib.parse.quote(requested_revision, safe=""),
        )
    metadata = _request_json(endpoint, token, "Hugging Face")
    revision = metadata.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(
        r"[0-9a-f]{40}", revision
    ):
        raise LauncherError(
            "Hugging Face did not resolve the URL to a full commit"
        )
    siblings = metadata.get("siblings")
    files = []
    if isinstance(siblings, list):
        for value in siblings:
            if not isinstance(value, dict):
                continue
            path = value.get("rfilename")
            if not isinstance(path, str) or not _recognized_file(path):
                continue
            if requested_path is not None and path != requested_path:
                continue
            lfs = value.get("lfs")
            if not isinstance(lfs, dict):
                continue
            size = lfs.get("size")
            digest = lfs.get("sha256")
            if (
                not isinstance(size, int)
                or size <= 0
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                continue
            files.append(
                RemoteFile(
                    identifier=path,
                    name=PurePosixPath(path).name,
                    size=size,
                    sha256=digest.lower(),
                    primary=requested_path == path,
                )
            )
    if requested_path is not None and not files:
        raise LauncherError(
            "Hugging Face file is not an LFS object with exact SHA-256 metadata"
        )
    if not files:
        raise LauncherError(
            "Hugging Face repository has no supported LFS model files"
        )
    card = metadata.get("cardData")
    declared_license = ""
    if isinstance(card, dict) and isinstance(card.get("license"), str):
        declared_license = card["license"]
    return RemoteDiscovery(
        provider="huggingface",
        source_url=(
            "https://huggingface.co/{}/tree/{}".format(repository, revision)
            if requested_path is None
            else "https://huggingface.co/{}/blob/{}/{}".format(
                repository, revision, requested_path
            )
        ),
        title=repository,
        repository=repository,
        revision=revision,
        files=tuple(files),
        declared_license=declared_license or "not declared by provider metadata",
        requires_auth=bool(metadata.get("private") or metadata.get("gated")),
    )


def discover_remote(
    url: str,
    *,
    version_id: int = 0,
    hf_token: str = "",
    civitai_token: str = "",
) -> RemoteDiscovery:
    provider = remote_provider(url)
    if provider == "civitai":
        return _discover_civitai(url, version_id, civitai_token)
    return _discover_huggingface(url, hf_token)


def automatic_file(discovery: RemoteDiscovery) -> Optional[RemoteFile]:
    if len(discovery.files) == 1:
        return discovery.files[0]
    primary = tuple(item for item in discovery.files if item.primary)
    return primary[0] if len(primary) == 1 else None


def select_file(discovery: RemoteDiscovery, selector: str) -> RemoteFile:
    matches = tuple(
        item
        for item in discovery.files
        if item.identifier == selector or item.name == selector
    )
    if len(matches) != 1:
        raise LauncherError(
            "remote file {!r} did not select exactly one file".format(selector)
        )
    return matches[0]


def compatible_kinds(file: RemoteFile) -> Tuple[ImportKind, ...]:
    lowered = file.name.lower()
    return tuple(
        kind
        for kind in IMPORT_KINDS.values()
        if lowered.endswith(kind.suffixes)
    )


def candidate_kinds(
    discovery: RemoteDiscovery, file: RemoteFile
) -> Tuple[ImportKind, ...]:
    """Narrow suffix-compatible destinations using provider metadata."""
    compatible = compatible_kinds(file)
    if (
        file.name.lower().endswith(".gguf")
        or discovery.provider != "civitai"
    ):
        return compatible
    model_type = discovery.model_type.lower().replace(" ", "")
    if model_type in _UNSUPPORTED_CIVITAI_TYPES:
        return ()
    identifiers = _CIVITAI_KIND_CANDIDATES.get(model_type)
    if identifiers is None:
        return compatible
    compatible_identifiers = {kind.identifier for kind in compatible}
    candidates = tuple(
        IMPORT_KINDS[identifier]
        for identifier in identifiers
        if identifier in compatible_identifiers
    )
    # A Civitai checkpoint must remain eligible for ComfyUI's checkpoint
    # loader. Otherwise the file suffix contradicts the provider category,
    # and silently treating it as a standalone UNet would be unsafe.
    if (
        model_type == "checkpoint"
        and "comfyui:checkpoint" not in compatible_identifiers
    ):
        return ()
    return candidates


def automatic_kind(
    discovery: RemoteDiscovery, file: RemoteFile
) -> Optional[ImportKind]:
    candidates = candidate_kinds(discovery, file)
    if len(candidates) == 1:
        return candidates[0]
    return None


def select_kind(identifier: str, file: RemoteFile) -> ImportKind:
    try:
        kind = IMPORT_KINDS[identifier]
    except KeyError:
        raise LauncherError(
            "unknown import type {!r}; choose {}".format(
                identifier, ", ".join(IMPORT_KINDS)
            )
        )
    if not file.name.lower().endswith(kind.suffixes):
        raise LauncherError(
            "{} cannot install file {!r}".format(kind.label, file.name)
        )
    return kind


def _slug(value: str, limit: int = 48) -> str:
    rendered = _SAFE_IDENTIFIER.sub("-", value.lower()).strip("._-")
    return (rendered[:limit].rstrip("._-") or "content")


def build_import_plan(
    discovery: RemoteDiscovery,
    file: RemoteFile,
    kind: ImportKind,
) -> RemoteImportPlan:
    if kind.identifier not in {
        item.identifier for item in compatible_kinds(file)
    }:
        raise LauncherError(
            "{} cannot install file {!r}".format(kind.label, file.name)
        )
    if discovery.provider == "civitai":
        identity = "civitai-v{}-f{}".format(
            discovery.model_version_id, file.identifier
        )
        source: Dict[str, object] = {
            "provider": "civitai",
            "host": discovery.provider_host,
            "model_id": discovery.model_id,
            "model_version_id": discovery.model_version_id,
            "filename": file.name,
            "download_url": file.download_url,
            "requires_auth": discovery.requires_auth,
        }
    else:
        identity = "hf-{}-{}".format(
            _slug(discovery.repository.replace("/", "-"), 32),
            file.sha256[:12],
        )
        source = {
            "repository": discovery.repository,
            "revision": discovery.revision,
            "path": file.identifier,
        }
    # Truncate only the provider identity. The destination kind must remain
    # complete because one hosted weight can validly be imported into more
    # than one ComfyUI category.
    artifact_identifier = "import-{}-{}".format(
        _slug(identity, 56),
        _slug(kind.identifier.replace(":", "-"), 32),
    )
    bundle_identifier = artifact_identifier
    destination = "{}/{}".format(
        kind.destination_prefix, _safe_remote_name(file.name)
    )
    warning = (
        "ROCmplete imported this remote file on request but did not "
        "independently verify that its declared permissions cover the "
        "hosted bytes."
    )
    pack: Mapping[str, object] = {
        "schema_version": 2,
        "artifacts": {
            artifact_identifier: {
                "description": "{} from {}".format(
                    file.name, discovery.title
                ),
                "source": source,
                "target": kind.target,
                "destination": destination,
                "size": file.size,
                "sha256": file.sha256,
                "license": {
                    "spdx": "NOASSERTION",
                    "status": "unverified",
                    "url": discovery.source_url,
                    "warning": warning,
                    "upstream_repository": discovery.repository,
                    "upstream_license": discovery.declared_license,
                    "upstream_license_url": discovery.source_url,
                },
            }
        },
        "bundles": {
            bundle_identifier: {
                "description": "{} imported from {}".format(
                    file.name, discovery.provider
                ),
                "application": kind.application,
                "artifacts": [artifact_identifier],
                "groups": [
                    "all",
                    "llama"
                    if kind.application == "llama-cpp"
                    else kind.application,
                ],
            }
        },
    }
    return RemoteImportPlan(
        discovery=discovery,
        file=file,
        kind=kind,
        artifact_identifier=artifact_identifier,
        bundle_identifier=bundle_identifier,
        destination=destination,
        pack=pack,
    )


def pack_bytes(plan: RemoteImportPlan) -> bytes:
    return (
        json.dumps(plan.pack, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def default_pack_path(plan: RemoteImportPlan) -> Path:
    return (
        PROJECT_ROOT
        / "local-content"
        / "imports"
        / "{}.json".format(plan.bundle_identifier)
    )


def save_pack(
    path: Path, plan: RemoteImportPlan, *, dry_run: bool = False
) -> bool:
    if path.suffix.lower() != ".json":
        raise LauncherError("import pack path must end in .json")
    contents = pack_bytes(plan)
    try:
        status = path.lstat()
    except FileNotFoundError:
        status = None
    except OSError as error:
        raise LauncherError("cannot inspect import pack {}: {}".format(
            path, error
        ))
    if status is not None:
        if path.is_symlink() or not path.is_file():
            raise LauncherError(
                "refusing unexpected import pack path: {}".format(path)
            )
        try:
            current = path.read_bytes()
        except OSError as error:
            raise LauncherError("cannot read import pack {}: {}".format(
                path, error
            ))
        if current != contents:
            raise LauncherError(
                "import pack already exists with different content: {}".format(
                    path
                )
            )
        return False
    if dry_run:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".{}.".format(path.name),
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(contents)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, str(path))
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    except OSError as error:
        raise LauncherError("cannot save import pack {}: {}".format(
            path, error
        ))
    return True
