"""Curated ROCmplete content catalog."""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Tuple, TypeVar

from .config import APPLICATIONS, GPU_PROFILES, LLAMA_BACKENDS
from .errors import LauncherError
from .project import PROJECT_ROOT

DEFAULT_CATALOG_PATH = PROJECT_ROOT / "catalog" / "catalog.json"
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]*")
_GGUF_ARCHITECTURE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CatalogEntry = TypeVar("_CatalogEntry")
SELECTOR_GROUPS = frozenset(
    (
        "all",
        "comfyui",
        "qwen",
        "wan",
        "ltx",
        "ltx-camera",
        "hunyuan",
        "krea",
        "llama",
        "dwarfstar",
    )
)


@dataclass(frozen=True)
class LicenseInfo:
    spdx: str
    status: str
    url: str
    warning: str = ""
    upstream_repository: str = ""
    upstream_license: str = ""
    upstream_license_url: str = ""


@dataclass(frozen=True)
class Agreement:
    identifier: str
    name: str
    url: str
    summary: str


@dataclass(frozen=True)
class ArtifactSource:
    repository: str
    revision: str
    path: str
    provider: str = "huggingface"
    model_id: int = 0
    model_version_id: int = 0
    requires_auth: bool = False
    provider_host: str = ""
    download_url: str = ""
    archive_member: str = ""
    archive_max_size: int = 0


@dataclass(frozen=True)
class Artifact:
    identifier: str
    description: str
    source: ArtifactSource
    destination: str
    size: int
    sha256: str
    license: LicenseInfo
    agreements: Tuple[str, ...] = ()
    target: str = "models"


@dataclass(frozen=True)
class Bundle:
    identifier: str
    description: str
    application: str
    artifacts: Tuple[str, ...]
    workflow: str = ""
    groups: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowPack:
    identifier: str
    description: str
    destination: str
    source_package: str
    source_version: str
    source_revision: str
    source_resource: str
    source_sha256: str
    rendered_sha256: str
    renderer: str
    license: str
    license_url: str


@dataclass(frozen=True)
class BenchmarkWorkflow:
    bundle: str
    resource: str
    sha256: str
    renderer: str = "identity"
    rendered_sha256: str = ""


@dataclass(frozen=True)
class LlamaPreset:
    identifier: str
    bundle: str
    artifact: str
    default_context: int
    speculative_type: str = ""
    draft_tokens: int = 0
    draft_tokens_by_backend: Mapping[str, int] = field(default_factory=dict)
    draft_artifact: str = ""
    context_override_architectures: Tuple[str, ...] = field(
        default_factory=tuple
    )
    jinja: bool = False
    agent_tools: bool = False
    reasoning_effort_budget: bool = False
    reasoning_preserve: bool = False
    chat_template: str = ""
    flash_attention: Mapping[str, str] = field(default_factory=dict)
    kv_cache: Mapping[str, str] = field(default_factory=dict)

    def draft_tokens_for_backend(self, backend: str) -> int:
        return self.draft_tokens_by_backend.get(backend, self.draft_tokens)


@dataclass(frozen=True)
class Catalog:
    agreements: Mapping[str, Agreement]
    artifacts: Mapping[str, Artifact]
    bundles: Mapping[str, Bundle]
    workflow_packs: Mapping[str, WorkflowPack]
    benchmarks: Mapping[str, BenchmarkWorkflow]
    llama_presets: Mapping[str, LlamaPreset] = field(default_factory=dict)

    def artifact(self, identifier: str) -> Artifact:
        try:
            return self.artifacts[identifier]
        except KeyError:
            raise LauncherError("unknown artifact {!r}".format(identifier))

    def agreement(self, identifier: str) -> Agreement:
        try:
            return self.agreements[identifier]
        except KeyError:
            raise LauncherError("unknown agreement {!r}".format(identifier))

    def bundle(self, identifier: str) -> Bundle:
        try:
            return self.bundles[identifier]
        except KeyError:
            raise LauncherError(
                "unknown bundle {!r}; use 'content list'".format(identifier)
            )

    def workflow(self, identifier: str) -> WorkflowPack:
        try:
            return self.workflow_packs[identifier]
        except KeyError:
            raise LauncherError(
                "unknown workflow {!r}; use 'workflows list'".format(identifier)
            )

    def benchmark(self, bundle: str) -> BenchmarkWorkflow:
        try:
            return self.benchmarks[bundle]
        except KeyError:
            raise LauncherError(
                "bundle {!r} has no managed benchmark".format(bundle)
            )

    def llama_preset(self, identifier: str) -> LlamaPreset:
        try:
            return self.llama_presets[identifier]
        except KeyError:
            raise LauncherError(
                "unknown llama.cpp preset {!r}; "
                "use 'content install llama-cpp qwen3.6 "
                "--dry-run'".format(identifier)
            )

    def bundle_artifacts(self, bundle: Bundle) -> Tuple[Artifact, ...]:
        return tuple(self.artifacts[item] for item in bundle.artifacts)

    def bundle_size(self, bundle: Bundle) -> int:
        return sum(
            {
                item.sha256: item.size
                for item in self.bundle_artifacts(bundle)
            }.values()
        )

    def bundle_agreements(self, bundle: Bundle) -> Tuple[Agreement, ...]:
        identifiers = []
        for artifact in self.bundle_artifacts(bundle):
            for identifier in artifact.agreements:
                if identifier not in identifiers:
                    identifiers.append(identifier)
        return tuple(self.agreements[item] for item in identifiers)


def _required_string(data: Mapping[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise LauncherError(
            "catalog field {} must be a non-empty string".format(field)
        )
    return value


def _optional_string(data: Mapping[str, object], field: str) -> str:
    value = data.get(field, "")
    if not isinstance(value, str):
        raise LauncherError("catalog field {} must be a string".format(field))
    return value


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise LauncherError("catalog field {} has an invalid identifier".format(field))
    return value


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LauncherError(
            "catalog field {} must be a non-empty string".format(field)
        )
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise LauncherError(
            "catalog field {} must be a safe relative path".format(field)
        )
    return value


def _https_url(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("https://"):
        raise LauncherError("catalog field {} must be an HTTPS URL".format(field))
    return value


def _civitai_download_url(
    value: object,
    host: str,
    model_version_id: int,
    field: str,
) -> str:
    if value in (None, ""):
        return "https://{}/api/download/models/{}".format(
            host, model_version_id
        )
    url = _https_url(value, field)
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.hostname != host
        or parsed.path
        != "/api/download/models/{}".format(model_version_id)
        or parsed.fragment
    ):
        raise LauncherError(
            "catalog field {} must be the selected Civitai version's "
            "download URL on {}".format(field, host)
        )
    return url


def _load_license(
    data: Mapping[str, object], field: str
) -> LicenseInfo:
    status = _required_string(data, "status")
    if status not in ("verified", "unverified"):
        raise LauncherError(
            "catalog field {}.status must be verified or unverified".format(field)
        )
    spdx = _required_string(data, "spdx")
    warning = _optional_string(data, "warning")
    upstream_repository = _optional_string(data, "upstream_repository")
    upstream_license = _optional_string(data, "upstream_license")
    upstream_license_url = _optional_string(data, "upstream_license_url")
    if status == "verified":
        if spdx == "NOASSERTION":
            raise LauncherError(
                "verified catalog license cannot use NOASSERTION"
            )
        if warning:
            raise LauncherError(
                "verified catalog license must not carry a risk warning"
            )
    else:
        if spdx != "NOASSERTION" or not warning:
            raise LauncherError(
                "unverified catalog license requires NOASSERTION and a warning"
            )
        if not upstream_repository or not upstream_license:
            raise LauncherError(
                "unverified catalog license requires upstream lineage"
            )
    return LicenseInfo(
        spdx=spdx,
        status=status,
        url=_https_url(data.get("url"), "{}.url".format(field)),
        warning=warning,
        upstream_repository=upstream_repository,
        upstream_license=upstream_license,
        upstream_license_url=(
            _https_url(
                upstream_license_url,
                "{}.upstream_license_url".format(field),
            )
            if upstream_license_url
            else ""
        ),
    )


def _load_artifact(identifier: str, data: Mapping[str, object]) -> Artifact:
    source_value = data.get("source")
    license_value = data.get("license")
    if not isinstance(source_value, dict) or not isinstance(license_value, dict):
        raise LauncherError(
            "artifact {} requires source and license objects".format(identifier)
        )
    provider = source_value.get("provider", "huggingface")
    if provider == "huggingface":
        revision = _required_string(source_value, "revision")
        if not _REVISION.fullmatch(revision):
            raise LauncherError("artifact revision must be a full commit hash")
        source = ArtifactSource(
            repository=_required_string(source_value, "repository"),
            revision=revision,
            path=_safe_relative_path(
                source_value.get("path"),
                "{}.source.path".format(identifier),
            ),
        )
    elif provider == "civitai":
        model_id = source_value.get("model_id")
        model_version_id = source_value.get("model_version_id")
        requires_auth = source_value.get("requires_auth", False)
        if not isinstance(model_id, int) or model_id <= 0:
            raise LauncherError(
                "Civitai artifact model_id must be a positive integer"
            )
        if (
            not isinstance(model_version_id, int)
            or model_version_id <= 0
        ):
            raise LauncherError(
                "Civitai artifact model_version_id must be a positive integer"
            )
        if not isinstance(requires_auth, bool):
            raise LauncherError(
                "Civitai artifact requires_auth must be a boolean"
            )
        provider_host = source_value.get("host", "civitai.com")
        if provider_host not in ("civitai.com", "civitai.red"):
            raise LauncherError(
                "Civitai artifact host must be civitai.com or civitai.red"
            )
        download_url = _civitai_download_url(
            source_value.get("download_url"),
            provider_host,
            model_version_id,
            "{}.source.download_url".format(identifier),
        )
        filename = _safe_relative_path(
            source_value.get("filename"),
            "{}.source.filename".format(identifier),
        )
        if PurePosixPath(filename).name != filename:
            raise LauncherError(
                "Civitai artifact filename must not contain directories"
            )
        archive_value = source_value.get("archive")
        archive_member = ""
        archive_max_size = 0
        if archive_value is not None:
            if not isinstance(archive_value, dict):
                raise LauncherError(
                    "Civitai artifact archive must be an object"
                )
            unknown = sorted(
                set(archive_value) - {"member", "max_size"}
            )
            if unknown:
                raise LauncherError(
                    "Civitai artifact archive has unsupported fields: "
                    "{}".format(", ".join(unknown))
                )
            archive_member = _safe_relative_path(
                archive_value.get("member"),
                "{}.source.archive.member".format(identifier),
            )
            archive_max_size = archive_value.get("max_size")
            if (
                not isinstance(archive_max_size, int)
                or archive_max_size <= 0
            ):
                raise LauncherError(
                    "Civitai artifact archive max_size must be positive"
                )
        source = ArtifactSource(
            repository="{}/models/{}".format(provider_host, model_id),
            revision=str(model_version_id),
            path=filename,
            provider="civitai",
            model_id=model_id,
            model_version_id=model_version_id,
            requires_auth=requires_auth,
            provider_host=provider_host,
            download_url=download_url,
            archive_member=archive_member,
            archive_max_size=archive_max_size,
        )
    else:
        raise LauncherError(
            "artifact {} has unsupported source provider {!r}".format(
                identifier, provider
            )
        )
    size = data.get("size")
    digest = data.get("sha256")
    if not isinstance(size, int) or size <= 0:
        raise LauncherError("artifact size must be a positive integer")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise LauncherError("artifact sha256 must be 64 lowercase hex digits")
    agreements_value = data.get("agreements", [])
    if not isinstance(agreements_value, list):
        raise LauncherError(
            "artifact {} agreements must be a list".format(identifier)
        )
    agreements = tuple(
        _identifier(item, "{}.agreements".format(identifier))
        for item in agreements_value
    )
    if len(set(agreements)) != len(agreements):
        raise LauncherError(
            "artifact {} contains duplicate agreements".format(identifier)
        )
    target = data.get("target", "models")
    if target not in (
        "models",
        "llama-models",
        "dwarfstar-models",
        "workflows",
    ):
        raise LauncherError(
            "artifact {} target must be models, llama-models, "
            "dwarfstar-models, or workflows".format(identifier)
        )
    destination = _safe_relative_path(
        data.get("destination"), "{}.destination".format(identifier)
    )
    if target == "workflows" and not destination.endswith(".json"):
        raise LauncherError(
            "workflow artifact {} destination must end in .json".format(
                identifier
            )
        )
    return Artifact(
        identifier=identifier,
        description=_required_string(data, "description"),
        source=source,
        destination=destination,
        size=size,
        sha256=digest,
        license=_load_license(
            license_value, "{}.license".format(identifier)
        ),
        agreements=agreements,
        target=target,
    )


def _expand_archive_collections(
    raw_artifacts: Mapping[str, object],
    raw_collections: Mapping[str, object],
) -> Dict[str, object]:
    """Expand compact, shared-archive declarations into ordinary artifacts."""
    artifacts = dict(raw_artifacts)
    collection_fields = {
        "source",
        "archive",
        "license",
        "agreements",
        "target",
        "members",
    }
    member_fields = {
        "description",
        "member",
        "destination",
        "target",
        "size",
        "sha256",
    }
    for raw_identifier, value in raw_collections.items():
        identifier = _identifier(
            raw_identifier, "archive collection identifier"
        )
        if not isinstance(value, dict):
            raise LauncherError(
                "archive collection {} must be an object".format(identifier)
            )
        unknown = sorted(set(value) - collection_fields)
        if unknown:
            raise LauncherError(
                "archive collection {} has unsupported fields: {}".format(
                    identifier, ", ".join(unknown)
                )
            )
        source = value.get("source")
        archive = value.get("archive")
        license_value = value.get("license")
        members = value.get("members")
        if not all(
            isinstance(item, dict)
            for item in (source, archive, license_value, members)
        ):
            raise LauncherError(
                "archive collection {} requires source, archive, license, "
                "and members objects".format(identifier)
            )
        if "archive" in source:
            raise LauncherError(
                "archive collection {} source must not contain archive".format(
                    identifier
                )
            )
        if not members:
            raise LauncherError(
                "archive collection {} must contain members".format(identifier)
            )
        unknown_archive = sorted(set(archive) - {"max_size"})
        if unknown_archive:
            raise LauncherError(
                "archive collection {} archive has unsupported fields: "
                "{}".format(identifier, ", ".join(unknown_archive))
            )
        agreements = value.get("agreements", [])
        target = value.get("target", "models")
        for raw_member_identifier, member in members.items():
            member_identifier = _identifier(
                raw_member_identifier,
                "{} member identifier".format(identifier),
            )
            if member_identifier in artifacts:
                raise LauncherError(
                    "duplicate artifact identifier {!r}".format(
                        member_identifier
                    )
                )
            if not isinstance(member, dict):
                raise LauncherError(
                    "archive collection {} member {} must be an object".format(
                        identifier, member_identifier
                    )
                )
            unknown = sorted(set(member) - member_fields)
            if unknown:
                raise LauncherError(
                    "archive collection {} member {} has unsupported fields: "
                    "{}".format(
                        identifier,
                        member_identifier,
                        ", ".join(unknown),
                    )
                )
            member_source = dict(source)
            member_source["archive"] = {
                "member": member.get("member"),
                "max_size": archive.get("max_size"),
            }
            artifacts[member_identifier] = {
                "description": member.get("description"),
                "source": member_source,
                "destination": member.get("destination"),
                "target": member.get("target", target),
                "size": member.get("size"),
                "sha256": member.get("sha256"),
                "license": license_value,
                "agreements": agreements,
            }
    return artifacts


def _load_agreement(identifier: str, data: Mapping[str, object]) -> Agreement:
    return Agreement(
        identifier=identifier,
        name=_required_string(data, "name"),
        url=_https_url(data.get("url"), "{}.url".format(identifier)),
        summary=_required_string(data, "summary"),
    )


def _load_bundle(identifier: str, data: Mapping[str, object]) -> Bundle:
    artifacts_value = data.get("artifacts", [])
    groups_value = data.get("groups")
    if "trees" in data:
        raise LauncherError(
            "bundle {} uses the removed trees field".format(identifier)
        )
    if not isinstance(artifacts_value, list):
        raise LauncherError("bundle {} artifacts must be a list".format(identifier))
    if not artifacts_value:
        raise LauncherError("bundle {} must contain artifacts".format(identifier))
    if not isinstance(groups_value, list) or not groups_value:
        raise LauncherError("bundle {} must contain selector groups".format(identifier))
    artifacts = tuple(
        _identifier(item, "{}.artifacts".format(identifier))
        for item in artifacts_value
    )
    groups = tuple(
        _identifier(item, "{}.groups".format(identifier))
        for item in groups_value
    )
    if len(set(artifacts)) != len(artifacts):
        raise LauncherError(
            "bundle {} contains duplicate artifacts".format(identifier)
        )
    if len(set(groups)) != len(groups):
        raise LauncherError("bundle {} contains duplicate groups".format(identifier))
    unknown_groups = sorted(set(groups) - SELECTOR_GROUPS)
    if unknown_groups:
        raise LauncherError(
            "bundle {} contains unknown selector groups: {}".format(
                identifier, ", ".join(unknown_groups)
            )
        )
    if "all" not in groups:
        raise LauncherError(
            "bundle {} must belong to the all selector group".format(identifier)
        )
    application = _identifier(data.get("application"), "application")
    if application not in APPLICATIONS:
        raise LauncherError(
            "bundle {} references unknown application {}".format(
                identifier, application
            )
        )
    return Bundle(
        identifier=identifier,
        description=_required_string(data, "description"),
        application=application,
        artifacts=artifacts,
        groups=groups,
        workflow=(
            _identifier(data.get("workflow"), "workflow")
            if data.get("workflow")
            else ""
        ),
    )


def _load_workflow(
    identifier: str, data: Mapping[str, object]
) -> WorkflowPack:
    source_digest = _required_string(data, "source_sha256")
    rendered_digest = _required_string(data, "rendered_sha256")
    if not _SHA256.fullmatch(source_digest):
        raise LauncherError("workflow source_sha256 must be 64 lowercase hex digits")
    if not _SHA256.fullmatch(rendered_digest):
        raise LauncherError(
            "workflow rendered_sha256 must be 64 lowercase hex digits"
        )
    source_revision = _required_string(data, "source_revision")
    if not _REVISION.fullmatch(source_revision):
        raise LauncherError(
            "workflow source_revision must be a full commit hash"
        )
    return WorkflowPack(
        identifier=identifier,
        description=_required_string(data, "description"),
        destination=_safe_relative_path(
            data.get("destination"), "{}.destination".format(identifier)
        ),
        source_package=_required_string(data, "source_package"),
        source_version=_required_string(data, "source_version"),
        source_revision=source_revision,
        source_resource=_safe_relative_path(
            data.get("source_resource"), "source_resource"
        ),
        source_sha256=source_digest,
        rendered_sha256=rendered_digest,
        renderer=_required_string(data, "renderer"),
        license=_required_string(data, "license"),
        license_url=_https_url(
            data.get("license_url"), "{}.license_url".format(identifier)
        ),
    )


def _load_benchmark(
    identifier: str, data: Mapping[str, object]
) -> BenchmarkWorkflow:
    digest = _required_string(data, "sha256")
    if not _SHA256.fullmatch(digest):
        raise LauncherError(
            "benchmark sha256 must be 64 lowercase hex digits"
        )
    renderer = data.get("renderer", "identity")
    if renderer not in (
        "identity",
        "hunyuan-i2v-480p-step-distilled",
        "hunyuan-t2v-480p-cfg-distilled",
    ):
        raise LauncherError(
            "benchmark {} has an unknown renderer".format(identifier)
        )
    rendered_digest = data.get("rendered_sha256", digest)
    if not isinstance(rendered_digest, str) or not _SHA256.fullmatch(
        rendered_digest
    ):
        raise LauncherError(
            "benchmark rendered_sha256 must be 64 lowercase hex digits"
        )
    return BenchmarkWorkflow(
        bundle=identifier,
        resource=_safe_relative_path(
            data.get("resource"), "{}.resource".format(identifier)
        ),
        sha256=digest,
        renderer=renderer,
        rendered_sha256=rendered_digest,
    )


def _load_llama_preset(
    identifier: str, data: Mapping[str, object]
) -> LlamaPreset:
    supported_fields = {
        "bundle",
        "artifact",
        "default_context",
        "speculative_type",
        "draft_tokens",
        "draft_tokens_by_backend",
        "draft_artifact",
        "context_override_architectures",
        "jinja",
        "agent_tools",
        "reasoning_effort_budget",
        "reasoning_preserve",
        "chat_template",
        "flash_attention",
        "kv_cache",
    }
    unsupported_fields = sorted(set(data) - supported_fields)
    if unsupported_fields:
        raise LauncherError(
            "llama.cpp preset {} has unsupported fields: {}".format(
                identifier, ", ".join(unsupported_fields)
            )
        )
    default_context = data.get("default_context")
    if (
        not isinstance(default_context, int)
        or isinstance(default_context, bool)
        or default_context <= 0
    ):
        raise LauncherError(
            "llama.cpp preset {} default_context must be a positive "
            "integer".format(identifier)
        )
    speculative_type = data.get("speculative_type", "")
    if not isinstance(speculative_type, str) or speculative_type not in (
        "",
        "draft-mtp",
        "draft-dflash",
    ):
        raise LauncherError(
            "llama.cpp preset {} speculative_type must be empty, "
            "draft-mtp, or draft-dflash".format(identifier)
        )
    draft_tokens = data.get("draft_tokens", 0)
    if (
        not isinstance(draft_tokens, int)
        or isinstance(draft_tokens, bool)
        or not 0 <= draft_tokens <= 15
    ):
        raise LauncherError(
            "llama.cpp preset {} draft_tokens must be an integer between "
            "0 and 15".format(identifier)
        )
    if not speculative_type and draft_tokens:
        raise LauncherError(
            "llama.cpp preset {} draft_tokens requires speculative_type".format(
                identifier
            )
        )
    if speculative_type and draft_tokens == 0:
        raise LauncherError(
            "llama.cpp preset {} speculative_type requires positive "
            "draft_tokens".format(identifier)
        )
    if speculative_type == "draft-mtp" and draft_tokens > 8:
        raise LauncherError(
            "llama.cpp preset {} draft-mtp draft_tokens must be between "
            "1 and 8".format(identifier)
        )
    draft_tokens_by_backend_value = data.get(
        "draft_tokens_by_backend", {}
    )
    if not isinstance(draft_tokens_by_backend_value, dict):
        raise LauncherError(
            "llama.cpp preset {} draft_tokens_by_backend must be an "
            "object".format(identifier)
        )
    draft_tokens_by_backend: Dict[str, int] = {}
    for raw_backend, value in draft_tokens_by_backend_value.items():
        backend = _identifier(
            raw_backend,
            "{}.draft_tokens_by_backend backend".format(identifier),
        )
        if backend not in LLAMA_BACKENDS:
            raise LauncherError(
                "llama.cpp preset {} draft_tokens_by_backend key must be "
                "one of {}".format(identifier, ", ".join(LLAMA_BACKENDS))
            )
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= 15
        ):
            raise LauncherError(
                "llama.cpp preset {} draft_tokens_by_backend value must "
                "be an integer between 1 and 15".format(identifier)
            )
        if speculative_type == "draft-mtp" and value > 8:
            raise LauncherError(
                "llama.cpp preset {} draft-mtp backend draft tokens must "
                "be between 1 and 8".format(identifier)
            )
        draft_tokens_by_backend[backend] = value
    if draft_tokens_by_backend and not speculative_type:
        raise LauncherError(
            "llama.cpp preset {} draft_tokens_by_backend requires "
            "speculative_type".format(identifier)
        )
    draft_artifact = data.get("draft_artifact", "")
    if draft_artifact:
        draft_artifact = _identifier(
            draft_artifact, "{}.draft_artifact".format(identifier)
        )
        if not speculative_type:
            raise LauncherError(
                "llama.cpp preset {} draft_artifact requires positive "
                "draft_tokens and speculative_type".format(identifier)
            )
    elif not isinstance(draft_artifact, str):
        raise LauncherError(
            "llama.cpp preset {} draft_artifact must be a string".format(
                identifier
            )
        )
    if speculative_type == "draft-dflash" and not draft_artifact:
        raise LauncherError(
            "llama.cpp preset {} draft-dflash requires draft_artifact".format(
                identifier
            )
        )
    context_override_value = data.get(
        "context_override_architectures", []
    )
    if not isinstance(context_override_value, list):
        raise LauncherError(
            "llama.cpp preset {} context_override_architectures must be "
            "an array".format(identifier)
        )
    context_override_architectures = []
    for value in context_override_value:
        if not isinstance(value, str) or not _GGUF_ARCHITECTURE.fullmatch(
            value
        ):
            raise LauncherError(
                "llama.cpp preset {} context_override_architectures must "
                "contain GGUF architecture names".format(identifier)
            )
        if value in context_override_architectures:
            raise LauncherError(
                "llama.cpp preset {} context_override_architectures must "
                "not contain duplicates".format(identifier)
            )
        context_override_architectures.append(value)
    jinja = data.get("jinja", False)
    if not isinstance(jinja, bool):
        raise LauncherError(
            "llama.cpp preset {} jinja must be a boolean".format(identifier)
        )
    chat_template = data.get("chat_template", "")
    # Keep this closed set aligned with the image files and the entrypoint's
    # independent validation; the catalog must not become a path loader.
    if not isinstance(chat_template, str) or chat_template not in (
        "",
        "kat-coder-v2.5",
        "muse-glimmer-atem",
        "qwen3-0.6b",
        "qwen3.6",
        "translategemma-manual",
    ):
        raise LauncherError(
            "llama.cpp preset {} chat_template must be empty, "
            "kat-coder-v2.5, muse-glimmer-atem, qwen3-0.6b, qwen3.6, "
            "or translategemma-manual".format(identifier)
        )
    if chat_template and jinja:
        raise LauncherError(
            "llama.cpp preset {} custom chat_template already enables "
            "Jinja".format(identifier)
        )
    agent_tools = data.get("agent_tools", False)
    if not isinstance(agent_tools, bool):
        raise LauncherError(
            "llama.cpp preset {} agent_tools must be a boolean".format(
                identifier
            )
        )
    if agent_tools and (
        not (jinja or chat_template) or default_context < 16384
    ):
        raise LauncherError(
            "llama.cpp preset {} agent_tools requires Jinja and at least "
            "16384 context tokens".format(identifier)
        )
    reasoning_effort_budget = data.get(
        "reasoning_effort_budget", False
    )
    if not isinstance(reasoning_effort_budget, bool):
        raise LauncherError(
            "llama.cpp preset {} reasoning_effort_budget must be a "
            "boolean".format(identifier)
        )
    if reasoning_effort_budget and not agent_tools:
        raise LauncherError(
            "llama.cpp preset {} reasoning_effort_budget requires "
            "agent_tools".format(identifier)
        )
    reasoning_preserve = data.get("reasoning_preserve", False)
    if not isinstance(reasoning_preserve, bool):
        raise LauncherError(
            "llama.cpp preset {} reasoning_preserve must be a "
            "boolean".format(identifier)
        )
    if reasoning_preserve and not agent_tools:
        raise LauncherError(
            "llama.cpp preset {} reasoning_preserve requires "
            "agent_tools".format(identifier)
        )
    flash_attention_value = data.get("flash_attention", {})
    if not isinstance(flash_attention_value, dict):
        raise LauncherError(
            "llama.cpp preset {} flash_attention must be an object".format(
                identifier
            )
        )
    flash_attention: Dict[str, str] = {}
    for raw_profile, value in flash_attention_value.items():
        profile = _identifier(
            raw_profile, "{}.flash_attention profile".format(identifier)
        )
        if profile not in GPU_PROFILES:
            raise LauncherError(
                "llama.cpp preset {} flash_attention profile must be one of "
                "{}".format(identifier, ", ".join(GPU_PROFILES))
            )
        if value not in ("on", "off", "auto"):
            raise LauncherError(
                "llama.cpp preset {} flash_attention value must be on, "
                "off, or auto".format(identifier)
            )
        flash_attention[profile] = value
    kv_cache_value = data.get("kv_cache", {})
    if not isinstance(kv_cache_value, dict):
        raise LauncherError(
            "llama.cpp preset {} kv_cache must be an object".format(
                identifier
            )
        )
    kv_cache: Dict[str, str] = {}
    for raw_profile, value in kv_cache_value.items():
        profile = _identifier(
            raw_profile, "{}.kv_cache profile".format(identifier)
        )
        if profile not in GPU_PROFILES:
            raise LauncherError(
                "llama.cpp preset {} kv_cache profile must be one of "
                "{}".format(identifier, ", ".join(GPU_PROFILES))
            )
        if value not in ("f16", "q8_0", "q4_0"):
            raise LauncherError(
                "llama.cpp preset {} kv_cache value must be f16, q8_0, "
                "or q4_0".format(identifier)
            )
        if value != "f16" and flash_attention.get(profile) != "on":
            raise LauncherError(
                "llama.cpp preset {} quantized kv_cache for {} requires "
                "flash_attention on".format(identifier, profile)
            )
        kv_cache[profile] = value
    return LlamaPreset(
        identifier=identifier,
        bundle=_identifier(data.get("bundle"), "{}.bundle".format(identifier)),
        artifact=_identifier(
            data.get("artifact"), "{}.artifact".format(identifier)
        ),
        default_context=default_context,
        speculative_type=speculative_type,
        draft_tokens=draft_tokens,
        draft_tokens_by_backend=draft_tokens_by_backend,
        draft_artifact=draft_artifact,
        context_override_architectures=tuple(
            context_override_architectures
        ),
        jinja=jinja,
        agent_tools=agent_tools,
        reasoning_effort_budget=reasoning_effort_budget,
        reasoning_preserve=reasoning_preserve,
        chat_template=chat_template,
        flash_attention=flash_attention,
        kv_cache=kv_cache,
    )


def _read_json_object(path: Path, description: str) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(
            "cannot load {} {}: {}".format(description, path, error)
        )
    if not isinstance(raw, dict):
        raise LauncherError("invalid {} {}".format(description, path))
    return raw


def _load_catalog_entries(
    raw_agreements: Mapping[str, object],
    raw_artifacts: Mapping[str, object],
    raw_bundles: Mapping[str, object],
) -> Tuple[
    Dict[str, Agreement],
    Dict[str, Artifact],
    Dict[str, Bundle],
]:
    agreements: Dict[str, Agreement] = {}
    artifacts: Dict[str, Artifact] = {}
    bundles: Dict[str, Bundle] = {}
    for raw_identifier, data in raw_agreements.items():
        identifier = _identifier(raw_identifier, "agreement identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid agreement entry")
        agreements[identifier] = _load_agreement(identifier, data)
    for raw_identifier, data in raw_artifacts.items():
        identifier = _identifier(raw_identifier, "artifact identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid artifact entry")
        artifacts[identifier] = _load_artifact(identifier, data)
    for raw_identifier, data in raw_bundles.items():
        identifier = _identifier(raw_identifier, "bundle identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid bundle entry")
        bundles[identifier] = _load_bundle(identifier, data)
    return agreements, artifacts, bundles


def _validate_catalog(catalog: Catalog) -> None:
    destinations: Dict[Tuple[str, str], str] = {}
    blob_sizes: Dict[str, int] = {}
    for identifier, artifact in catalog.artifacts.items():
        missing_agreements = [
            item
            for item in artifact.agreements
            if item not in catalog.agreements
        ]
        if missing_agreements:
            raise LauncherError(
                "artifact {} references unknown agreements: {}".format(
                    identifier, ", ".join(missing_agreements)
                )
            )
        destination_key = (artifact.target, artifact.destination)
        previous = destinations.get(destination_key)
        if previous is not None:
            raise LauncherError(
                "artifacts {} and {} share destination {}".format(
                    previous, identifier, artifact.destination
                )
            )
        destinations[destination_key] = identifier
        previous_size = blob_sizes.get(artifact.sha256)
        if previous_size is not None and previous_size != artifact.size:
            raise LauncherError(
                "SHA-256 {} has inconsistent sizes".format(artifact.sha256)
            )
        blob_sizes[artifact.sha256] = artifact.size
    for identifier, bundle in catalog.bundles.items():
        missing = [
            item for item in bundle.artifacts if item not in catalog.artifacts
        ]
        if missing:
            raise LauncherError(
                "bundle {} references unknown artifacts: {}".format(
                    identifier, ", ".join(missing)
                )
            )
        if (
            bundle.workflow
            and bundle.workflow not in catalog.workflow_packs
        ):
            raise LauncherError(
                "bundle {} references unknown workflow".format(identifier)
            )
        if bundle.application == "dwarfstar":
            if bundle.workflow or len(bundle.artifacts) != 1:
                raise LauncherError(
                    "DwarfStar bundle {} must contain exactly one direct "
                    "model artifact".format(identifier)
                )
            artifact = catalog.artifacts[bundle.artifacts[0]]
            if (
                artifact.target != "dwarfstar-models"
                or not artifact.destination.lower().endswith(".gguf")
            ):
                raise LauncherError(
                    "DwarfStar bundle {} must reference one dwarfstar-models "
                    ".gguf artifact".format(identifier)
                )
    for identifier in catalog.benchmarks:
        if identifier not in catalog.bundles:
            raise LauncherError(
                "benchmark references unknown bundle {}".format(identifier)
            )
    missing_benchmarks = [
        identifier
        for identifier, bundle in catalog.bundles.items()
        if bundle.workflow and identifier not in catalog.benchmarks
    ]
    if missing_benchmarks:
        raise LauncherError(
            "bundles without managed benchmarks: {}".format(
                ", ".join(missing_benchmarks)
            )
        )
    for identifier, preset in catalog.llama_presets.items():
        bundle = catalog.bundles.get(preset.bundle)
        if bundle is None:
            raise LauncherError(
                "llama.cpp preset {} references unknown bundle {}".format(
                    identifier, preset.bundle
                )
            )
        if bundle.application != "llama-cpp":
            raise LauncherError(
                "llama.cpp preset {} bundle must use llama-cpp".format(
                    identifier
                )
            )
        if preset.artifact not in bundle.artifacts:
            raise LauncherError(
                "llama.cpp preset {} artifact is not in bundle {}".format(
                    identifier, preset.bundle
                )
            )
        artifact = catalog.artifacts.get(preset.artifact)
        if artifact is None or artifact.target != "llama-models":
            raise LauncherError(
                "llama.cpp preset {} must reference a llama-models artifact".format(
                    identifier
                )
            )
        if not artifact.destination.lower().endswith(".gguf"):
            raise LauncherError(
                "llama.cpp preset {} artifact must install a .gguf file".format(
                    identifier
                )
            )
        if preset.draft_artifact:
            if preset.draft_artifact == preset.artifact:
                raise LauncherError(
                    "llama.cpp preset {} draft artifact must differ from "
                    "the target artifact".format(identifier)
                )
            if preset.draft_artifact not in bundle.artifacts:
                raise LauncherError(
                    "llama.cpp preset {} draft artifact is not in bundle "
                    "{}".format(identifier, preset.bundle)
                )
            draft = catalog.artifacts.get(preset.draft_artifact)
            if draft is None or draft.target != "llama-models":
                raise LauncherError(
                    "llama.cpp preset {} must reference a llama-models "
                    "draft artifact".format(identifier)
                )
            if not draft.destination.lower().endswith(".gguf"):
                raise LauncherError(
                    "llama.cpp preset {} draft artifact must install a "
                    ".gguf file".format(identifier)
                )


def load_catalog(path: Path = DEFAULT_CATALOG_PATH) -> Catalog:
    raw = _read_json_object(path, "catalog")
    if not isinstance(raw, dict) or raw.get("schema_version") != 21:
        raise LauncherError("unsupported or invalid catalog schema")
    allowed = {
        "schema_version",
        "agreements",
        "artifacts",
        "archive_collections",
        "bundles",
        "workflows",
        "benchmarks",
        "llama_presets",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise LauncherError(
            "catalog contains unsupported collections: {}".format(
                ", ".join(unknown)
            )
        )
    collections = (
        raw.get("agreements"),
        raw.get("artifacts"),
        raw.get("archive_collections"),
        raw.get("bundles"),
        raw.get("workflows"),
        raw.get("benchmarks"),
        raw.get("llama_presets"),
    )
    if not all(isinstance(item, dict) for item in collections):
        raise LauncherError("catalog collections must be objects")
    (
        raw_agreements,
        raw_artifacts,
        raw_archive_collections,
        raw_bundles,
        raw_workflows,
        raw_benchmarks,
        raw_llama_presets,
    ) = collections
    raw_artifacts = _expand_archive_collections(
        raw_artifacts, raw_archive_collections
    )

    agreements, artifacts, bundles = _load_catalog_entries(
        raw_agreements,
        raw_artifacts,
        raw_bundles,
    )
    workflows: Dict[str, WorkflowPack] = {}
    benchmarks: Dict[str, BenchmarkWorkflow] = {}
    llama_presets: Dict[str, LlamaPreset] = {}
    for raw_identifier, data in raw_workflows.items():
        identifier = _identifier(raw_identifier, "workflow identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid workflow entry")
        workflows[identifier] = _load_workflow(identifier, data)
    for raw_identifier, data in raw_benchmarks.items():
        identifier = _identifier(raw_identifier, "benchmark bundle identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid benchmark entry")
        benchmarks[identifier] = _load_benchmark(identifier, data)
    for raw_identifier, data in raw_llama_presets.items():
        identifier = _identifier(raw_identifier, "llama.cpp preset identifier")
        if not isinstance(data, dict):
            raise LauncherError("invalid llama.cpp preset entry")
        llama_presets[identifier] = _load_llama_preset(identifier, data)
    catalog = Catalog(
        agreements=agreements,
        artifacts=artifacts,
        bundles=bundles,
        workflow_packs=workflows,
        benchmarks=benchmarks,
        llama_presets=llama_presets,
    )
    _validate_catalog(catalog)
    return catalog


def _load_content_pack(path: Path) -> Catalog:
    raw = _read_json_object(path, "content pack")
    schema_version = raw.get("schema_version")
    if schema_version != 2:
        raise LauncherError(
            "unsupported or invalid content pack schema in {}".format(path)
        )
    allowed = {"schema_version", "agreements", "artifacts", "bundles"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise LauncherError(
            "content pack {} contains unsupported collections: {}".format(
                path, ", ".join(unknown)
            )
        )
    collections = tuple(
        raw.get(name, {}) for name in ("agreements", "artifacts", "bundles")
    )
    if not all(isinstance(item, dict) for item in collections):
        raise LauncherError("content pack collections must be objects")
    agreements, artifacts, bundles = _load_catalog_entries(
        collections[0], collections[1], collections[2]
    )
    if not bundles:
        raise LauncherError(
            "content pack {} must declare at least one bundle".format(path)
        )
    workflow_bundles = [
        identifier
        for identifier, bundle in bundles.items()
        if bundle.workflow
    ]
    if workflow_bundles:
        raise LauncherError(
            "content packs cannot define workflow-backed bundles: {}".format(
                ", ".join(workflow_bundles)
            )
        )
    return Catalog(
        agreements=agreements,
        artifacts=artifacts,
        bundles=bundles,
        workflow_packs={},
        benchmarks={},
        llama_presets={},
    )


def _merge_pack_entries(
    existing: Mapping[str, _CatalogEntry],
    additions: Mapping[str, _CatalogEntry],
    collection: str,
    path: Path,
) -> Dict[str, _CatalogEntry]:
    merged = dict(existing)
    for identifier, value in additions.items():
        if identifier in merged:
            raise LauncherError(
                "content pack {} {} {!r} conflicts with an existing "
                "definition".format(path, collection, identifier)
            )
        merged[identifier] = value
    return merged


def load_content_packs(
    catalog: Catalog, paths: Iterable[Path]
) -> Tuple[Catalog, Tuple[str, ...]]:
    agreements = dict(catalog.agreements)
    artifacts = dict(catalog.artifacts)
    bundles = dict(catalog.bundles)
    selected = []
    for path in paths:
        pack = _load_content_pack(path)
        agreements = _merge_pack_entries(
            agreements, pack.agreements, "agreement", path
        )
        artifacts = _merge_pack_entries(
            artifacts, pack.artifacts, "artifact", path
        )
        bundles = _merge_pack_entries(
            bundles, pack.bundles, "bundle", path
        )
        selected.extend(pack.bundles)
    merged = Catalog(
        agreements=agreements,
        artifacts=artifacts,
        bundles=bundles,
        workflow_packs=catalog.workflow_packs,
        benchmarks=catalog.benchmarks,
        llama_presets=catalog.llama_presets,
    )
    _validate_catalog(merged)
    return merged, tuple(selected)
