"""Checkpointed, hardware-bound ROCmplete smoke acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import re
import stat
import struct
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from . import podman
from .benchmark import BenchmarkOptions, run_benchmark
from .bundles import artifact_path
from .catalog import Bundle, Catalog
from .config import (
    APPLICATIONS,
    ROCM_BASE_IMAGE,
)
from .errors import LauncherError
from .hardware_profiles import ARCHITECTURE_PROFILES
from .layout import StorageLayout
from .llama_benchmark import run_llama_benchmark
from .project import PROJECT_ROOT
from .runtime.diagnostic import (
    cpu_isolation_diagnostic_command,
    gpu_diagnostic_command,
    parse_gpu_diagnostic_output,
)
from .runtime.dwarfstar import DwarfStarOptions, dwarfstar_command
from .runtime.llama import LlamaBenchmarkOptions, llama_benchmark_command


SCHEMA_VERSION = 3
POLICY_VERSION = 5
CASE_STATUSES = (
    "pending",
    "running",
    "pass",
    "fail",
    "blocked",
    "n/p",
    "interrupted",
)
RESULT_STATUSES = ("pass", "fail", "blocked")
SUITE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_APPLICATIONS = tuple(APPLICATIONS)
SOURCE_IDENTITY_PATHS = (
    ".containerignore",
    "Containerfile",
    "rocmplete",
    "applications",
    "catalog",
    "containers",
    "src",
)


@dataclass(frozen=True)
class AcceptanceCase:
    identifier: str
    description: str
    application: Optional[str]
    bundle: str = ""
    visual: bool = False
    review_criteria: Tuple[str, ...] = ()


IMAGE_REVIEW_CRITERIA = (
    "The PNG opens and shows a recognizable red cube on a blue table.",
    "There is no blank image, random noise, severe tiling, or corruption.",
    "Composition, sharpness, and aesthetic quality are not graded.",
)
VIDEO_REVIEW_CRITERIA = (
    "The MP4 plays and shows a recognizable red cube on a blue table.",
    "There is visible temporal change; some blur in this five-frame smoke "
    "output is acceptable.",
    "Frames do not collapse into black output, random noise, or corruption.",
)


SMOKE_CASES = (
    AcceptanceCase(
        "host-gpu",
        "GPU operation and exact device isolation",
        None,
    ),
    AcceptanceCase(
        "comfyui-image",
        "ComfyUI Qwen Image FP8 Lightning generation",
        "comfyui",
        "qwen-image-2512-fp8-lightning",
        visual=True,
        review_criteria=IMAGE_REVIEW_CRITERIA,
    ),
    AcceptanceCase(
        "comfyui-video",
        "ComfyUI Wan 2.2 FP8 Lightning five-frame generation",
        "comfyui",
        "wan-2.2-t2v-14b-fp8-lightning",
        visual=True,
        review_criteria=VIDEO_REVIEW_CRITERIA,
    ),
    AcceptanceCase(
        "llama-cpp",
        "llama.cpp Qwen3 0.6B GPU offload benchmark",
        "llama-cpp",
        "llama-qwen3-0.6b-q8-0",
    ),
    AcceptanceCase(
        "dwarfstar",
        "DwarfStar DeepSeek V4 Flash 0731 direct-answer generation",
        "dwarfstar",
        "dwarfstar-deepseek-v4-flash-0731-iq2xxs",
    ),
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _fallback_source_identity() -> str:
    digest = hashlib.sha256()
    paths = []
    for name in SOURCE_IDENTITY_PATHS:
        candidate = PROJECT_ROOT / name
        if candidate.is_dir():
            paths.extend(
                path
                for path in candidate.rglob("*")
                if "__pycache__" not in path.parts
                and path.suffix not in (".pyc", ".pyo")
            )
        else:
            paths.append(candidate)
    try:
        for path in sorted(paths):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_symlink():
                value = os.readlink(str(path)).encode(
                    "utf-8", "surrogateescape"
                )
                kind = b"symlink"
            elif path.is_file():
                value = path.read_bytes()
                kind = b"file"
            else:
                continue
            digest.update(kind + b"\0")
            digest.update(relative.encode("utf-8", "surrogateescape"))
            digest.update(b"\0" + value + b"\0")
    except OSError:
        return "unknown"
    return "source-tree-{}".format(digest.hexdigest())


def source_identity() -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return _fallback_source_identity()
    value = revision.stdout.strip()
    if (
        revision.returncode != 0
        or len(value) not in (40, 64)
        or any(
            character not in b"0123456789abcdef"
            for character in value
        )
    ):
        return _fallback_source_identity()

    pathspec = ("--",) + SOURCE_IDENTITY_PATHS
    try:
        difference = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "diff",
                "--binary",
                "HEAD",
                *pathspec,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        untracked = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                *pathspec,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return _fallback_source_identity()
    if difference.returncode != 0 or untracked.returncode != 0:
        return _fallback_source_identity()
    if not difference.stdout and not untracked.stdout:
        return value.decode("ascii")

    digest = hashlib.sha256()
    digest.update(b"revision\0" + value + b"\0")
    digest.update(b"tracked-diff\0" + difference.stdout + b"\0")
    try:
        for encoded in sorted(
            item for item in untracked.stdout.split(b"\0") if item
        ):
            relative = Path(
                encoded.decode("utf-8", "surrogateescape")
            )
            if relative.is_absolute() or ".." in relative.parts:
                continue
            path = PROJECT_ROOT / relative
            if path.is_symlink():
                contents = os.readlink(str(path)).encode(
                    "utf-8", "surrogateescape"
                )
                kind = b"untracked-symlink"
            elif path.is_file():
                contents = path.read_bytes()
                kind = b"untracked-file"
            else:
                continue
            digest.update(kind + b"\0" + encoded + b"\0")
            digest.update(contents + b"\0")
    except OSError:
        return _fallback_source_identity()
    return "{}-dirty-{}".format(
        value.decode("ascii"), digest.hexdigest()[:16]
    )


def selected_cases(
    profile: str, applications: Sequence[str] = ()
) -> Tuple[Tuple[AcceptanceCase, str], ...]:
    explicit_applications = bool(applications)
    requested = frozenset(applications or DEFAULT_APPLICATIONS)
    cases = []
    for case in SMOKE_CASES:
        if case.application is not None and case.application not in requested:
            continue
        reason = ""
        if (
            case.application == "dwarfstar"
            and profile not in ("auto", "strix-halo")
            and not explicit_applications
        ):
            reason = (
                "DwarfStar is opt-in outside Strix Halo; select "
                "--application dwarfstar after checking memory capacity"
            )
        cases.append((case, reason))
    return tuple(cases)


def required_images(
    cases: Sequence[Tuple[AcceptanceCase, str]]
) -> Tuple[Tuple[str, str], ...]:
    selected = [("base", ROCM_BASE_IMAGE)]
    for case, reason in cases:
        if case.application is None or reason:
            continue
        pair = (case.application, APPLICATIONS[case.application].image)
        if pair not in selected:
            selected.append(pair)
    return tuple(selected)


def required_bundles(
    catalog: Catalog, cases: Sequence[Tuple[AcceptanceCase, str]]
) -> Tuple[Bundle, ...]:
    bundles = []
    for case, reason in cases:
        if not case.bundle or reason:
            continue
        bundle = catalog.bundle(case.bundle)
        if bundle not in bundles:
            bundles.append(bundle)
    return tuple(bundles)


def probe_hardware(image: str, render_node: str) -> Mapping[str, str]:
    podman.require_rootless()
    output = podman.capture_stdout(
        gpu_diagnostic_command(image, (render_node,)),
        "GPU acceptance probe failed",
    )
    fields = parse_gpu_diagnostic_output(output)
    architecture = fields["Architecture"]
    profile = ARCHITECTURE_PROFILES.get(architecture)
    if profile is None:
        raise LauncherError(
            "GPU acceptance does not support architecture {!r}".format(
                architecture
            )
        )
    result = dict(fields)
    result["Profile"] = profile
    return result


def _bundle_identity(catalog: Catalog, bundle: Bundle) -> Mapping[str, object]:
    value: Dict[str, object] = {
        "identifier": bundle.identifier,
        "artifacts": [
            {
                "identifier": artifact.identifier,
                "sha256": artifact.sha256,
                "size": artifact.size,
            }
            for artifact in catalog.bundle_artifacts(bundle)
        ],
    }
    if bundle.identifier in catalog.benchmarks:
        benchmark = catalog.benchmark(bundle.identifier)
        value["benchmark"] = {
            "sha256": benchmark.sha256,
            "renderer": benchmark.renderer,
            "rendered_sha256": benchmark.rendered_sha256,
        }
    return value


def acceptance_definition(
    catalog: Catalog,
    cases: Sequence[Tuple[AcceptanceCase, str]],
    *,
    profile: str,
    architecture: str,
    render_node: str,
    image_ids: Mapping[str, str],
    source_identity: str,
    memory_policy: str,
    kernel_policy: str,
) -> Mapping[str, object]:
    bundles = required_bundles(catalog, cases)
    definition = {
        "policy_version": POLICY_VERSION,
        "source_identity": source_identity,
        "profile": profile,
        "architecture": architecture,
        "render_node": render_node,
        "kernel_policy": kernel_policy,
        "images": dict(sorted(image_ids.items())),
        "bundles": [
            _bundle_identity(catalog, bundle) for bundle in bundles
        ],
        "cases": [
            {
                "identifier": case.identifier,
                "description": case.description,
                "application": case.application,
                "bundle": case.bundle,
                "not_applicable": reason,
                "visual": case.visual,
                "review_criteria": list(case.review_criteria),
            }
            for case, reason in cases
        ],
    }
    if any(
        case.application == "comfyui"
        for case, _ in cases
    ):
        definition["memory_policy"] = memory_policy
    return definition


def acceptance_fingerprint(definition: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(definition)).hexdigest()


def default_result_path(data_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = "{}-{}.json".format(stamp, uuid.uuid4().hex[:8])
    return StorageLayout(data_dir).acceptance_results / name


def create_result(
    definition: Mapping[str, object],
    cases: Sequence[Tuple[AcceptanceCase, str]],
    *,
    hardware: Mapping[str, str],
) -> MutableMapping[str, object]:
    suite_id = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ-"
    ) + uuid.uuid4().hex[:8]
    entries = []
    for case, reason in cases:
        entries.append(
            {
                "identifier": case.identifier,
                "description": case.description,
                "application": case.application,
                "bundle": case.bundle or None,
                "visual": case.visual,
                "review_criteria": list(case.review_criteria),
                "status": "n/p" if reason else "pending",
                "reason": reason or None,
                "attempts": 0,
                "artifacts": [],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "status": "running",
        "started_at": _timestamp(),
        "finished_at": None,
        "fingerprint": acceptance_fingerprint(definition),
        "definition": dict(definition),
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
        },
        "hardware": dict(hardware),
        "cases": entries,
    }


def load_result(
    path: Path, expected_fingerprint: Optional[str] = None
) -> MutableMapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(
            "cannot read acceptance result {}: {}".format(path, error)
        )
    if not isinstance(value, dict):
        raise LauncherError("acceptance result must contain a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise LauncherError("acceptance result has an unsupported schema")
    _validate_result_structure(value)
    if expected_fingerprint is not None:
        validate_result_fingerprint(value, expected_fingerprint)
    return value


def _validate_result_structure(value: Mapping[str, object]) -> None:
    suite_id = value.get("suite_id")
    if not isinstance(suite_id, str) or SUITE_ID_PATTERN.fullmatch(
        suite_id
    ) is None:
        raise LauncherError("acceptance result has an invalid suite ID")
    fingerprint = value.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
    ):
        raise LauncherError("acceptance result has an invalid fingerprint")
    if not isinstance(value.get("started_at"), str):
        raise LauncherError("acceptance result has an invalid start time")
    finished_at = value.get("finished_at")
    if finished_at is not None and not isinstance(finished_at, str):
        raise LauncherError("acceptance result has an invalid finish time")
    if not isinstance(value.get("host"), dict) or not isinstance(
        value.get("hardware"), dict
    ):
        raise LauncherError("acceptance result has invalid host metadata")
    definition = value.get("definition")
    if not isinstance(definition, dict):
        raise LauncherError("acceptance result has no definition")
    if definition.get("policy_version") != POLICY_VERSION:
        raise LauncherError("acceptance result has an unsupported policy")
    for field in (
        "source_identity",
        "profile",
        "architecture",
        "render_node",
        "kernel_policy",
    ):
        if not isinstance(definition.get(field), str):
            raise LauncherError(
                "acceptance result definition has an invalid {}".format(field)
            )
    if not isinstance(definition.get("images"), dict) or not isinstance(
        definition.get("bundles"), list
    ):
        raise LauncherError(
            "acceptance result definition has invalid prerequisites"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not cases:
        raise LauncherError("acceptance result has no cases")
    defined_cases = definition.get("cases")
    if not isinstance(defined_cases, list) or not defined_cases:
        raise LauncherError("acceptance result definition has no cases")
    if len(cases) != len(defined_cases):
        raise LauncherError(
            "acceptance result cases do not match its definition"
        )
    identifiers = []
    for entry, defined in zip(cases, defined_cases):
        if (
            not isinstance(entry, dict)
            or not isinstance(defined, dict)
            or not isinstance(entry.get("identifier"), str)
            or entry.get("status") not in CASE_STATUSES
        ):
            raise LauncherError("acceptance result contains an invalid case")
        identifiers.append(entry["identifier"])
        identifier = defined.get("identifier")
        description = defined.get("description")
        application = defined.get("application")
        bundle = defined.get("bundle")
        not_applicable = defined.get("not_applicable")
        visual = defined.get("visual")
        criteria = defined.get("review_criteria")
        if (
            not isinstance(identifier, str)
            or not isinstance(description, str)
            or (
                application is not None
                and not isinstance(application, str)
            )
            or not isinstance(bundle, str)
            or not isinstance(not_applicable, str)
            or not isinstance(visual, bool)
            or not isinstance(criteria, list)
            or any(not isinstance(item, str) for item in criteria)
        ):
            raise LauncherError(
                "acceptance result definition contains an invalid case"
            )
        expected_bundle = bundle or None
        if (
            entry.get("identifier") != identifier
            or entry.get("description") != description
            or entry.get("application") != application
            or entry.get("bundle") != expected_bundle
            or entry.get("visual") != visual
            or entry.get("review_criteria") != criteria
        ):
            raise LauncherError(
                "acceptance result case metadata does not match its definition"
            )
        attempts = entry.get("attempts")
        artifacts = entry.get("artifacts")
        reason = entry.get("reason")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 0
            or not isinstance(artifacts, list)
            or any(not isinstance(item, str) for item in artifacts)
            or (reason is not None and not isinstance(reason, str))
        ):
            raise LauncherError(
                "acceptance result contains invalid case state"
            )
        if not_applicable:
            if entry.get("status") != "n/p" or reason != not_applicable:
                raise LauncherError(
                    "acceptance not-applicable case state is invalid"
                )
        elif entry.get("status") == "n/p":
            raise LauncherError(
                "acceptance applicable case cannot be not-applicable"
            )
    if len(identifiers) != len(set(identifiers)):
        raise LauncherError("acceptance result contains duplicate cases")
    status = value.get("status")
    expected_status = _result_status(value)
    if status not in RESULT_STATUSES or status != expected_status:
        raise LauncherError(
            "acceptance result status does not match its cases"
        )


def validate_result_fingerprint(
    value: MutableMapping[str, object], expected_fingerprint: str
) -> None:
    if value.get("fingerprint") != expected_fingerprint:
        raise LauncherError(
            "acceptance source, images, content, hardware, or selection "
            "changed; start a new run instead of resuming"
        )
    definition = value["definition"]
    if (
        not isinstance(definition, dict)
        or acceptance_fingerprint(definition) != expected_fingerprint
    ):
        raise LauncherError(
            "acceptance result definition does not match its fingerprint"
        )


def _result_status(result: Mapping[str, object]) -> str:
    entries = result.get("cases", [])
    statuses = {
        entry.get("status")
        for entry in entries
        if isinstance(entry, dict)
    }
    if "fail" in statuses:
        return "fail"
    if statuses - {"pass", "n/p"}:
        return "blocked"
    return "pass"


def _write_atomic(path: Path, contents: bytes, *, create: bool) -> None:
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        if create:
            try:
                os.link(str(temporary), str(path))
            except FileExistsError:
                raise LauncherError(
                    "refusing to replace acceptance output: {}".format(path)
                )
        else:
            os.replace(str(temporary), str(path))
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except LauncherError:
        raise
    except OSError as error:
        raise LauncherError(
            "cannot write acceptance output {}: {}".format(path, error)
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def checkpoint(
    path: Path,
    result: MutableMapping[str, object],
    *,
    create: bool = False,
) -> None:
    result["status"] = _result_status(result)
    _write_atomic(path, _canonical_json(result), create=create)


def finish(
    path: Path,
    result: MutableMapping[str, object],
    *,
    create_report: bool = False,
) -> Path:
    result["finished_at"] = _timestamp()
    checkpoint(path, result)
    report = path.with_suffix(".md")
    _write_atomic(
        report,
        render_markdown(result).encode("utf-8"),
        create=create_report,
    )
    return report


def render_markdown(result: Mapping[str, object]) -> str:
    hardware = result.get("hardware", {})
    if not isinstance(hardware, dict):
        hardware = {}
    lines = [
        "# ROCmplete smoke acceptance",
        "",
        "- Suite: `{}`".format(result.get("suite_id", "unknown")),
        "- Status: `{}`".format(
            str(result.get("status", "unknown")).upper()
        ),
        "- Host: `{}`".format(
            (result.get("host") or {}).get("hostname", "unknown")
            if isinstance(result.get("host"), dict)
            else "unknown"
        ),
        "- Device: `{}`".format(hardware.get("Device", "unknown")),
        "- Architecture: `{}`".format(
            hardware.get("Architecture", "unknown")
        ),
        "- Profile: `{}`".format(hardware.get("Profile", "unknown")),
        "- Render node: `{}`".format(
            (result.get("definition") or {}).get("render_node", "unknown")
            if isinstance(result.get("definition"), dict)
            else "unknown"
        ),
        "",
        "| Case | Application | Status | Duration | Artifacts / reason |",
        "| --- | --- | --- | ---: | --- |",
    ]
    entries = result.get("cases", [])
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            artifacts = entry.get("artifacts")
            detail = entry.get("reason") or ""
            if isinstance(artifacts, list) and artifacts:
                detail = "<br>".join("`{}`".format(item) for item in artifacts)
            duration = entry.get("wall_seconds")
            rendered_duration = (
                "{:.2f}s".format(duration)
                if isinstance(duration, (int, float))
                else "—"
            )
            lines.append(
                "| `{}` | {} | **{}** | {} | {} |".format(
                    entry.get("identifier", "unknown"),
                    entry.get("application") or "host",
                    str(entry.get("status", "unknown")).upper(),
                    rendered_duration,
                    str(detail).replace("|", "\\|"),
                )
            )
    lines.extend(
        [
            "",
            "A visual case is `PASS` only after its generated artifact was "
            "reviewed.",
            "",
        ]
    )
    visual_entries = (
        [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("visual") is True
            and isinstance(entry.get("review_criteria"), list)
            and entry.get("review_criteria")
        ]
        if isinstance(entries, list)
        else []
    )
    if visual_entries:
        lines.extend(["## Visual review criteria", ""])
        for entry in visual_entries:
            lines.extend(
                [
                    "### `{}`".format(
                        entry.get("identifier", "unknown")
                    ),
                    "",
                ]
            )
            for criterion in entry["review_criteria"]:
                lines.append("- {}".format(criterion))
            lines.append("")
    return "\n".join(lines)


def smoke_comfy_prompt(
    source: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a small fail-closed four-step graph from the pinned benchmark."""
    prompt = copy.deepcopy(source)
    switches = []
    latents = []
    positive = []
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        metadata = node.get("_meta")
        if not isinstance(inputs, dict):
            continue
        title = metadata.get("title") if isinstance(metadata, dict) else ""
        if class_type == "PrimitiveBoolean" and title == "Enable 4 Steps LoRA?":
            switches.append(inputs)
        elif class_type == "EmptySD3LatentImage":
            latents.append(inputs)
        elif class_type == "CLIPTextEncode" and "Positive" in str(title):
            positive.append(inputs)
    if len(switches) != 1 or len(latents) != 1 or len(positive) != 1:
        raise LauncherError(
            "Qwen Image smoke benchmark no longer has the expected switch, "
            "latent, and positive prompt nodes"
        )
    switches[0]["value"] = True
    latents[0]["width"] = 768
    latents[0]["height"] = 768
    positive[0]["text"] = (
        "A red cube on a blue table, studio lighting, sharp focus"
    )
    return prompt


def smoke_comfy_video_prompt(
    source: Mapping[str, object]
) -> Mapping[str, object]:
    """Create a five-frame fail-closed Wan graph from the pinned benchmark."""
    prompt = copy.deepcopy(source)
    switches = []
    latents = []
    durations = []
    positive = []
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        metadata = node.get("_meta")
        if not isinstance(inputs, dict):
            continue
        title = metadata.get("title") if isinstance(metadata, dict) else ""
        if class_type == "PrimitiveBoolean" and title == "Enable Lightning LoRA":
            switches.append(inputs)
        elif class_type == "EmptyHunyuanLatentVideo":
            latents.append(inputs)
        elif class_type == "PrimitiveFloat" and title == "Float (Duration)":
            durations.append(inputs)
        elif class_type == "CLIPTextEncode" and "Positive" in str(title):
            positive.append(inputs)
    if not all(
        len(items) == 1
        for items in (switches, latents, durations, positive)
    ):
        raise LauncherError(
            "Wan video smoke benchmark no longer has the expected switch, "
            "latent, duration, and positive prompt nodes"
        )
    switches[0]["value"] = True
    latents[0]["width"] = 832
    latents[0]["height"] = 480
    durations[0]["value"] = 0.25
    positive[0]["text"] = (
        "A red cube slowly rotating on a blue table, studio lighting"
    )
    return prompt


def _validate_png(path: Path) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            header = stream.read(32)
        size = path.stat().st_size
    except OSError as error:
        raise LauncherError(
            "cannot inspect generated PNG {}: {}".format(path, error)
        )
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise LauncherError("generated output is not a PNG: {}".format(path))
    width, height = struct.unpack(">II", header[16:24])
    if width < 64 or height < 64 or size < 1024:
        raise LauncherError(
            "generated PNG is unexpectedly small: {}".format(path)
        )
    return {
        "path": str(path),
        "size": size,
        "width": width,
        "height": height,
    }


def _validate_mp4(path: Path) -> Mapping[str, object]:
    try:
        with path.open("rb") as stream:
            header = stream.read(64)
        size = path.stat().st_size
    except OSError as error:
        raise LauncherError(
            "cannot inspect generated MP4 {}: {}".format(path, error)
        )
    if b"ftyp" not in header or size < 1024:
        raise LauncherError(
            "generated output is not a valid-sized MP4: {}".format(path)
        )
    return {"path": str(path), "size": size}


def _acceptance_output_path(
    root: Path,
    suite_id: str,
    attempt: int,
    leaf: str,
) -> Path:
    if SUITE_ID_PATTERN.fullmatch(suite_id) is None:
        raise LauncherError("acceptance suite ID is unsafe for output paths")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise LauncherError("acceptance attempt is invalid")
    try:
        root.mkdir(parents=True, exist_ok=True)
        root_status = root.lstat()
    except OSError as error:
        raise LauncherError(
            "cannot prepare acceptance output root {}: {}".format(root, error)
        )
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(
        root_status.st_mode
    ):
        raise LauncherError(
            "acceptance output root is not a directory: {}".format(root)
        )
    relative = Path(leaf)
    if relative.is_absolute() or ".." in relative.parts:
        raise LauncherError("acceptance output path is unsafe")
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (root / relative).resolve(strict=False)
        candidate.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise LauncherError(
            "acceptance output escapes its managed root: {} ({})".format(
                root / relative, error
            )
        )
    return candidate


def run_host_case(image: str, render_node: str) -> Mapping[str, object]:
    hardware = probe_hardware(image, render_node)
    cpu_output = podman.capture_stdout(
        cpu_isolation_diagnostic_command(image),
        "CPU device-isolation probe failed",
    )
    if "CPU device isolation: passed" not in cpu_output:
        raise LauncherError(
            "CPU device-isolation probe returned incomplete output"
        )
    return {"hardware": dict(hardware)}


def run_comfyui_case(
    catalog: Catalog,
    *,
    identifier: str,
    data_dir: Path,
    profile: str,
    render_node: str,
    port: int,
    suite_id: str,
    attempt: int,
    memory_policy: str,
    kernel_policy: str,
) -> Mapping[str, object]:
    if SUITE_ID_PATTERN.fullmatch(suite_id) is None:
        raise LauncherError("acceptance suite ID is unsafe for output paths")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise LauncherError("acceptance attempt is invalid")
    layout = StorageLayout(data_dir)
    layout.prepare_runtime("comfyui")
    cases = {
        "comfyui-image": (
            "qwen-image-2512-fp8-lightning",
            ".png",
            smoke_comfy_prompt,
            _validate_png,
        ),
        "comfyui-video": (
            "wan-2.2-t2v-14b-fp8-lightning",
            ".mp4",
            smoke_comfy_video_prompt,
            _validate_mp4,
        ),
    }
    try:
        bundle_identifier, suffix, transform, validator = cases[identifier]
    except KeyError:
        raise LauncherError(
            "unknown ComfyUI acceptance case {!r}".format(identifier)
        )
    run_id = "acceptance-{}-{}-{}".format(suite_id, identifier, attempt)
    result_path = run_benchmark(
        catalog,
        catalog.bundle(bundle_identifier),
        BenchmarkOptions(
            image=APPLICATIONS["comfyui"].image,
            profile=profile,
            port=port,
            data_dir=data_dir,
            render_node=render_node,
            runs=1,
            seed=10,
            memory_policy=memory_policy,
            kernel_policy=kernel_policy,
            cache_mode="persistent",
        ),
        podman.selinux_volume_suffix(),
        run_id=run_id,
        prompt_transform=transform,
    )
    output_dir = (
        layout.comfyui
        / "output"
        / "rocmplete-benchmarks"
        / run_id
    )
    outputs = sorted(output_dir.rglob("*{}".format(suffix)))
    if len(outputs) != 1:
        raise LauncherError(
            "{} smoke expected one {} file, found {} below {}".format(
                identifier, suffix, len(outputs), output_dir
            )
        )
    return {
        "artifacts": [str(outputs[0]), str(result_path)],
        "media": validator(outputs[0]),
    }


def run_llama_case(
    catalog: Catalog,
    *,
    data_dir: Path,
    profile: str,
    render_node: str,
    suite_id: str,
    attempt: int,
) -> Mapping[str, object]:
    preset = catalog.llama_preset("qwen3-0.6b-q8-0")
    artifact = catalog.artifact(preset.artifact)
    installed = artifact_path(data_dir, artifact)
    if not installed.is_file():
        raise LauncherError(
            "llama.cpp acceptance model is not installed: {}".format(installed)
        )
    options = LlamaBenchmarkOptions(
        image=APPLICATIONS["llama-cpp"].image,
        profile=profile,
        data_dir=data_dir,
        managed_model=artifact.destination,
        render_nodes=(render_node,),
        repetitions=1,
        prompt_tokens=32,
        generation_tokens=16,
    )
    command = llama_benchmark_command(
        options, podman.selinux_volume_suffix()
    )
    output = _acceptance_output_path(
        StorageLayout(data_dir).acceptance_results / "cases",
        suite_id,
        attempt,
        "{}-llama-{}.json".format(suite_id, attempt),
    )
    result = run_llama_benchmark(
        command,
        data_dir=data_dir,
        image=options.image,
        profile=profile,
        backend=options.backend,
        render_nodes=(render_node,),
        model={
            "kind": "catalog",
            "preset": preset.identifier,
            "path": str(installed),
            "repository": artifact.source.repository,
            "revision": artifact.source.revision,
            "source_path": artifact.source.path,
            "size": artifact.size,
            "sha256": artifact.sha256,
        },
        parameters={
            "repetitions": 1,
            "prompt_tokens": 32,
            "generation_tokens": 16,
        },
        output=output,
    )
    return {"artifacts": [str(result)]}


def run_dwarfstar_case(
    catalog: Catalog,
    *,
    data_dir: Path,
    profile: str,
    render_node: str,
) -> Mapping[str, object]:
    bundle = catalog.bundle("dwarfstar-deepseek-v4-flash-0731-iq2xxs")
    artifact = catalog.artifact(bundle.artifacts[0])
    installed = artifact_path(data_dir, artifact)
    if not installed.is_file():
        raise LauncherError(
            "DwarfStar acceptance model is not installed: {}".format(
                installed
            )
        )
    StorageLayout(data_dir).prepare_runtime("dwarfstar")
    options = DwarfStarOptions(
        image=APPLICATIONS["dwarfstar"].image,
        mode="cli",
        data_dir=data_dir,
        model=installed,
        render_nodes=(render_node,),
        profile=profile,
        context=4096,
        output_tokens=64,
        prompt="Reply with exactly: DwarfStar acceptance passed",
        no_thinking=True,
    )
    command = dwarfstar_command(
        options, podman.selinux_volume_suffix()
    )
    podman.run_managed_foreground(
        command,
        APPLICATIONS["dwarfstar"].container_name,
        "DwarfStar acceptance generation failed",
    )
    return {
        "model": str(installed),
        "context": options.context,
        "output_tokens": options.output_tokens,
    }


def run_application_case(
    identifier: str,
    catalog: Catalog,
    *,
    data_dir: Path,
    profile: str,
    render_node: str,
    port: int,
    suite_id: str,
    attempt: int,
    memory_policy: str,
    kernel_policy: str,
) -> Mapping[str, object]:
    if identifier in ("comfyui-image", "comfyui-video"):
        return run_comfyui_case(
            catalog,
            identifier=identifier,
            data_dir=data_dir,
            profile=profile,
            render_node=render_node,
            port=port,
            suite_id=suite_id,
            attempt=attempt,
            memory_policy=memory_policy,
            kernel_policy=kernel_policy,
        )
    if identifier == "llama-cpp":
        return run_llama_case(
            catalog,
            data_dir=data_dir,
            profile=profile,
            render_node=render_node,
            suite_id=suite_id,
            attempt=attempt,
        )
    if identifier == "dwarfstar":
        return run_dwarfstar_case(
            catalog,
            data_dir=data_dir,
            profile=profile,
            render_node=render_node,
        )
    raise LauncherError("unknown acceptance case {!r}".format(identifier))


def case_entry(
    result: MutableMapping[str, object], identifier: str
) -> MutableMapping[str, object]:
    entries = result.get("cases")
    if not isinstance(entries, list):
        raise LauncherError("acceptance result cases are invalid")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("identifier") == identifier
    ]
    if len(matches) != 1:
        raise LauncherError(
            "acceptance result does not contain exactly one {!r} case".format(
                identifier
            )
        )
    return matches[0]


def pending_case_identifiers(
    result: Mapping[str, object]
) -> Tuple[str, ...]:
    entries = result.get("cases")
    if not isinstance(entries, list):
        raise LauncherError("acceptance result cases are invalid")
    return tuple(
        str(entry["identifier"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("status")
        in ("pending", "running", "fail", "interrupted")
    )


def blocked_visual_identifiers(
    result: Mapping[str, object]
) -> Tuple[str, ...]:
    entries = result.get("cases")
    if not isinstance(entries, list):
        raise LauncherError("acceptance result cases are invalid")
    return tuple(
        str(entry["identifier"])
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("visual") is True
        and entry.get("status") == "blocked"
        and isinstance(entry.get("artifacts"), list)
        and entry.get("artifacts")
    )


def begin_case(entry: MutableMapping[str, object]) -> int:
    attempts = entry.get("attempts", 0)
    if not isinstance(attempts, int) or attempts < 0:
        raise LauncherError("acceptance case attempt count is invalid")
    attempts += 1
    entry.update(
        {
            "attempts": attempts,
            "status": "running",
            "started_at": _timestamp(),
            "finished_at": None,
            "reason": None,
            "wall_seconds": None,
            "artifacts": [],
        }
    )
    return attempts


def complete_case(
    entry: MutableMapping[str, object],
    outcome: Mapping[str, object],
    *,
    started: float,
) -> None:
    artifacts = outcome.get("artifacts", [])
    entry.update(
        {
            "status": "blocked" if entry.get("visual") else "pass",
            "finished_at": _timestamp(),
            "wall_seconds": time.monotonic() - started,
            "artifacts": list(artifacts) if isinstance(artifacts, list) else [],
            "details": dict(outcome),
            "reason": (
                "generated output requires human sanity review"
                if entry.get("visual")
                else None
            ),
        }
    )


def fail_case(
    entry: MutableMapping[str, object],
    error: BaseException,
    *,
    started: float,
    interrupted: bool = False,
) -> None:
    entry.update(
        {
            "status": "interrupted" if interrupted else "fail",
            "finished_at": _timestamp(),
            "wall_seconds": time.monotonic() - started,
            "reason": str(error),
        }
    )
