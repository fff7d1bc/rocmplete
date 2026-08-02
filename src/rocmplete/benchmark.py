"""Managed, reproducible ComfyUI bundle benchmarks."""

from __future__ import annotations

import copy
import html
import hashlib
import json
import os
import platform
import shutil
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

from . import podman
from .bundles import (
    content_status_ready,
    content_status_state,
    inspect_bundle,
)
from .catalog import BenchmarkWorkflow, Bundle, Catalog
from .config import APPLICATIONS, COMFY_BENCHMARK_CONTAINER_NAME
from .content_verification import VerificationStore
from .errors import LauncherError
from .layout import StorageLayout
from .project import PROJECT_ROOT
from .runtime.web import WebOptions, web_command

BENCHMARK_CONTAINER_NAME = COMFY_BENCHMARK_CONTAINER_NAME
BENCHMARK_SCHEMA_VERSION = 4
BENCHMARK_SUITE_SCHEMA_VERSION = 3
DEFAULT_BENCHMARK_PORT = 8190
SYNTHETIC_INPUT_NAME = "rocmplete/benchmark-input-768.png"


@dataclass(frozen=True)
class BenchmarkOptions:
    image: str
    profile: str
    port: int
    data_dir: Path
    render_node: str
    runs: int = 2
    seed: int = 10
    unconfined: bool = False
    dry_run: bool = False
    memory_policy: str = "balanced"
    kernel_policy: str = "default"
    cache_mode: str = "persistent"


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def load_benchmark_prompt(spec: BenchmarkWorkflow) -> Mapping[str, object]:
    path = PROJECT_ROOT / "catalog" / spec.resource
    try:
        contents = path.read_bytes()
    except OSError as error:
        raise LauncherError(
            "cannot read benchmark workflow {}: {}".format(path, error)
        )
    digest = hashlib.sha256(contents).hexdigest()
    if digest != spec.sha256:
        raise LauncherError(
            "benchmark workflow does not match catalog (expected {}, got {})".format(
                spec.sha256, digest
            )
        )
    try:
        prompt = json.loads(contents)
    except json.JSONDecodeError as error:
        raise LauncherError(
            "benchmark workflow contains invalid JSON: {}".format(error)
        )
    if not isinstance(prompt, dict) or not prompt:
        raise LauncherError("benchmark workflow root must be a non-empty object")
    if spec.renderer == "identity":
        return prompt
    if spec.renderer == "hunyuan-i2v-480p-step-distilled":
        prompt = _render_hunyuan_480p_prompt(prompt, mode="i2v")
    elif spec.renderer == "hunyuan-t2v-480p-cfg-distilled":
        prompt = _render_hunyuan_480p_prompt(prompt, mode="t2v")
    else:
        raise LauncherError(
            "unknown benchmark renderer: {}".format(spec.renderer)
        )
    rendered_digest = hashlib.sha256(_canonical_json(prompt)).hexdigest()
    if rendered_digest != spec.rendered_sha256:
        raise LauncherError(
            "rendered benchmark workflow does not match catalog "
            "(expected {}, got {})".format(
                spec.rendered_sha256, rendered_digest
            )
        )
    return prompt


def _render_hunyuan_480p_prompt(
    source: Mapping[str, object], mode: str
) -> Mapping[str, object]:
    prompt = copy.deepcopy(source)
    if mode == "i2v":
        model = "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors"
        dimension_type = "HunyuanVideo15ImageToVideo"
        steps, shift = 12, 7
    elif mode == "t2v":
        model = "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors"
        dimension_type = "EmptyHunyuanVideo15Latent"
        steps, shift = 50, 5
    else:
        raise LauncherError("unknown Hunyuan benchmark mode: {}".format(mode))

    save_nodes = [
        identifier
        for identifier, node in prompt.items()
        if isinstance(node, dict)
        and node.get("class_type") == "SaveVideo"
        and isinstance(node.get("inputs"), dict)
        and node["inputs"].get("filename_prefix") == "video/hunyuan_video_1.5"
    ]
    if len(save_nodes) != 1:
        raise LauncherError(
            "official Hunyuan benchmark must contain one base SaveVideo"
        )
    reachable = set()
    pending = list(save_nodes)
    while pending:
        identifier = pending.pop()
        if identifier in reachable:
            continue
        node = prompt.get(identifier)
        if not isinstance(node, dict):
            raise LauncherError("Hunyuan benchmark has a broken node link")
        reachable.add(identifier)
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            raise LauncherError("Hunyuan benchmark node inputs are invalid")
        for value in inputs.values():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
            ):
                pending.append(value[0])
    prompt = {
        identifier: node
        for identifier, node in prompt.items()
        if identifier in reachable
    }

    counts = {
        "UNETLoader": 0,
        dimension_type: 0,
        "BasicScheduler": 0,
        "CFGGuider": 0,
        "ModelSamplingSD3": 0,
    }
    for node in prompt.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_type == "UNETLoader":
            inputs["unet_name"] = model
            counts[class_type] += 1
        elif class_type == dimension_type:
            inputs["width"], inputs["height"] = 832, 480
            counts[class_type] += 1
        elif class_type == "BasicScheduler":
            inputs["steps"] = steps
            counts[class_type] += 1
        elif class_type == "CFGGuider":
            inputs["cfg"] = 1
            counts[class_type] += 1
        elif class_type == "ModelSamplingSD3":
            inputs["shift"] = shift
            counts[class_type] += 1
    if any(count != 1 for count in counts.values()):
        raise LauncherError(
            "official Hunyuan base graph changed unexpectedly: {}".format(counts)
        )
    return prompt


def prepare_prompt(
    source: Mapping[str, object],
    seed: int,
    output_prefix: str,
) -> Tuple[Mapping[str, object], bool]:
    prompt = copy.deepcopy(source)
    has_input = False
    for raw_node in prompt.values():
        if not isinstance(raw_node, dict):
            raise LauncherError("benchmark workflow contains an invalid node")
        inputs = raw_node.get("inputs")
        if not isinstance(inputs, dict):
            raise LauncherError(
                "benchmark workflow node has invalid inputs"
            )
        for field in ("seed", "noise_seed"):
            if field in inputs and not isinstance(inputs[field], list):
                inputs[field] = seed
        class_type = raw_node.get("class_type")
        if class_type == "LoadImage":
            inputs["image"] = SYNTHETIC_INPUT_NAME
            has_input = True
        elif class_type == "LoadVideo":
            raise LauncherError(
                "managed benchmark does not support video-conditioned input"
            )
        if class_type in ("SaveImage", "SaveVideo", "VHS_VideoCombine"):
            inputs["filename_prefix"] = output_prefix
    return prompt, has_input


def _png_chunk(kind: bytes, contents: bytes) -> bytes:
    body = kind + contents
    return (
        struct.pack(">I", len(contents))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def synthetic_png() -> bytes:
    width = 768
    height = 768
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            rows.extend(
                (
                    (x * 255) // (width - 1),
                    (y * 255) // (height - 1),
                    ((x ^ y) * 255) // 1023,
                )
            )
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + _png_chunk(b"IEND", b"")
    )


def ensure_synthetic_input(data_dir: Path) -> Tuple[Path, str]:
    contents = synthetic_png()
    digest = hashlib.sha256(contents).hexdigest()
    destination = (
        StorageLayout(data_dir).comfyui / "input" / SYNTHETIC_INPUT_NAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            current = destination.read_bytes()
        except OSError as error:
            raise LauncherError(
                "cannot read synthetic benchmark input: {}".format(error)
            )
        if current != contents:
            raise LauncherError(
                "synthetic input path contains unexpected content: {}".format(
                    destination
                )
            )
    else:
        try:
            destination.write_bytes(contents)
        except OSError as error:
            raise LauncherError(
                "cannot write synthetic benchmark input: {}".format(error)
            )
    return destination, digest


def _request_json(
    url: str, body: Optional[Mapping[str, object]] = None
) -> Mapping[str, object]:
    data = _canonical_json(body) if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            value = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise LauncherError("ComfyUI request failed: {}".format(error))
    if not isinstance(value, dict):
        raise LauncherError("ComfyUI returned an unexpected response")
    return value


def wait_for_server(base_url: str, timeout: float = 120.0) -> Mapping[str, object]:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            return _request_json(base_url + "/system_stats")
        except LauncherError as error:
            last_error = str(error)
            time.sleep(1)
    raise LauncherError(
        "ComfyUI did not become ready within {:.0f}s: {}".format(
            timeout, last_error
        )
    )


def queue_prompt(
    base_url: str, prompt: Mapping[str, object]
) -> str:
    response = _request_json(
        base_url + "/prompt",
        {"prompt": prompt, "client_id": str(uuid.uuid4())},
    )
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        errors = response.get("node_errors")
        raise LauncherError(
            "ComfyUI rejected benchmark prompt: {}".format(errors or response)
        )
    return prompt_id


def wait_for_prompt(
    base_url: str, prompt_id: str
) -> Mapping[str, object]:
    while True:
        history = _request_json(base_url + "/history/" + prompt_id)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            status = entry.get("status")
            if isinstance(status, dict) and status.get("completed") is True:
                if status.get("status_str") == "error":
                    raise LauncherError(
                        "ComfyUI reported benchmark execution failure"
                    )
                return entry
        time.sleep(1)


def _assert_port_available(port: int) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        raise LauncherError(
            "benchmark port {} is already in use".format(port)
        )
    finally:
        probe.close()


def _write_result(path: Path, result: Mapping[str, object]) -> None:
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
    )
    created = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as output:
            created = True
            output.write(_canonical_json(result))
        os.replace(str(temporary), str(path))
    except OSError as error:
        raise LauncherError(
            "cannot write benchmark result {}: {}".format(path, error)
        )
    finally:
        if created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _content_destination(status: object) -> str:
    artifact = getattr(status, "artifact", None)
    if artifact is not None:
        return str(artifact.destination)
    tree = getattr(status, "tree", None)
    file = getattr(status, "file", None)
    if tree is not None and file is not None:
        return "{}/{}".format(tree.destination, file.path)
    return "unknown content"


def _read_json_object(
    path: Path, description: str
) -> MutableMapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(
            "cannot read {} {}: {}".format(description, path, error)
        )
    if not isinstance(value, dict):
        raise LauncherError(
            "{} {} must contain a JSON object".format(description, path)
        )
    return value


def _image_metadata(image: str) -> Mapping[str, object]:
    try:
        raw = podman.capture(
            ["podman", "image", "inspect", image, "--format", "json"],
            "cannot inspect benchmark image",
        )
        value = json.loads(raw)
    except (LauncherError, json.JSONDecodeError):
        return {"reference": image}
    if isinstance(value, list) and value and isinstance(value[0], dict):
        inspected = value[0]
    elif isinstance(value, dict):
        inspected = value
    else:
        return {"reference": image}
    config = inspected.get("Config")
    labels = config.get("Labels", {}) if isinstance(config, dict) else {}
    if not isinstance(labels, dict):
        labels = {}
    return {
        "reference": image,
        "id": inspected.get("Id", ""),
        "digest": inspected.get("Digest", ""),
        "created": inspected.get("Created", ""),
        "labels": {
            key: value
            for key, value in labels.items()
            if isinstance(key, str)
            and (
                key.startswith("io.github.fff7d1bc.rocmplete.")
                or key.startswith("org.opencontainers.image.")
            )
        },
    }


def _container_logs() -> str:
    if not podman.container_exists(BENCHMARK_CONTAINER_NAME):
        return ""
    try:
        return podman.capture(
            ["podman", "logs", BENCHMARK_CONTAINER_NAME],
            "cannot read benchmark logs",
        )
    except LauncherError as error:
        return str(error)


def _remove_container() -> None:
    # ``podman rm --ignore`` is safe after normal --rm completion. Enter the
    # signal-masked removal path without a separate existence-check race.
    podman.remove_container(
        BENCHMARK_CONTAINER_NAME,
        stop_timeout=0,
    )


def _cleanup_benchmark_resources(
    isolated_cache_root: Optional[Path],
    primary_error: Optional[BaseException],
) -> None:
    cleanup_errors = []  # type: List[BaseException]
    try:
        _remove_container()
    except BaseException as error:
        cleanup_errors.append(error)
    if isolated_cache_root is not None:
        try:
            if isolated_cache_root.exists():
                shutil.rmtree(isolated_cache_root)
        except OSError as error:
            cleanup_errors.append(
                LauncherError(
                    "cannot remove isolated benchmark cache {}: {}".format(
                        isolated_cache_root, error
                    )
                )
            )
    if not cleanup_errors:
        return
    if primary_error is not None:
        for error in cleanup_errors:
            print(
                "WARNING: benchmark cleanup also failed after the primary "
                "error: {}".format(error),
                file=sys.stderr,
            )
        return
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    raise LauncherError(
        "benchmark cleanup failed: {}".format(
            "; ".join(str(error) for error in cleanup_errors)
        )
    )


def run_benchmark(
    catalog: Catalog,
    bundle: Bundle,
    options: BenchmarkOptions,
    volume_suffix: str,
    run_id: Optional[str] = None,
    prompt_transform: Optional[
        Callable[[Mapping[str, object]], Mapping[str, object]]
    ] = None,
) -> Path:
    statuses = inspect_bundle(catalog, bundle, options.data_dir)
    unavailable = [
        status
        for status in statuses
        if not content_status_ready(status)
    ]
    if unavailable:
        detail = ", ".join(
            "{} ({})".format(
                _content_destination(item), content_status_state(item)
            )
            for item in unavailable
        )
        if not options.dry_run:
            raise LauncherError(
                "bundle content is not verified and ready: {}; run "
                "'./rocmplete content install {}'".format(
                    detail, bundle.identifier
                )
            )
        print("WARNING: bundle is not ready: {}".format(detail))
    spec = catalog.benchmark(bundle.identifier)
    source = load_benchmark_prompt(spec)
    prompt_transform_sha256 = None
    if prompt_transform is not None:
        source = prompt_transform(source)
        prompt_transform_sha256 = hashlib.sha256(
            _canonical_json(source)
        ).hexdigest()
    workflow = catalog.workflow(bundle.workflow)
    if run_id is None:
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            + uuid.uuid4().hex[:8]
        )
    output_prefix = "rocmplete-benchmarks/{}/{}".format(
        run_id, bundle.identifier
    )
    preview, needs_input = prepare_prompt(source, options.seed, output_prefix)
    if options.cache_mode not in ("persistent", "isolated"):
        raise LauncherError(
            "unknown benchmark cache mode: {}".format(options.cache_mode)
        )
    safe_run_id_characters = frozenset(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789._-"
    )
    if (
        not run_id
        or run_id in (".", "..")
        or any(character not in safe_run_id_characters for character in run_id)
    ):
        raise LauncherError("benchmark run identifier is not path-safe")
    isolated_cache_root = None
    environment: Tuple[str, ...] = ()
    if options.cache_mode == "isolated":
        isolated_cache_root = (
            StorageLayout(options.data_dir).benchmarks / ".cache" / run_id
        )
        container_root = "/data/benchmarks/.cache/{}".format(run_id)
        environment = (
            "HOME={}/home".format(container_root),
            "XDG_CACHE_HOME={}/xdg".format(container_root),
            "HF_HOME={}/huggingface".format(container_root),
            "TORCH_HOME={}/torch".format(container_root),
            "TRITON_CACHE_DIR={}/triton".format(container_root),
        )
    command = web_command(
        WebOptions(
            image=options.image,
            profile=options.profile,
            listen="127.0.0.1",
            port=options.port,
            data_dir=options.data_dir,
            render_nodes=(options.render_node,),
            detach=True,
            unconfined=options.unconfined,
            disable_bundled_extensions=True,
            comfy_args=("--disable-all-custom-nodes",),
            container_name=BENCHMARK_CONTAINER_NAME,
            container_role="benchmark",
            memory_policy=options.memory_policy,
            kernel_policy=options.kernel_policy,
            environment=environment,
        ),
        volume_suffix,
    )
    if options.dry_run:
        print("Benchmark source SHA-256: {}".format(spec.sha256))
        print(
            "Benchmark rendered SHA-256: {}".format(
                spec.rendered_sha256
            )
        )
        print("Runs: {} (cold + {} warm)".format(options.runs, options.runs - 1))
        print("Cache mode: {}".format(options.cache_mode))
        print("Synthetic input: {}".format("yes" if needs_input else "not used"))
        print("Container command:")
        import shlex

        print("  {}".format(shlex.join(command)))
        return StorageLayout(options.data_dir).benchmarks / (run_id + ".json")

    podman.require_rootless()
    if not podman.image_exists(options.image):
        raise LauncherError(
            "image not found: {} (run './rocmplete build comfyui')".format(
                options.image
            )
        )
    if podman.container_exists(BENCHMARK_CONTAINER_NAME):
        raise LauncherError(
            "benchmark container already exists; remove it with "
            "'podman rm -f {}'".format(BENCHMARK_CONTAINER_NAME)
        )
    if podman.container_exists(APPLICATIONS["comfyui"].container_name):
        raise LauncherError(
            "the main ROCmplete container is running; stop it before "
            "benchmarking"
        )
    _assert_port_available(options.port)
    input_path = None
    input_digest = None
    if needs_input:
        input_path, input_digest = ensure_synthetic_input(options.data_dir)
    image_metadata = _image_metadata(options.image)
    if isolated_cache_root is not None:
        if isolated_cache_root.exists() or isolated_cache_root.is_symlink():
            raise LauncherError(
                "isolated benchmark cache already exists: {}".format(
                    isolated_cache_root
                )
            )
        isolated_cache_root.mkdir(parents=True)

    result_path = StorageLayout(options.data_dir).benchmarks / (
        run_id + ".json"
    )
    workflow_result: MutableMapping[str, object] = {
        "identifier": workflow.identifier,
        "rendered_sha256": workflow.rendered_sha256,
        "benchmark_source_sha256": spec.sha256,
        "benchmark_renderer": spec.renderer,
        "benchmark_rendered_sha256": spec.rendered_sha256,
    }
    if prompt_transform_sha256 is not None:
        workflow_result["effective_prompt_sha256"] = (
            prompt_transform_sha256
        )
    result: MutableMapping[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "bundle": bundle.identifier,
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "profile": options.profile,
        "render_node": options.render_node,
        "memory_policy": options.memory_policy,
        "kernel_policy": options.kernel_policy,
        "cache_mode": options.cache_mode,
        "unconfined": options.unconfined,
        "image": image_metadata,
        "workflow": workflow_result,
        "artifacts": [
            {
                "identifier": status.artifact.identifier,
                "sha256": status.artifact.sha256,
                "size": status.artifact.size,
            }
            for status in statuses
        ],
        "synthetic_input": (
            {
                "path": str(input_path),
                "sha256": input_digest,
                "width": 768,
                "height": 768,
            }
            if input_path
            else None
        ),
        "runs": [],
    }
    primary_error = None  # type: Optional[BaseException]
    try:
        if podman.run(command) != 0:
            raise LauncherError("cannot start benchmark container")
        stats = wait_for_server(
            "http://127.0.0.1:{}".format(options.port)
        )
        result["system"] = stats
        runs: List[Dict[str, object]] = []
        for index in range(options.runs):
            seed = options.seed + index
            prompt, _ = prepare_prompt(source, seed, output_prefix)
            started = time.monotonic()
            prompt_id = queue_prompt(
                "http://127.0.0.1:{}".format(options.port), prompt
            )
            history = wait_for_prompt(
                "http://127.0.0.1:{}".format(options.port), prompt_id
            )
            runs.append(
                {
                    "index": index,
                    "kind": "cold" if index == 0 else "warm",
                    "process_cache": "cold" if index == 0 else "warm",
                    "compiler_cache": (
                        "isolated-cold"
                        if index == 0 and options.cache_mode == "isolated"
                        else (
                            "persistent-reused"
                            if index == 0
                            else "warm"
                        )
                    ),
                    "seed": seed,
                    "prompt_id": prompt_id,
                    "wall_seconds": time.monotonic() - started,
                    "outputs": history.get("outputs", {}),
                }
            )
            result["runs"] = runs
            _write_result(result_path, result)
        result["status"] = "completed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["output_directory"] = str(
            StorageLayout(options.data_dir).comfyui
            / "output"
            / "rocmplete-benchmarks"
            / run_id
        )
        _write_result(result_path, result)
        return result_path
    except BaseException as error:
        primary_error = error
        result["status"] = "interrupted" if isinstance(
            error, KeyboardInterrupt
        ) else "failed"
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["error"] = str(error)
        result["container_log"] = _container_logs()
        _write_result(result_path, result)
        raise
    finally:
        _cleanup_benchmark_resources(isolated_cache_root, primary_error)


def _suite_signature(
    catalog: Catalog,
    bundles: Sequence[Bundle],
    options: BenchmarkOptions,
    image_id: str,
) -> str:
    definition = {
        "bundles": [
            {
                "identifier": bundle.identifier,
                "benchmark_sha256": catalog.benchmark(bundle.identifier).sha256,
                "benchmark_renderer": catalog.benchmark(
                    bundle.identifier
                ).renderer,
                "benchmark_rendered_sha256": catalog.benchmark(
                    bundle.identifier
                ).rendered_sha256,
                "workflow_sha256": catalog.workflow(bundle.workflow).rendered_sha256,
            }
            for bundle in bundles
        ],
        "configuration": {
            "image": options.image,
            "image_id": image_id,
            "profile": options.profile,
            "render_node": options.render_node,
            "runs": options.runs,
            "seed": options.seed,
            "memory_policy": options.memory_policy,
            "kernel_policy": options.kernel_policy,
            "cache_mode": options.cache_mode,
            "unconfined": options.unconfined,
        },
    }
    return hashlib.sha256(_canonical_json(definition)).hexdigest()


def _suite_result_summary(
    path: Path,
    *,
    bundle: str,
    run_id: str,
    image_id: str,
    options: BenchmarkOptions,
) -> Mapping[str, object]:
    result = _read_json_object(path, "benchmark result")
    expected = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": run_id,
        "bundle": bundle,
        "status": "completed",
        "profile": options.profile,
        "render_node": options.render_node,
        "memory_policy": options.memory_policy,
        "kernel_policy": options.kernel_policy,
        "cache_mode": options.cache_mode,
        "unconfined": options.unconfined,
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise LauncherError(
                "benchmark result {} does not belong to this suite: "
                "{} is {!r}, expected {!r}".format(
                    path, field, result.get(field), expected_value
                )
            )
    image = result.get("image")
    if not isinstance(image, dict) or image.get("id") != image_id:
        raise LauncherError(
            "benchmark result {} does not belong to this suite: "
            "image ID changed".format(path)
        )
    runs = result.get("runs", [])
    if not isinstance(runs, list) or len(runs) != options.runs:
        raise LauncherError(
            "benchmark result {} does not belong to this suite: "
            "expected {} runs".format(path, options.runs)
        )
    for index, run in enumerate(runs):
        if (
            not isinstance(run, dict)
            or run.get("index") != index
            or run.get("seed") != options.seed + index
        ):
            raise LauncherError(
                "benchmark result {} does not belong to this suite: "
                "run {} metadata changed".format(path, index)
            )
    cold = None
    warm = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        seconds = run.get("wall_seconds")
        if not isinstance(seconds, (int, float)):
            continue
        if run.get("kind") == "cold" and cold is None:
            cold = float(seconds)
        elif run.get("kind") == "warm":
            warm.append(float(seconds))
    return {
        "status": result.get("status", "unknown"),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_mean_seconds": (
            sum(warm) / len(warm) if warm else None
        ),
    }


def _format_seconds(value: object) -> str:
    return "{:.2f}".format(value) if isinstance(value, (int, float)) else "—"


def render_suite_markdown(suite: Mapping[str, object]) -> str:
    configuration = suite.get("configuration", {})
    if not isinstance(configuration, dict):
        configuration = {}
    lines = [
        "# ROCmplete benchmark suite",
        "",
        "- Suite: `{}`".format(suite.get("suite_id", "unknown")),
        "- Status: `{}`".format(suite.get("status", "unknown")),
        "- Profile: `{}`".format(configuration.get("profile", "unknown")),
        "- Image: `{}`".format(configuration.get("image", "unknown")),
        "- Runs per bundle: `{}`".format(configuration.get("runs", "unknown")),
        "- Memory policy: `{}`".format(
            configuration.get("memory_policy", "unknown")
        ),
        "- Kernel policy: `{}`".format(
            configuration.get("kernel_policy", "unknown")
        ),
        "",
        "| Bundle | Status | Cold (s) | Warm mean (s) | Result |",
        "|---|---:|---:|---:|---|",
    ]
    entries = suite.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        lines.append(
            "| `{}` | {} | {} | {} | `{}` |".format(
                str(entry.get("bundle", "unknown")).replace("|", "\\|"),
                str(entry.get("status", "unknown")).replace("|", "\\|"),
                _format_seconds(summary.get("cold_seconds")),
                _format_seconds(summary.get("warm_mean_seconds")),
                str(entry.get("result", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def render_suite_html(suite: Mapping[str, object]) -> str:
    configuration = suite.get("configuration", {})
    if not isinstance(configuration, dict):
        configuration = {}
    rows = []
    entries = suite.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        values = (
            entry.get("bundle", "unknown"),
            entry.get("status", "unknown"),
            _format_seconds(summary.get("cold_seconds")),
            _format_seconds(summary.get("warm_mean_seconds")),
            entry.get("result", ""),
        )
        rows.append(
            "<tr>{}</tr>".format(
                "".join(
                    "<td>{}</td>".format(html.escape(str(value)))
                    for value in values
                )
            )
        )
    title = "ROCmplete benchmark suite {}".format(
        suite.get("suite_id", "unknown")
    )
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}
th{{background:#eee}}code{{background:#eee;padding:.1rem .25rem}}
</style></head><body>
<h1>{title}</h1>
<p>Status: <code>{status}</code>; profile: <code>{profile}</code>;
image: <code>{image}</code>; runs per bundle: <code>{runs}</code>.</p>
<table><thead><tr><th>Bundle</th><th>Status</th><th>Cold (s)</th>
<th>Warm mean (s)</th><th>Result</th></tr></thead><tbody>
{rows}
</tbody></table></body></html>
""".format(
        title=html.escape(title),
        status=html.escape(str(suite.get("status", "unknown"))),
        profile=html.escape(str(configuration.get("profile", "unknown"))),
        image=html.escape(str(configuration.get("image", "unknown"))),
        runs=html.escape(str(configuration.get("runs", "unknown"))),
        rows="\n".join(rows),
    )


def write_suite_reports(
    suite_path: Path,
    suite: Mapping[str, object],
    report_format: str,
) -> Tuple[Path, ...]:
    outputs = []
    if report_format in ("markdown", "both"):
        path = suite_path.with_suffix(".md")
        path.write_text(render_suite_markdown(suite))
        outputs.append(path)
    if report_format in ("html", "both"):
        path = suite_path.with_suffix(".html")
        path.write_text(render_suite_html(suite))
        outputs.append(path)
    return tuple(outputs)


def run_benchmark_suite(
    catalog: Catalog,
    bundles: Sequence[Bundle],
    options: BenchmarkOptions,
    volume_suffix: str,
    resume_path: Optional[Path] = None,
    keep_going: bool = False,
    report_format: str = "both",
) -> Path:
    if not bundles:
        raise LauncherError("benchmark suite selection is empty")
    if options.dry_run:
        signature = _suite_signature(
            catalog, bundles, options, options.image
        )
        print("Suite bundles: {}".format(len(bundles)))
        print("Suite signature: {}".format(signature))
        for bundle in bundles:
            print("\n== {} ==".format(bundle.identifier))
            run_benchmark(catalog, bundle, options, volume_suffix)
        return (
            StorageLayout(options.data_dir).benchmarks
            / "suites"
            / "dry-run.json"
        )

    unavailable = []
    verification_store = VerificationStore.load(options.data_dir)
    for bundle in bundles:
        states = inspect_bundle(
            catalog, bundle, options.data_dir, verification_store
        )
        missing = [item for item in states if not content_status_ready(item)]
        if missing:
            unavailable.append(
                "{}: {}".format(
                    bundle.identifier,
                    ", ".join(
                        "{} ({})".format(
                            _content_destination(item), content_status_state(item)
                        )
                        for item in missing
                    ),
                )
            )
    if unavailable:
        raise LauncherError(
            "benchmark suite requires every selected bundle to be installed:\n  "
            + "\n  ".join(unavailable)
        )

    image_metadata = _image_metadata(options.image)
    image_id = image_metadata.get("id")
    if not isinstance(image_id, str) or not image_id:
        raise LauncherError(
            "cannot determine the immutable ID of benchmark image {}".format(
                options.image
            )
        )
    signature = _suite_signature(catalog, bundles, options, image_id)

    if resume_path is not None:
        suite_path = resume_path.resolve()
        suite = _read_json_object(suite_path, "benchmark suite")
        if suite.get("schema_version") != BENCHMARK_SUITE_SCHEMA_VERSION:
            raise LauncherError("benchmark suite has an unsupported schema")
        if suite.get("signature") != signature:
            raise LauncherError(
                "benchmark suite configuration or catalog inputs changed; "
                "start a new suite instead of resuming"
            )
    else:
        suite_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
            + uuid.uuid4().hex[:8]
        )
        suite_path = (
            StorageLayout(options.data_dir).benchmarks
            / "suites"
            / (suite_id + ".json")
        )
        suite = {
            "schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
            "suite_id": suite_id,
            "signature": signature,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "image": image_metadata,
            "configuration": {
                "image": options.image,
                "profile": options.profile,
                "render_node": options.render_node,
                "runs": options.runs,
                "seed": options.seed,
                "memory_policy": options.memory_policy,
                "kernel_policy": options.kernel_policy,
                "cache_mode": options.cache_mode,
                "unconfined": options.unconfined,
            },
            "entries": [
                {"bundle": bundle.identifier, "status": "pending"}
                for bundle in bundles
            ],
        }
        _write_result(suite_path, suite)

    suite_id = suite.get("suite_id")
    if not isinstance(suite_id, str) or not suite_id:
        raise LauncherError("benchmark suite identifier is invalid")
    entries = suite.get("entries")
    if not isinstance(entries, list):
        raise LauncherError("benchmark suite entries must be a list")
    by_bundle = {
        entry.get("bundle"): entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("bundle"), str)
    }
    if tuple(by_bundle) != tuple(bundle.identifier for bundle in bundles):
        raise LauncherError("benchmark suite selection does not match the catalog")

    failures = 0
    try:
        suite["status"] = "running"
        _write_result(suite_path, suite)
        for bundle in bundles:
            entry = by_bundle[bundle.identifier]
            run_id = "{}-{}".format(suite_id, bundle.identifier)
            result_path = (
                StorageLayout(options.data_dir).benchmarks
                / (run_id + ".json")
            )
            existing_result = entry.get("result")
            if entry.get("status") == "completed" and isinstance(
                existing_result, str
            ):
                recorded_path = Path(existing_result)
                if recorded_path.resolve() != result_path.resolve():
                    raise LauncherError(
                        "benchmark suite result path changed for {}".format(
                            bundle.identifier
                        )
                    )
                if result_path.exists():
                    entry["summary"] = _suite_result_summary(
                        result_path,
                        bundle=bundle.identifier,
                        run_id=run_id,
                        image_id=image_id,
                        options=options,
                    )
                    print(
                        "Skipping completed benchmark: {}".format(
                            bundle.identifier
                        )
                    )
                    continue
            entry["status"] = "running"
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            entry.pop("error", None)
            _write_result(suite_path, suite)
            entry["result"] = str(result_path)
            try:
                run_benchmark(
                    catalog,
                    bundle,
                    options,
                    volume_suffix,
                    run_id=run_id,
                )
                entry["status"] = "completed"
                entry["summary"] = _suite_result_summary(
                    result_path,
                    bundle=bundle.identifier,
                    run_id=run_id,
                    image_id=image_id,
                    options=options,
                )
            except LauncherError as error:
                failures += 1
                entry["status"] = "failed"
                entry["error"] = str(error)
                if not keep_going:
                    raise
            finally:
                entry["finished_at"] = datetime.now(timezone.utc).isoformat()
                _write_result(suite_path, suite)
        suite["status"] = "completed-with-failures" if failures else "completed"
        suite["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_result(suite_path, suite)
        write_suite_reports(suite_path, suite, report_format)
        return suite_path
    except BaseException as error:
        suite["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        suite["finished_at"] = datetime.now(timezone.utc).isoformat()
        suite["error"] = str(error)
        _write_result(suite_path, suite)
        write_suite_reports(suite_path, suite, report_format)
        raise
