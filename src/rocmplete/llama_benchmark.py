"""Structured one-shot llama.cpp benchmark execution."""

from __future__ import annotations

import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from . import podman
from .config import LLAMA_BENCHMARK_CONTAINER_NAME
from .errors import LauncherError
from .layout import StorageLayout

SCHEMA_VERSION = 3
COMPARISON_SCHEMA_VERSION = 1
CONTAINER_NAME = LLAMA_BENCHMARK_CONTAINER_NAME


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_result_path(data_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return StorageLayout(data_dir).llama_benchmarks / "{}-{}.json".format(
        stamp, uuid.uuid4().hex[:8]
    )


def _default_comparison_path(data_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return StorageLayout(data_dir).llama_benchmarks / (
        "{}-backend-comparison-{}.json".format(stamp, uuid.uuid4().hex[:8])
    )


def _remove_container() -> None:
    # ``podman rm --ignore`` is safe after normal --rm completion. Enter the
    # signal-masked removal path without a separate existence-check race.
    podman.remove_container(CONTAINER_NAME, stop_timeout=0)


def _parse_results(raw: str) -> Sequence[Mapping[str, object]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LauncherError(
            "llama-bench returned invalid JSON: {}".format(error)
        )
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, dict) for item in value)
    ):
        raise LauncherError(
            "llama-bench returned an empty or invalid result list"
        )
    return value


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        raise LauncherError(
            "refusing to replace existing benchmark result: {}".format(path)
        )
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex)
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        try:
            os.link(str(temporary), str(path))
        except FileExistsError:
            raise LauncherError(
                "refusing to replace existing benchmark result: {}".format(
                    path
                )
            )
    except OSError as error:
        raise LauncherError(
            "cannot write llama.cpp benchmark result {}: {}".format(
                path, error
            )
        )
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_result(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(
            "cannot read llama.cpp benchmark result {}: {}".format(
                path, error
            )
        )
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("application") != "llama-cpp"
    ):
        raise LauncherError(
            "llama.cpp benchmark result has an unsupported schema: {}".format(
                path
            )
        )
    return value


def _result_rate(
    result: Mapping[str, object],
    *,
    prompt_tokens: int,
    generation_tokens: int,
) -> float:
    rows = result.get("results")
    if not isinstance(rows, list):
        raise LauncherError("llama.cpp benchmark result has no result list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("n_prompt") == prompt_tokens
        and row.get("n_gen") == generation_tokens
    ]
    if len(matches) != 1:
        raise LauncherError(
            "llama.cpp benchmark result does not contain exactly one "
            "pp{}/tg{} row".format(prompt_tokens, generation_tokens)
        )
    rate = matches[0].get("avg_ts")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or rate <= 0
    ):
        raise LauncherError(
            "llama.cpp benchmark result contains an invalid token rate"
        )
    return float(rate)


def benchmark_rates(
    result: Mapping[str, object],
) -> Mapping[str, float]:
    parameters = result.get("parameters")
    if not isinstance(parameters, dict):
        raise LauncherError(
            "llama.cpp benchmark result has no parameter object"
        )
    prompt_tokens = parameters.get("prompt_tokens")
    generation_tokens = parameters.get("generation_tokens")
    if (
        isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens < 1
        or isinstance(generation_tokens, bool)
        or not isinstance(generation_tokens, int)
        or generation_tokens < 1
    ):
        raise LauncherError(
            "llama.cpp benchmark result has invalid token parameters"
        )
    prompt_rate = _result_rate(
        result,
        prompt_tokens=prompt_tokens,
        generation_tokens=0,
    )
    generation_rate = _result_rate(
        result,
        prompt_tokens=0,
        generation_tokens=generation_tokens,
    )
    return {
        "prompt_tokens_per_second": prompt_rate,
        "generation_tokens_per_second": generation_rate,
        "estimated_inference_seconds": (
            prompt_tokens / prompt_rate
            + generation_tokens / generation_rate
        ),
    }


def _throughput_comparison(
    rates: Mapping[str, Mapping[str, float]], key: str
) -> Mapping[str, object]:
    rocm = rates["rocm"][key]
    vulkan = rates["vulkan"][key]
    if rocm == vulkan:
        return {"winner": "tie", "faster_percent": 0.0}
    winner = "rocm" if rocm > vulkan else "vulkan"
    faster = max(rocm, vulkan)
    slower = min(rocm, vulkan)
    return {
        "winner": winner,
        "faster_percent": (faster / slower - 1.0) * 100.0,
    }


def _time_comparison(
    rates: Mapping[str, Mapping[str, float]]
) -> Mapping[str, object]:
    key = "estimated_inference_seconds"
    rocm = rates["rocm"][key]
    vulkan = rates["vulkan"][key]
    if rocm == vulkan:
        return {"winner": "tie", "lower_percent": 0.0}
    winner = "rocm" if rocm < vulkan else "vulkan"
    faster = min(rocm, vulkan)
    slower = max(rocm, vulkan)
    return {
        "winner": winner,
        "lower_percent": (slower - faster) / slower * 100.0,
    }


def write_backend_comparison(
    *,
    data_dir: Path,
    image: str,
    profile: str,
    render_nodes: Sequence[str],
    model: Mapping[str, object],
    parameters: Mapping[str, int],
    results: Mapping[str, Path],
    errors: Mapping[str, str],
    output: Optional[Path] = None,
) -> Tuple[Path, Mapping[str, object]]:
    entries: Dict[str, Mapping[str, object]] = {}
    rates: Dict[str, Mapping[str, float]] = {}
    for backend in ("rocm", "vulkan"):
        if backend in results:
            result_path = results[backend]
            result = _read_result(result_path)
            if result.get("backend") != backend:
                raise LauncherError(
                    "llama.cpp benchmark result backend does not match "
                    "comparison entry: {}".format(result_path)
                )
            backend_rates = benchmark_rates(result)
            rates[backend] = backend_rates
            entries[backend] = {
                "status": "pass",
                "result": str(result_path),
                **backend_rates,
            }
        else:
            entries[backend] = {
                "status": "fail",
                "error": errors.get(backend, "benchmark did not run"),
            }
    comparison: Optional[Mapping[str, object]] = None
    if set(rates) == {"rocm", "vulkan"}:
        comparison = {
            "prompt_processing": _throughput_comparison(
                rates, "prompt_tokens_per_second"
            ),
            "token_generation": _throughput_comparison(
                rates, "generation_tokens_per_second"
            ),
            "estimated_inference_time": _time_comparison(rates),
        }
    value: Mapping[str, object] = {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "created_at": _timestamp(),
        "application": "llama-cpp",
        "kind": "backend-comparison",
        "image": {
            "reference": image,
            "id": podman.image_id(image),
        },
        "profile": profile,
        "render_nodes": list(render_nodes),
        "gpu_split": "layer" if len(render_nodes) > 1 else "none",
        "model": dict(model),
        "parameters": dict(parameters),
        "backends": entries,
        "comparison": comparison,
    }
    destination = output or _default_comparison_path(data_dir)
    _write_result(destination, value)
    return destination, value


def run_llama_benchmark(
    command: Sequence[str],
    *,
    data_dir: Path,
    image: str,
    profile: str,
    backend: str,
    render_nodes: Sequence[str],
    model: Mapping[str, object],
    parameters: Mapping[str, int],
    output: Optional[Path] = None,
) -> Path:
    if podman.container_exists(CONTAINER_NAME):
        raise LauncherError(
            "benchmark container {!r} already exists".format(CONTAINER_NAME)
        )
    primary_error = None  # type: Optional[BaseException]
    try:
        raw = podman.capture_stdout(
            list(command), "llama.cpp benchmark failed"
        )
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            _remove_container()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            print(
                "WARNING: llama.cpp benchmark cleanup also failed after "
                "the primary error: {}".format(cleanup_error),
                file=sys.stderr,
            )
    results = _parse_results(raw)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _timestamp(),
        "application": "llama-cpp",
        "image": {
            "reference": image,
            "id": podman.image_id(image),
        },
        "profile": profile,
        "backend": backend,
        "render_nodes": list(render_nodes),
        "gpu_split": "layer" if len(render_nodes) > 1 else "none",
        "model": dict(model),
        "parameters": dict(parameters),
        "results": results,
    }
    destination = output or _default_result_path(data_dir)
    _write_result(destination, result)
    return destination
