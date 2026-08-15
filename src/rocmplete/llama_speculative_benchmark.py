"""Checkpointed server-side speculative-depth benchmarks for llama.cpp."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Dict,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import podman
from .config import LLAMA_SPECULATIVE_BENCHMARK_CONTAINER_NAME
from .errors import LauncherError
from .layout import StorageLayout


RESULT_SCHEMA = "rocmplete.llama-speculative-depth-sweep.v1"
PROMPT_GENERATOR_VERSION = 1
DEFAULT_CONTEXT_DEPTHS = (4096, 32768, 65536, 122880)
DEFAULT_SERVER_CONTEXT = 131072
DEFAULT_GENERATION_TOKENS = 512
DEFAULT_REPETITIONS = 3
DEFAULT_SEED = 42
DECISION_THRESHOLD_PERCENT = 3.0
CONTAINER_NAME = LLAMA_SPECULATIVE_BENCHMARK_CONTAINER_NAME

_SYSTEM_PROMPT = (
    "You are a senior Go engineer working in a deterministic performance "
    "evaluation. Study the supplied repository packet before answering. "
    "Reason carefully about concurrency, cancellation, ownership, and tests."
)
_TASK_PROMPT = """The repository needs a bounded work queue with these semantics:

- Push blocks while the queue is full and returns the caller's context error
  if cancellation wins before the item is committed.
- Pop blocks while the queue is empty, returns io.EOF after Close, and never
  loses an item that was committed before Close.
- Close is idempotent and wakes every blocked caller without leaking a
  goroutine.

Explain the synchronization invariants, identify the races a naive condition-
variable implementation would contain, and propose an idiomatic Go
implementation plus focused tests. Be concrete enough that another engineer
could apply the change directly."""


@dataclass(frozen=True)
class SpeculativeBenchmarkOptions:
    data_dir: Path
    preset: str
    image: str
    image_id: str
    source_identity: str
    profile: str
    backend: str
    render_nodes: Sequence[str]
    port: int
    context: int
    thinking: str
    native_reasoning: str
    speculative_type: str
    incumbent_depth: int
    depths: Sequence[int]
    context_depths: Sequence[int]
    repetitions: int
    generation_tokens: int
    seed: int
    sampling: Mapping[str, object]
    model: Mapping[str, object]
    commands: Mapping[int, Sequence[str]] = field(repr=False)
    output: Optional[Path] = None
    resume: Optional[Path] = None
    keep_going: bool = False


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def default_result_path(data_dir: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return StorageLayout(data_dir).llama_benchmarks / (
        "{}-speculative-depth-sweep-{}.json".format(
            stamp, uuid.uuid4().hex[:8]
        )
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".{}.".format(path.name),
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise LauncherError(
            "cannot checkpoint speculative benchmark {}: {}".format(path, error)
        )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _create_json(path: Path, value: Mapping[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise LauncherError(
            "speculative benchmark result already exists: {}".format(path)
        )
    except OSError as error:
        raise LauncherError(
            "cannot create speculative benchmark result {}: {}".format(
                path, error
            )
        )


def _definition(options: SpeculativeBenchmarkOptions) -> Mapping[str, object]:
    return {
        "policy_version": 1,
        "source_identity": options.source_identity,
        "image": {
            "reference": options.image,
            "id": options.image_id,
        },
        "profile": options.profile,
        "backend": options.backend,
        "render_nodes": list(options.render_nodes),
        "model": dict(options.model),
        "preset": options.preset,
        "speculative_type": options.speculative_type,
        "incumbent_depth": options.incumbent_depth,
        "depths": list(options.depths),
        "context_depths": list(options.context_depths),
        "repetitions": options.repetitions,
        "generation_tokens": options.generation_tokens,
        "seed": options.seed,
        "server_context": options.context,
        "parallel": 1,
        "thinking": {
            "client": options.thinking,
            "native": options.native_reasoning,
        },
        "sampling": dict(options.sampling),
        "prompt_generator_version": PROMPT_GENERATOR_VERSION,
        "fresh_server_per_trial": True,
    }


def _trial_plan(
    options: SpeculativeBenchmarkOptions,
) -> List[MutableMapping[str, object]]:
    trials = []
    for repetition in range(1, options.repetitions + 1):
        seed = options.seed + repetition - 1
        for context_depth in options.context_depths:
            for depth in options.depths:
                identifier = "d{}-c{}-s{}-r{}".format(
                    depth, context_depth, seed, repetition
                )
                trials.append(
                    {
                        "identifier": identifier,
                        "depth": depth,
                        "context_depth": context_depth,
                        "seed": seed,
                        "repetition": repetition,
                        "status": "pending",
                    }
                )
    # A digest sort avoids a systematic thermal advantage for shallow depths
    # while remaining stable across Python versions and resume attempts.
    trials.sort(
        key=lambda item: hashlib.sha256(
            "schedule-v1:{}".format(item["identifier"]).encode("ascii")
        ).digest()
    )
    return trials


def _new_result(
    options: SpeculativeBenchmarkOptions,
) -> MutableMapping[str, object]:
    definition = _definition(options)
    return {
        "schema": RESULT_SCHEMA,
        "suite_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + uuid.uuid4().hex[:8],
        "fingerprint": _digest(definition),
        "definition": definition,
        "status": "preparing",
        "started_at": _timestamp(),
        "workloads": [],
        "trials": _trial_plan(options),
        "summary": {},
    }


def _load_result(
    path: Path, options: SpeculativeBenchmarkOptions
) -> MutableMapping[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError(
            "cannot read speculative benchmark result {}: {}".format(path, error)
        )
    if not isinstance(value, dict) or value.get("schema") != RESULT_SCHEMA:
        raise LauncherError(
            "speculative benchmark result has an unsupported schema: {}".format(
                path
            )
        )
    expected_definition = _definition(options)
    expected_fingerprint = _digest(expected_definition)
    if (
        value.get("fingerprint") != expected_fingerprint
        or value.get("definition") != expected_definition
    ):
        raise LauncherError(
            "speculative benchmark resume inputs do not match {}".format(path)
        )
    expected_trials = {
        item["identifier"]: item for item in _trial_plan(options)
    }
    trials = value.get("trials")
    if not isinstance(trials, list) or {
        item.get("identifier")
        for item in trials
        if isinstance(item, dict)
    } != set(expected_trials):
        raise LauncherError(
            "speculative benchmark result has an invalid trial plan: {}".format(
                path
            )
        )
    for item in trials:
        if not isinstance(item, dict):
            raise LauncherError(
                "speculative benchmark result has an invalid trial entry"
            )
        expected = expected_trials[str(item["identifier"])]
        if any(
            item.get(key) != expected[key]
            for key in ("depth", "context_depth", "seed", "repetition")
        ):
            raise LauncherError(
                "speculative benchmark trial identity changed: {}".format(
                    item["identifier"]
                )
            )
        status = item.get("status")
        if status not in (
            "pending",
            "running",
            "complete",
            "failed",
            "interrupted",
        ):
            raise LauncherError(
                "speculative benchmark trial has invalid status: {}".format(status)
            )
        if status != "complete":
            for key in tuple(item):
                if key not in expected:
                    del item[key]
            item.update(expected)
        else:
            for key in (
                "completion_tokens",
                "prompt_tokens",
                "drafted_tokens",
                "accepted_draft_tokens",
            ):
                if (
                    isinstance(item.get(key), bool)
                    or not isinstance(item.get(key), int)
                    or int(item[key]) < 0
                ):
                    raise LauncherError(
                        "completed speculative benchmark trial {} has "
                        "invalid {}".format(item["identifier"], key)
                    )
            for key in (
                "predicted_ms",
                "prompt_ms",
                "request_seconds",
                "generation_tokens_per_second",
            ):
                value_number = item.get(key)
                if (
                    isinstance(value_number, bool)
                    or not isinstance(value_number, (int, float))
                    or not math.isfinite(float(value_number))
                    or float(value_number) <= 0
                ):
                    raise LauncherError(
                        "completed speculative benchmark trial {} has "
                        "invalid {}".format(item["identifier"], key)
                    )
            response_sha256 = item.get("response_sha256")
            if (
                not isinstance(response_sha256, str)
                or len(response_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in response_sha256
                )
            ):
                raise LauncherError(
                    "completed speculative benchmark trial {} has invalid "
                    "response digest".format(item["identifier"])
                )
    workloads = value.get("workloads")
    if not isinstance(workloads, list):
        raise LauncherError(
            "speculative benchmark result has invalid workload calibration"
        )
    if workloads:
        _validate_workloads(workloads, options)
    return value


def _source_line(index: int) -> str:
    mask = (1 << 64) - 1
    first = (index * 0x9E3779B97F4A7C15 + 0xD1B54A32D192ED03) & mask
    second = (first ^ (first >> 29) ^ 0x94D049BB133111EB) & mask
    shift = index % 63 + 1
    return (
        "func transform{index:06d}(input uint64) uint64 {{ "
        "value := bits.RotateLeft64(input^0x{first:016x}, {shift}); "
        "if value&0x{second:016x} == 0 {{ return value + 0x{first:016x} }}; "
        "return value ^ 0x{second:016x} }}\n"
    ).format(
        index=index,
        first=first,
        second=second,
        shift=shift,
    )


def benchmark_messages(line_count: int, seed: int) -> Tuple[Mapping[str, str], ...]:
    if line_count < 0:
        raise LauncherError("benchmark prompt line count must not be negative")
    repository = "".join(_source_line(index) for index in range(line_count))
    user = (
        "Benchmark trajectory seed: {}\n\n"
        "Repository packet:\n```go\npackage packet\n\nimport \"math/bits\"\n\n{}"
        "```\n\n{}"
    ).format(seed, repository, _TASK_PROMPT)
    return (
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )


def _request_json(
    url: str,
    payload: Optional[Mapping[str, object]] = None,
    *,
    timeout: Optional[float] = None,
) -> Mapping[str, object]:
    data = None if payload is None else _canonical_json(payload)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[-4000:]
        raise LauncherError(
            "llama.cpp server rejected {}: HTTP {}{}".format(
                url,
                error.code,
                ": {}".format(detail) if detail else "",
            )
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise LauncherError(
            "cannot reach llama.cpp server at {}: {}".format(url, error)
        )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LauncherError(
            "llama.cpp server returned invalid JSON from {}: {}".format(url, error)
        )
    if not isinstance(value, dict):
        raise LauncherError(
            "llama.cpp server returned a non-object response from {}".format(url)
        )
    return value


def _container_log_tail(limit: int = 12000) -> str:
    if not podman.container_exists(CONTAINER_NAME):
        return ""
    try:
        raw = podman.capture_bytes(
            ["podman", "logs", CONTAINER_NAME],
            "cannot capture speculative benchmark server logs",
        )
    except LauncherError as error:
        return str(error)
    return raw.decode("utf-8", errors="replace")[-limit:]


def _container_running() -> bool:
    if not podman.container_exists(CONTAINER_NAME):
        return False
    try:
        return podman.capture(
            ["podman", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
            "cannot inspect speculative benchmark server",
        ) == "true"
    except LauncherError:
        return False


def _wait_for_server(port: int) -> None:
    url = "http://127.0.0.1:{}/health".format(port)
    last_notice = time.monotonic()
    while True:
        if not _container_running():
            detail = _container_log_tail()
            raise LauncherError(
                "speculative benchmark server stopped during startup{}".format(
                    ":\n{}".format(detail) if detail else ""
                )
            )
        try:
            value = _request_json(url, timeout=2.0)
            if value.get("status") == "ok":
                return
        except LauncherError:
            pass
        now = time.monotonic()
        if now - last_notice >= 30.0:
            print("  still waiting for the model server...", flush=True)
            last_notice = now
        time.sleep(0.5)


@contextmanager
def _running_server(command: Sequence[str], port: int) -> Iterator[float]:
    if podman.container_exists(CONTAINER_NAME):
        raise LauncherError(
            "speculative benchmark container {!r} already exists".format(
                CONTAINER_NAME
            )
        )
    primary_error = None  # type: Optional[BaseException]
    started = time.monotonic()
    try:
        status = podman.run_quiet_stdout(list(command))
        if status != 0:
            raise LauncherError(
                "cannot start speculative benchmark server (exit status {})".format(
                    status
                )
            )
        _wait_for_server(port)
        yield time.monotonic() - started
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            podman.remove_container(CONTAINER_NAME, stop_timeout=10)
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            print(
                "WARNING: speculative benchmark cleanup also failed after "
                "the primary error: {}".format(cleanup_error),
                file=sys.stderr,
            )


def _prompt_token_count(
    port: int,
    messages: Sequence[Mapping[str, str]],
    native_reasoning: str,
) -> int:
    template_payload: Dict[str, object] = {
        "messages": list(messages),
        "add_generation_prompt": True,
    }
    if native_reasoning == "off":
        template_payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif native_reasoning != "on":
        template_payload["reasoning_effort"] = native_reasoning
    rendered = _request_json(
        "http://127.0.0.1:{}/apply-template".format(port),
        template_payload,
    )
    prompt = rendered.get("prompt")
    if not isinstance(prompt, str):
        raise LauncherError("llama.cpp apply-template response has no prompt")
    tokenized = _request_json(
        "http://127.0.0.1:{}/tokenize".format(port),
        {
            "content": prompt,
            "add_special": False,
            "parse_special": True,
        },
    )
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in tokens
    ):
        raise LauncherError("llama.cpp tokenize response has no token list")
    return len(tokens)


def _calibrate_line_count(
    port: int,
    target: int,
    seed: int,
    native_reasoning: str,
) -> Tuple[int, int]:
    measured: Dict[int, int] = {}

    def count(lines: int) -> int:
        if lines not in measured:
            measured[lines] = _prompt_token_count(
                port,
                benchmark_messages(lines, seed),
                native_reasoning,
            )
        return measured[lines]

    if count(0) >= target:
        return 0, measured[0]
    low = 0
    high = 64
    while count(high) < target:
        low = high
        high *= 2
        if high > 1_000_000:
            raise LauncherError(
                "cannot calibrate {}-token speculative benchmark prompt".format(
                    target
                )
            )
    while high - low > 1:
        middle = (low + high) // 2
        if count(middle) < target:
            low = middle
        else:
            high = middle
    candidates = (low, high)
    best = min(candidates, key=lambda lines: (abs(count(lines) - target), lines))
    return best, count(best)


def _calibrate_workloads(
    options: SpeculativeBenchmarkOptions,
) -> List[Mapping[str, object]]:
    depth = (
        options.incumbent_depth
        if options.incumbent_depth in options.commands
        else options.depths[0]
    )
    workloads = []
    print(
        "Calibrating deterministic prompts with draft depth {}...".format(depth),
        flush=True,
    )
    with _running_server(options.commands[depth], options.port):
        for target in options.context_depths:
            line_count, prompt_tokens = _calibrate_line_count(
                options.port,
                target,
                options.seed,
                options.native_reasoning,
            )
            messages = benchmark_messages(line_count, options.seed)
            workload = {
                "target_prompt_tokens": target,
                "line_count": line_count,
                "calibrated_prompt_tokens": prompt_tokens,
                "prompt_sha256": _digest(messages),
            }
            workloads.append(workload)
            print(
                "  target {:>6}: {:>6} tokens, {:>5} generated source lines".format(
                    target, prompt_tokens, line_count
                ),
                flush=True,
            )
    return workloads


def _validate_workloads(
    workloads: Sequence[object], options: SpeculativeBenchmarkOptions
) -> Mapping[int, Mapping[str, object]]:
    by_context: Dict[int, Mapping[str, object]] = {}
    for raw in workloads:
        if not isinstance(raw, dict):
            raise LauncherError(
                "speculative benchmark workload calibration has a non-object entry"
            )
        target = raw.get("target_prompt_tokens")
        line_count = raw.get("line_count")
        calibrated = raw.get("calibrated_prompt_tokens")
        prompt_sha256 = raw.get("prompt_sha256")
        if (
            isinstance(target, bool)
            or not isinstance(target, int)
            or target not in options.context_depths
            or target in by_context
            or isinstance(line_count, bool)
            or not isinstance(line_count, int)
            or line_count < 0
            or line_count > 1_000_000
            or isinstance(calibrated, bool)
            or not isinstance(calibrated, int)
            or calibrated < 1
            or calibrated + options.generation_tokens > options.context
        ):
            raise LauncherError(
                "speculative benchmark workload calibration is invalid"
            )
        expected_digest = _digest(benchmark_messages(line_count, options.seed))
        if prompt_sha256 != expected_digest:
            raise LauncherError(
                "speculative benchmark calibrated prompt digest changed"
            )
        by_context[target] = raw
    if set(by_context) != set(options.context_depths):
        raise LauncherError(
            "speculative benchmark workload calibration is incomplete"
        )
    return by_context


def _finite_number(value: object, field_name: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0)
    ):
        raise LauncherError(
            "llama.cpp response has invalid {}".format(field_name)
        )
    return float(value)


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LauncherError(
            "llama.cpp response has invalid {}".format(field_name)
        )
    return value


def parse_trial_response(
    response: Mapping[str, object],
    *,
    request_seconds: float,
    startup_seconds: float,
    prompt_sha256: str,
) -> Mapping[str, object]:
    choices = response.get("choices")
    usage = response.get("usage")
    timings = response.get("timings")
    if (
        not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
        or not isinstance(usage, dict)
        or not isinstance(timings, dict)
    ):
        raise LauncherError("llama.cpp response lacks choices, usage, or timings")
    choice = choices[0]
    message = choice.get("message")
    finish_reason = choice.get("finish_reason")
    if not isinstance(message, dict) or not isinstance(finish_reason, str):
        raise LauncherError("llama.cpp response has an invalid assistant message")
    prompt_tokens = _integer(usage.get("prompt_tokens"), "prompt_tokens")
    completion_tokens = _integer(
        usage.get("completion_tokens"), "completion_tokens", minimum=1
    )
    predicted_tokens = _integer(
        timings.get("predicted_n"), "timings.predicted_n", minimum=1
    )
    drafted = _integer(timings.get("draft_n"), "timings.draft_n")
    accepted = _integer(
        timings.get("draft_n_accepted"), "timings.draft_n_accepted"
    )
    if accepted > drafted:
        raise LauncherError("llama.cpp accepted more draft tokens than it proposed")
    cache_n = _integer(timings.get("cache_n", 0), "timings.cache_n")
    details = usage.get("prompt_tokens_details", {})
    cached_tokens = 0
    if isinstance(details, dict):
        cached_tokens = _integer(
            details.get("cached_tokens", 0), "usage.cached_tokens"
        )
    if cache_n or cached_tokens:
        raise LauncherError(
            "fresh speculative benchmark server unexpectedly reused a prompt cache"
        )
    predicted_ms = _finite_number(
        timings.get("predicted_ms"), "timings.predicted_ms", positive=True
    )
    prompt_ms = _finite_number(
        timings.get("prompt_ms"), "timings.prompt_ms", positive=True
    )
    generation_rate = _finite_number(
        timings.get("predicted_per_second"),
        "timings.predicted_per_second",
        positive=True,
    )
    prompt_rate = _finite_number(
        timings.get("prompt_per_second"),
        "timings.prompt_per_second",
        positive=True,
    )
    if predicted_tokens != completion_tokens:
        raise LauncherError(
            "llama.cpp usage and timing completion-token counts disagree"
        )
    response_identity = {
        "message": message,
        "finish_reason": finish_reason,
    }
    return {
        "request_seconds": round(request_seconds, 6),
        "startup_seconds": round(startup_seconds, 6),
        "prompt_sha256": prompt_sha256,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_ms": prompt_ms,
        "predicted_ms": predicted_ms,
        "prompt_tokens_per_second": prompt_rate,
        "generation_tokens_per_second": generation_rate,
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "acceptance_percent": (100.0 * accepted / drafted) if drafted else 0.0,
        "finish_reason": finish_reason,
        "response_sha256": _digest(response_identity),
        "message": message,
        "system_fingerprint": response.get("system_fingerprint", ""),
    }


def _trial_request(
    options: SpeculativeBenchmarkOptions,
    trial: Mapping[str, object],
    workload: Mapping[str, object],
) -> Tuple[Mapping[str, object], float, str]:
    line_count = int(workload["line_count"])
    seed = int(trial["seed"])
    messages = benchmark_messages(line_count, seed)
    prompt_sha256 = _digest(messages)
    payload: Dict[str, object] = {
        "model": options.preset,
        "messages": list(messages),
        "max_tokens": options.generation_tokens,
        "seed": seed,
        "stream": False,
        **dict(options.sampling),
    }
    if options.native_reasoning == "off":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif options.native_reasoning != "on":
        payload["reasoning_effort"] = options.native_reasoning
    started = time.monotonic()
    response = _request_json(
        "http://127.0.0.1:{}/v1/chat/completions".format(options.port),
        payload,
    )
    request_seconds = time.monotonic() - started
    return response, request_seconds, prompt_sha256


def _run_trial(
    options: SpeculativeBenchmarkOptions,
    trial: Mapping[str, object],
    workload: Mapping[str, object],
) -> Mapping[str, object]:
    depth = int(trial["depth"])
    with _running_server(options.commands[depth], options.port) as startup_seconds:
        response, request_seconds, prompt_sha256 = _trial_request(
            options, trial, workload
        )
        log_tail = _container_log_tail(4000)
        parsed = dict(
            parse_trial_response(
                response,
                request_seconds=request_seconds,
                startup_seconds=startup_seconds,
                prompt_sha256=prompt_sha256,
            )
        )
        parsed["server_log_tail"] = log_tail
        return parsed


def summarize_trials(
    trials: Sequence[Mapping[str, object]], incumbent_depth: int
) -> Mapping[str, object]:
    complete = [item for item in trials if item.get("status") == "complete"]
    by_depth: Dict[int, List[Mapping[str, object]]] = {}
    for item in complete:
        by_depth.setdefault(int(item["depth"]), []).append(item)
    depth_summaries: Dict[str, Mapping[str, object]] = {}
    for depth, entries in sorted(by_depth.items()):
        predicted_tokens = sum(int(item["completion_tokens"]) for item in entries)
        predicted_ms = sum(float(item["predicted_ms"]) for item in entries)
        timed_generation_tokens = sum(
            float(item["generation_tokens_per_second"])
            * float(item["predicted_ms"])
            / 1000.0
            for item in entries
        )
        prompt_tokens = sum(int(item["prompt_tokens"]) for item in entries)
        prompt_ms = sum(float(item["prompt_ms"]) for item in entries)
        drafted = sum(int(item["drafted_tokens"]) for item in entries)
        accepted = sum(int(item["accepted_draft_tokens"]) for item in entries)
        request_seconds = sum(float(item["request_seconds"]) for item in entries)
        depth_summaries[str(depth)] = {
            "complete_trials": len(entries),
            "generation_tokens": predicted_tokens,
            "timed_generation_tokens": timed_generation_tokens,
            "generation_seconds": predicted_ms / 1000.0,
            "generation_tokens_per_second": (
                timed_generation_tokens / (predicted_ms / 1000.0)
            ),
            "prompt_tokens": prompt_tokens,
            "prompt_seconds": prompt_ms / 1000.0,
            "prompt_tokens_per_second": prompt_tokens / (prompt_ms / 1000.0),
            "request_seconds": request_seconds,
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "acceptance_percent": (100.0 * accepted / drafted) if drafted else 0.0,
        }
    expected_by_depth = len(trials) // len({int(item["depth"]) for item in trials})
    eligible = {
        depth: value
        for depth, value in depth_summaries.items()
        if int(value["complete_trials"]) == expected_by_depth
    }
    winner_depth = None
    improvement = None
    decision = "inconclusive"
    if eligible:
        winner_depth = int(
            max(
                eligible,
                key=lambda depth: float(
                    eligible[depth]["generation_tokens_per_second"]
                ),
            )
        )
        incumbent = eligible.get(str(incumbent_depth))
        if incumbent is not None:
            incumbent_rate = float(incumbent["generation_tokens_per_second"])
            winner_rate = float(
                eligible[str(winner_depth)]["generation_tokens_per_second"]
            )
            improvement = (winner_rate / incumbent_rate - 1.0) * 100.0
            decision = (
                "retest-candidate"
                if winner_depth != incumbent_depth
                and improvement >= DECISION_THRESHOLD_PERCENT
                else "keep-incumbent"
            )
    consistency: Dict[str, Set[str]] = {}
    for item in complete:
        key = "c{}-s{}".format(item["context_depth"], item["seed"])
        consistency.setdefault(key, set()).add(str(item["response_sha256"]))
    return {
        "complete_trials": len(complete),
        "total_trials": len(trials),
        "failed_trials": sum(item.get("status") == "failed" for item in trials),
        "depths": depth_summaries,
        "incumbent_depth": incumbent_depth,
        "winner_depth": winner_depth,
        "winner_improvement_percent": improvement,
        "decision_threshold_percent": DECISION_THRESHOLD_PERCENT,
        "screening_decision": decision,
        "response_digest_counts": {
            key: len(value) for key, value in sorted(consistency.items())
        },
    }


def run_speculative_benchmark(
    options: SpeculativeBenchmarkOptions,
) -> Tuple[Path, MutableMapping[str, object]]:
    if options.resume is not None:
        result_path = options.resume
        result = _load_result(result_path, options)
    else:
        result_path = options.output or default_result_path(options.data_dir)
        result = _new_result(options)
        _create_json(result_path, result)
    trials = result["trials"]
    if not isinstance(trials, list):
        raise LauncherError("speculative benchmark checkpoint has no trial list")
    if result.get("status") == "complete" and all(
        isinstance(item, dict) and item.get("status") == "complete"
        for item in trials
    ):
        return result_path, result
    result.pop("error", None)
    try:
        workloads = result.get("workloads")
        if not isinstance(workloads, list):
            raise LauncherError(
                "speculative benchmark checkpoint has invalid workloads"
            )
        if not workloads:
            workloads = _calibrate_workloads(options)
            result["workloads"] = workloads
            _atomic_json(result_path, result)
        by_context = _validate_workloads(workloads, options)
        result["status"] = "running"
        _atomic_json(result_path, result)
        total = len(trials)
        for index, trial in enumerate(trials, start=1):
            if trial.get("status") == "complete":
                continue
            print(
                "[{}/{}] depth {}, target {}, seed {}".format(
                    index,
                    total,
                    trial["depth"],
                    trial["context_depth"],
                    trial["seed"],
                ),
                flush=True,
            )
            trial["status"] = "running"
            trial["started_at"] = _timestamp()
            _atomic_json(result_path, result)
            try:
                metrics = _run_trial(
                    options,
                    trial,
                    by_context[int(trial["context_depth"])],
                )
                trial.update(metrics)
                trial["status"] = "complete"
                trial["finished_at"] = _timestamp()
                print(
                    "  {:.2f} t/s, accepted {}/{} ({:.1f}%), {:.2f}s request".format(
                        float(trial["generation_tokens_per_second"]),
                        int(trial["accepted_draft_tokens"]),
                        int(trial["drafted_tokens"]),
                        float(trial["acceptance_percent"]),
                        float(trial["request_seconds"]),
                    ),
                    flush=True,
                )
            except KeyboardInterrupt:
                trial["status"] = "interrupted"
                trial["finished_at"] = _timestamp()
                raise
            except (LauncherError, OSError) as error:
                trial["status"] = "failed"
                trial["error"] = str(error)
                trial["server_log_tail"] = _container_log_tail(4000)
                trial["finished_at"] = _timestamp()
                print(
                    "  failed: {}".format(error),
                    file=sys.stderr,
                    flush=True,
                )
                if not options.keep_going:
                    raise
            finally:
                result["summary"] = summarize_trials(
                    trials, options.incumbent_depth
                )
                _atomic_json(result_path, result)
        failed = any(item.get("status") == "failed" for item in trials)
        result["status"] = "failed" if failed else "complete"
        result.pop("error", None)
        result["finished_at"] = _timestamp()
        result["summary"] = summarize_trials(trials, options.incumbent_depth)
        _atomic_json(result_path, result)
        return result_path, result
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["finished_at"] = _timestamp()
        result["summary"] = summarize_trials(trials, options.incumbent_depth)
        _atomic_json(result_path, result)
        raise
    except (LauncherError, OSError) as error:
        result["status"] = "failed"
        result["error"] = str(error)
        result["finished_at"] = _timestamp()
        result["summary"] = summarize_trials(trials, options.incumbent_depth)
        _atomic_json(result_path, result)
        raise
