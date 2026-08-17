"""Reproducible coding-agent evaluation on pinned public repositories."""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import podman
from .agent_models import (
    DWARFSTAR_MODEL,
    DWARFSTAR_PROVIDER_ID,
    PROVIDER_ID,
    agent_sampling_parameters,
    reasoning_native_value,
)
from .agent_sandbox import SANDBOX_RUNTIME
from .catalog import Catalog
from .config import APPLICATIONS, DWARFSTAR_DEFAULT_MODEL_BUNDLE
from .errors import LauncherError
from .layout import StorageLayout, validate_managed_parent
from .pi_agent import (
    create_launch_plan as create_pi_launch_plan,
    create_sandbox_plan as create_pi_sandbox_plan,
    prepare_state as prepare_pi_state,
    sandbox_paths as pi_sandbox_paths,
)
from .project import PROJECT_ROOT


DEFINITION_PATH = PROJECT_ROOT / "evaluations" / "coding" / "tasks.json"
SUITE_SCHEMA = 2
RESULT_SCHEMA = "rocmplete.coding-agent-evaluation.v2"
DEFAULT_CONTEXT = 131072
DEFAULT_PORT = 8187
ANSWER_FILE = "ROCMLETE_EVAL_ANSWER.md"
REVIEW_MIN_WORDS = 200
REVIEW_MAX_WORDS = 2000
_EVALUATION_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "ROCmplete Evaluation",
    "GIT_AUTHOR_EMAIL": "evaluation@invalid.local",
    "GIT_COMMITTER_NAME": "ROCmplete Evaluation",
    "GIT_COMMITTER_EMAIL": "evaluation@invalid.local",
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_NETWORK_COMMAND = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:curl|wget|ssh|scp|nc|ncat|telnet)\s"
    r"|\bgit\s+(?:clone|fetch|pull|remote)\b"
    r"|\bgo\s+(?:get|install)\b"
    r"|\b(?:pip|pip3)\s+install\b"
    r"|\bpython(?:3)?\s+-m\s+pip\s+install\b",
    re.IGNORECASE,
)
_PROMPT_METRIC = re.compile(
    r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens.*?"
    r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second",
    re.IGNORECASE,
)
_GENERATION_METRIC = re.compile(
    r"(?<!prompt )eval time\s*=.*?/\s*(\d+)\s+(?:runs|tokens).*?"
    r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second",
    re.IGNORECASE,
)
_REPOSITORY_TOOLCHAINS = {
    "fzr": "go",
    "nonet": "go",
    "reencode": "go",
    "rocmplete": "python-stdlib",
    "ssh-host-proxy": "go",
}
_PYTHON_DEPENDENCY_FILES = frozenset(
    (
        "constraints.txt",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
    )
)
_ALL_DEPENDENCY_FILES = frozenset(("go.mod", "go.sum")) | _PYTHON_DEPENDENCY_FILES


@dataclass(frozen=True)
class HiddenTest:
    resource: Path
    sha256: str
    destination: str


@dataclass(frozen=True)
class CodingTask:
    identifier: str
    kind: str
    toolchain: str
    repository: str
    remote: str
    base_commit: str
    base_tree: str
    reference_commit: str
    difficulty: str
    safety_critical: bool
    prompt: str
    hidden: Optional[HiddenTest] = None
    answer: str = ""


@dataclass(frozen=True)
class CodingSuite:
    identifier: str
    description: str
    fixture_instructions: str
    tasks: Tuple[CodingTask, ...]
    fingerprint: str


@dataclass(frozen=True)
class AgentEvaluationOptions:
    data_dir: Path
    preset: str = ""
    dwarfstar: bool = False
    tasks: Sequence[str] = ()
    repetitions: int = 1
    context: int = DEFAULT_CONTEXT
    thinking: str = "high"
    profile: str = "auto"
    backend: str = "rocm"
    port: int = DEFAULT_PORT
    render_nodes: Sequence[str] = ()
    output: Optional[Path] = None
    keep_going: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class PreparedAttempt:
    task: CodingTask
    repetition: int
    root: Path
    fixture: Path
    protected: Path


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LauncherError("coding evaluation {} must be a string".format(field))
    if "\x00" in value:
        raise LauncherError("coding evaluation {} contains NUL".format(field))
    return value


def _exact_keys(
    value: Mapping[str, object],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise LauncherError("{} is missing: {}".format(label, ", ".join(missing)))
    if unknown:
        raise LauncherError("{} has unknown fields: {}".format(label, ", ".join(unknown)))


def _resource_path(root: Path, relative: str, description: str) -> Path:
    candidate = root / relative
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        status = resolved.stat()
    except (OSError, ValueError) as error:
        raise LauncherError("invalid {} {}: {}".format(description, relative, error))
    if not stat.S_ISREG(status.st_mode):
        raise LauncherError("{} is not a regular file: {}".format(description, resolved))
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise LauncherError("cannot hash {}: {}".format(path, error))
    return digest.hexdigest()


def load_coding_suite(path: Path = DEFINITION_PATH) -> CodingSuite:
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise LauncherError("cannot load coding evaluation {}: {}".format(path, error))
    if not isinstance(raw, dict):
        raise LauncherError("coding evaluation definition must be an object")
    _exact_keys(
        raw,
        ("schema", "identifier", "description", "fixture_instructions", "tasks"),
        (),
        "coding evaluation definition",
    )
    if raw["schema"] != SUITE_SCHEMA:
        raise LauncherError(
            "unsupported coding evaluation schema: {}".format(raw["schema"])
        )
    identifier = _required_string(raw["identifier"], "identifier")
    if not _IDENTIFIER.fullmatch(identifier):
        raise LauncherError("invalid coding evaluation identifier: {}".format(identifier))
    task_values = raw["tasks"]
    if not isinstance(task_values, list) or not task_values:
        raise LauncherError("coding evaluation tasks must be a non-empty list")
    tasks: List[CodingTask] = []
    seen = set()
    root = path.parent.resolve(strict=True)
    fingerprint = hashlib.sha256(raw_bytes)
    for index, value in enumerate(task_values):
        label = "coding evaluation task {}".format(index)
        if not isinstance(value, dict):
            raise LauncherError("{} must be an object".format(label))
        _exact_keys(
            value,
            (
                "identifier",
                "kind",
                "toolchain",
                "repository",
                "remote",
                "base_commit",
                "base_tree",
                "reference_commit",
                "difficulty",
                "safety_critical",
                "prompt",
            ),
            ("hidden", "answer"),
            label,
        )
        task_id = _required_string(value["identifier"], "task identifier")
        if not _IDENTIFIER.fullmatch(task_id) or task_id in seen:
            raise LauncherError("invalid or duplicate coding task: {}".format(task_id))
        seen.add(task_id)
        kind = _required_string(value["kind"], "task kind")
        if kind not in ("implementation", "review"):
            raise LauncherError("unknown task kind for {}: {}".format(task_id, kind))
        toolchain = _required_string(value["toolchain"], "task toolchain")
        repository = _required_string(value["repository"], "repository")
        remote = _required_string(value["remote"], "remote")
        expected_remote = "https://github.com/fff7d1bc/{}.git".format(repository)
        if (
            remote != expected_remote
            or _REPOSITORY_TOOLCHAINS.get(repository) != toolchain
        ):
            raise LauncherError("task {} has an unreviewed repository remote".format(task_id))
        commits = []
        for field in ("base_commit", "base_tree", "reference_commit"):
            item = _required_string(value[field], field)
            if not _HEX40.fullmatch(item):
                raise LauncherError("task {} has invalid {}".format(task_id, field))
            commits.append(item)
        if not isinstance(value["safety_critical"], bool):
            raise LauncherError("task {} safety_critical must be boolean".format(task_id))
        hidden = None
        answer = ""
        if kind == "implementation":
            hidden_raw = value.get("hidden")
            if not isinstance(hidden_raw, dict):
                raise LauncherError("implementation task {} requires hidden tests".format(task_id))
            _exact_keys(
                hidden_raw,
                ("resource", "sha256", "destination"),
                (),
                "hidden tests for {}".format(task_id),
            )
            relative = _required_string(hidden_raw["resource"], "hidden resource")
            resource = _resource_path(root, relative, "hidden test")
            digest = _required_string(hidden_raw["sha256"], "hidden SHA-256")
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or _sha256(resource) != digest:
                raise LauncherError("hidden test hash mismatch for {}".format(task_id))
            destination = _required_string(hidden_raw["destination"], "hidden destination")
            destination_path = PurePosixPath(destination)
            safe_destination = (
                not destination_path.is_absolute()
                and bool(destination_path.parts)
                and all(part not in ("", ".", "..") for part in destination_path.parts)
            )
            if toolchain == "go":
                safe_destination = (
                    safe_destination
                    and len(destination_path.parts) == 1
                    and destination_path.name.endswith("_test.go")
                )
            elif toolchain == "python-stdlib":
                safe_destination = (
                    safe_destination
                    and len(destination_path.parts) == 2
                    and destination_path.parts[0] == "tests"
                    and destination_path.name.startswith("test_")
                    and destination_path.name.endswith(".py")
                )
            else:
                safe_destination = False
            if not safe_destination:
                raise LauncherError("unsafe hidden-test destination for {}".format(task_id))
            hidden = HiddenTest(resource=resource, sha256=digest, destination=destination)
            fingerprint.update(resource.read_bytes())
        else:
            answer = _required_string(value.get("answer"), "review answer")
            if answer != ANSWER_FILE:
                raise LauncherError(
                    "review task {} has an unsupported answer path".format(
                        task_id
                    )
                )
            if "hidden" in value:
                raise LauncherError("review task {} must not have hidden tests".format(task_id))
        tasks.append(
            CodingTask(
                identifier=task_id,
                kind=kind,
                toolchain=toolchain,
                repository=repository,
                remote=remote,
                base_commit=commits[0],
                base_tree=commits[1],
                reference_commit=commits[2],
                difficulty=_required_string(value["difficulty"], "difficulty"),
                safety_critical=value["safety_critical"],
                prompt=_required_string(value["prompt"], "prompt"),
                hidden=hidden,
                answer=answer,
            )
        )
    return CodingSuite(
        identifier=identifier,
        description=_required_string(raw["description"], "description"),
        fixture_instructions=_required_string(
            raw["fixture_instructions"], "fixture instructions"
        ),
        tasks=tuple(tasks),
        fingerprint=fingerprint.hexdigest(),
    )


def select_coding_tasks(
    suite: CodingSuite, identifiers: Sequence[str]
) -> Tuple[CodingTask, ...]:
    if not identifiers:
        return suite.tasks
    by_id = {task.identifier: task for task in suite.tasks}
    unknown = sorted(set(identifiers) - set(by_id))
    if unknown:
        raise LauncherError("unknown coding evaluation tasks: {}".format(", ".join(unknown)))
    selected = []
    seen = set()
    for identifier in identifiers:
        if identifier not in seen:
            selected.append(by_id[identifier])
            seen.add(identifier)
    return tuple(selected)


def _evaluation_root(data_dir: Path) -> Path:
    return StorageLayout(data_dir).agent_evaluations


def _prepare_directory(path: Path, data_dir: Path, description: str) -> None:
    root = _evaluation_root(data_dir)
    validate_managed_parent(path / ".boundary", root, data_dir, description)
    try:
        status = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(parents=True, mode=0o700)
        except OSError as error:
            raise LauncherError("cannot create {} {}: {}".format(description, path, error))
        return
    except OSError as error:
        raise LauncherError("cannot inspect {} {}: {}".format(description, path, error))
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise LauncherError("{} is not a real directory: {}".format(description, path))


def _run_capture(
    command: Sequence[str],
    description: str,
    *,
    cwd: Optional[Path] = None,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(command),
            cwd=str(cwd) if cwd else None,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as error:
        raise LauncherError("cannot {}: {}".format(description, error))


def _checked_output(
    command: Sequence[str], description: str, *, cwd: Optional[Path] = None
) -> bytes:
    result = _run_capture(command, description, cwd=cwd)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise LauncherError(
            "{}: {}".format(
                description,
                detail or "exit status {}".format(result.returncode),
            )
        )
    return result.stdout


def _ensure_source_mirror(task: CodingTask, data_dir: Path) -> Path:
    sources = _evaluation_root(data_dir) / "sources"
    _prepare_directory(sources, data_dir, "coding evaluation sources")
    mirror = sources / "{}.git".format(task.repository)
    try:
        status = mirror.lstat()
    except FileNotFoundError:
        temporary = sources / ".{}.{}.tmp".format(task.repository, uuid.uuid4().hex)
        result = _run_capture(
            ("git", "clone", "--mirror", task.remote, str(temporary)),
            "clone coding evaluation source {}".format(task.repository),
        )
        if result.returncode != 0:
            if temporary.exists() and temporary.parent == sources:
                shutil.rmtree(temporary)
            raise LauncherError(
                "cannot clone coding evaluation source {}: {}".format(
                    task.repository,
                    result.stderr.decode("utf-8", errors="replace").strip()
                    or "exit status {}".format(result.returncode),
                )
            )
        try:
            os.replace(temporary, mirror)
        except OSError as error:
            if temporary.exists() and temporary.parent == sources:
                shutil.rmtree(temporary)
            raise LauncherError("cannot install coding evaluation source mirror: {}".format(error))
    else:
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise LauncherError(
                "coding evaluation source is not a real directory: {}".format(
                    mirror
                )
            )
        configured = _checked_output(
            ("git", "-C", str(mirror), "config", "--get", "remote.origin.url"),
            "inspect coding evaluation source remote",
        ).decode("utf-8", errors="replace").strip()
        if configured != task.remote:
            raise LauncherError(
                "coding evaluation source remote changed for {}".format(
                    task.repository
                )
            )
    present = _run_capture(
        ("git", "-C", str(mirror), "cat-file", "-e", "{}^{{commit}}".format(task.base_commit)),
        "inspect coding evaluation base commit",
    )
    if present.returncode != 0:
        fetched = _run_capture(
            ("git", "-C", str(mirror), "fetch", "origin", task.base_commit),
            "fetch coding evaluation base commit",
        )
        if fetched.returncode != 0:
            raise LauncherError(
                "cannot fetch coding evaluation base commit {}".format(
                    task.base_commit
                )
            )
    tree = _checked_output(
        ("git", "-C", str(mirror), "rev-parse", "{}^{{tree}}".format(task.base_commit)),
        "resolve coding evaluation source tree",
    ).decode("ascii", errors="replace").strip()
    if tree != task.base_tree:
        raise LauncherError(
            "coding evaluation source tree mismatch for {}: {} != {}".format(
                task.identifier, tree, task.base_tree
            )
        )
    return mirror


def _extract_git_archive(archive: bytes, destination: Path) -> None:
    try:
        destination.mkdir(mode=0o700)
        handle = tarfile.open(fileobj=io.BytesIO(archive), mode="r:")
    except (OSError, tarfile.TarError) as error:
        raise LauncherError("cannot open coding evaluation source archive: {}".format(error))
    with handle:
        for member in handle.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise LauncherError("coding evaluation archive has an unsafe path")
            target = destination.joinpath(*relative.parts)
            try:
                target.relative_to(destination)
            except ValueError:
                raise LauncherError("coding evaluation archive path escapes fixture")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise LauncherError(
                    "coding evaluation archive has an unsupported file type: "
                    "{}".format(member.name)
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise LauncherError(
                    "cannot read coding evaluation archive member {}".format(
                        member.name
                    )
                )
            try:
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
            except OSError as error:
                raise LauncherError(
                    "cannot extract coding evaluation source {}: {}".format(
                        member.name, error
                    )
                )


def _git_fixture(
    fixture: Path, instructions: str, *, replace_existing_notes: bool = False
) -> None:
    agent_notes = fixture / "AGENTS.md"
    if agent_notes.exists() or agent_notes.is_symlink():
        if not replace_existing_notes:
            raise LauncherError("coding evaluation source unexpectedly contains AGENTS.md")
        try:
            status = agent_notes.lstat()
            if not stat.S_ISREG(status.st_mode):
                raise LauncherError(
                    "coding evaluation source has unsafe AGENTS.md"
                )
            agent_notes.unlink()
        except OSError as error:
            raise LauncherError(
                "cannot replace coding evaluation source instructions: {}".format(
                    error
                )
            )
    try:
        agent_notes.write_text(instructions)
    except OSError as error:
        raise LauncherError(
            "cannot write coding evaluation fixture instructions: {}".format(
                error
            )
        )
    commands = (
        ("git", "init", "--quiet"),
        ("git", "config", "user.name", "ROCmplete Evaluation"),
        ("git", "config", "user.email", "evaluation@invalid.local"),
        ("git", "add", "--all", "--", "."),
        ("git", "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Evaluation fixture"),
    )
    for command in commands:
        result = _run_capture(command, "initialize coding evaluation fixture", cwd=fixture)
        if result.returncode != 0:
            raise LauncherError(
                "cannot initialize coding evaluation fixture: {}".format(
                    result.stderr.decode("utf-8", errors="replace").strip()
                    or "exit status {}".format(result.returncode)
                )
            )


def _is_test_path(relative: Path) -> bool:
    return relative.name.endswith("_test.go") or (
        relative.parts[:1] == ("tests",)
        and relative.suffix == ".py"
        and relative.name.startswith("test")
    )


def _is_dependency_path(task: CodingTask, relative: Path) -> bool:
    if task.toolchain == "go":
        return relative.name in ("go.mod", "go.sum")
    return relative.name in _PYTHON_DEPENDENCY_FILES


def _protected_files(
    fixture: Path, task: Optional[CodingTask] = None
) -> Tuple[Path, ...]:
    selected = []
    for path in fixture.rglob("*"):
        if ".git" in path.relative_to(fixture).parts or not path.is_file():
            continue
        relative = path.relative_to(fixture)
        dependency = (
            _is_dependency_path(task, relative)
            if task is not None
            else relative.name in _ALL_DEPENDENCY_FILES
        )
        if _is_test_path(relative) or dependency or relative.name == "AGENTS.md":
            selected.append(relative)
    return tuple(sorted(selected, key=str))


def _snapshot_protected(
    fixture: Path, protected: Path, task: Optional[CodingTask] = None
) -> None:
    protected.mkdir(mode=0o700)
    for relative in _protected_files(fixture, task):
        target = protected / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture / relative, target)


def _go_environment(root: Path) -> Mapping[str, str]:
    module_cache = root / "cache" / "go-mod"
    build_cache = root / "cache" / "go-build"
    module_cache.mkdir(parents=True, exist_ok=True)
    build_cache.mkdir(parents=True, exist_ok=True)
    child = dict(os.environ)
    child.update(
        {
            "GOMODCACHE": str(module_cache),
            "GOCACHE": str(build_cache),
            "GOFLAGS": "-buildvcs=false",
        }
    )
    return child


def _task_environment(
    task: CodingTask, root: Path, worktree: Path
) -> Mapping[str, str]:
    if task.toolchain == "go":
        return _go_environment(root)
    child = dict(os.environ)
    child.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(worktree / "src"),
        }
    )
    return child


def _test_command(task: CodingTask) -> Tuple[str, ...]:
    if task.toolchain == "go":
        return ("go", "test", "-count=1", "./...")
    return ("python3", "-m", "unittest", "discover", "-s", "tests")


def _build_command(task: CodingTask) -> Tuple[str, ...]:
    if task.toolchain == "go":
        return ("go", "build", "./...")
    return (
        "python3",
        "-m",
        "compileall",
        "-q",
        "applications",
        "containers",
        "src/rocmplete",
        "tests",
        "tools",
    )


def _write_process_log(path: Path, result: subprocess.CompletedProcess) -> None:
    try:
        path.write_bytes(
            b"command exit: "
            + str(result.returncode).encode("ascii")
            + b"\n\nstdout:\n"
            + result.stdout
            + b"\n\nstderr:\n"
            + result.stderr
        )
    except OSError as error:
        raise LauncherError("cannot write evaluation log {}: {}".format(path, error))


def prepare_attempt(
    suite: CodingSuite,
    task: CodingTask,
    repetition: int,
    suite_root: Path,
    data_dir: Path,
) -> PreparedAttempt:
    attempt_root = suite_root / "tasks" / task.identifier / "attempt-{:02d}".format(repetition)
    _prepare_directory(attempt_root.parent, data_dir, "coding evaluation task results")
    _prepare_directory(attempt_root, data_dir, "coding evaluation attempt")
    fixture = attempt_root / "fixture"
    protected = attempt_root / "protected"
    if fixture.exists() or protected.exists():
        raise LauncherError("coding evaluation attempt already exists: {}".format(attempt_root))
    mirror = _ensure_source_mirror(task, data_dir)
    archive = _checked_output(
        ("git", "-C", str(mirror), "archive", "--format=tar", task.base_commit),
        "archive coding evaluation fixture",
    )
    _extract_git_archive(archive, fixture)
    _git_fixture(
        fixture,
        suite.fixture_instructions,
        replace_existing_notes=task.repository == "rocmplete",
    )
    _snapshot_protected(fixture, protected, task)
    environment = _task_environment(task, _evaluation_root(data_dir), fixture)
    if task.toolchain == "go":
        download = _run_capture(
            ("go", "mod", "download"),
            "prepare Go modules",
            cwd=fixture,
            env=environment,
        )
        _write_process_log(attempt_root / "module-prepare.log", download)
        if download.returncode != 0:
            raise LauncherError("cannot prepare Go modules for {}".format(task.identifier))
    baseline = _run_capture(
        _test_command(task),
        "run baseline tests",
        cwd=fixture,
        env=environment,
    )
    _write_process_log(attempt_root / "baseline.log", baseline)
    if baseline.returncode != 0:
        raise LauncherError("baseline tests fail for coding task {}".format(task.identifier))
    return PreparedAttempt(
        task=task,
        repetition=repetition,
        root=attempt_root,
        fixture=fixture,
        protected=protected,
    )


def _validate_agent_tree(fixture: Path) -> Tuple[bool, str]:
    files = 0
    total = 0
    for path in fixture.rglob("*"):
        relative = path.relative_to(fixture)
        if relative.parts[:1] == (".git",):
            continue
        try:
            status = path.lstat()
        except OSError as error:
            return False, "cannot inspect {}: {}".format(relative, error)
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            return False, "unsupported file type: {}".format(relative)
        files += 1
        total += status.st_size
        if status.st_size > 20 * 1024 * 1024:
            return False, "file exceeds 20 MiB: {}".format(relative)
        if files > 5000 or total > 100 * 1024 * 1024:
            return False, "worktree exceeds evaluation size limits"
    return True, ""


def _status_entries(fixture: Path) -> Tuple[Tuple[str, str], ...]:
    output = _checked_output(
        ("git", "status", "--porcelain=v1", "--untracked-files=all", "-z"),
        "inspect coding evaluation worktree",
        cwd=fixture,
    )
    records = output.split(b"\x00")
    entries = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        if len(record) < 4:
            raise LauncherError("git returned malformed coding evaluation status")
        code = record[:2].decode("ascii", errors="replace")
        path = record[3:].decode("utf-8", errors="surrogateescape")
        entries.append((code, path))
        if code[:1] in ("R", "C"):
            if index >= len(records) or not records[index]:
                raise LauncherError("git returned malformed rename status")
            index += 1
    return tuple(entries)


def _capture_patch(attempt: PreparedAttempt) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    entries = _status_entries(attempt.fixture)
    for code, path in entries:
        if code == "??":
            result = _run_capture(
                ("git", "add", "--intent-to-add", "--", path),
                "prepare untracked evaluation diff",
                cwd=attempt.fixture,
            )
            if result.returncode != 0:
                raise LauncherError(
                    "cannot include untracked evaluation path in diff: "
                    "{}".format(path)
                )
    patch = _checked_output(
        ("git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"),
        "capture coding evaluation patch",
        cwd=attempt.fixture,
    )
    if len(patch) > 20 * 1024 * 1024:
        raise LauncherError("coding evaluation patch exceeds 20 MiB")
    try:
        (attempt.root / "agent.patch").write_bytes(patch)
        (attempt.root / "status.json").write_text(json.dumps(entries, indent=2) + "\n")
    except OSError as error:
        raise LauncherError("cannot preserve coding evaluation patch: {}".format(error))
    return hashlib.sha256(patch).hexdigest(), entries


def _copy_for_grading(attempt: PreparedAttempt) -> Path:
    grade = attempt.root / "grading"
    if grade.exists() or grade.is_symlink():
        raise LauncherError("coding evaluation grading path already exists")
    shutil.copytree(attempt.fixture, grade, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    for source in attempt.protected.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(attempt.protected)
        if _is_test_path(relative):
            continue
        target = grade / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.copy2(source, target)
    return grade


def _run_grade_command(
    command: Sequence[str], grade: Path, environment: Mapping[str, str], log: Path
) -> int:
    result = _run_capture(command, "grade coding evaluation", cwd=grade, env=environment)
    _write_process_log(log, result)
    return result.returncode


def _dependency_changed(
    task: CodingTask, entries: Sequence[Tuple[str, str]]
) -> bool:
    return any(_is_dependency_path(task, Path(path)) for _, path in entries)


def _generated_artifacts(
    task: CodingTask, entries: Sequence[Tuple[str, str]]
) -> Tuple[str, ...]:
    # Reviewed Go repositories are single-main-package projects. A bare root
    # file named after the repository is the default output from `go build`,
    # not source. Retain the artifact in evidence but do not accept the patch.
    if task.toolchain != "go":
        return ()
    return tuple(path for _, path in entries if path == task.repository)


def grade_implementation(
    attempt: PreparedAttempt,
    *,
    pi_exit: int,
    network_attempts: Sequence[str],
    data_dir: Path,
) -> Mapping[str, object]:
    valid, invalid_reason = _validate_agent_tree(attempt.fixture)
    if not valid:
        return {"outcome": "invalid", "reason": invalid_reason, "pi_exit": pi_exit}
    patch_sha256, entries = _capture_patch(attempt)
    grade = _copy_for_grading(attempt)
    environment = dict(
        _task_environment(attempt.task, _evaluation_root(data_dir), grade)
    )
    if attempt.task.toolchain == "go":
        environment.update({"GOPROXY": "off", "GOSUMDB": "off"})
    regression = _run_grade_command(
        _test_command(attempt.task),
        grade,
        environment,
        attempt.root / "regression.log",
    )
    hidden = attempt.task.hidden
    if hidden is None:
        raise LauncherError(
            "implementation task {} has no hidden test".format(
                attempt.task.identifier
            )
        )
    if _sha256(hidden.resource) != hidden.sha256:
        raise LauncherError("hidden test changed while grading {}".format(attempt.task.identifier))
    hidden_destination = grade / hidden.destination
    if hidden_destination.exists() or hidden_destination.is_symlink():
        raise LauncherError(
            "hidden-test destination collision for {}".format(
                attempt.task.identifier
            )
        )
    hidden_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hidden.resource, hidden_destination)
    hidden_exit = _run_grade_command(
        _test_command(attempt.task),
        grade,
        environment,
        attempt.root / "hidden.log",
    )
    build_exit = _run_grade_command(
        _build_command(attempt.task),
        grade,
        environment,
        attempt.root / "build.log",
    )
    dependency_changed = _dependency_changed(attempt.task, entries)
    generated_artifacts = _generated_artifacts(attempt.task, entries)
    solved = (
        pi_exit == 0
        and regression == 0
        and hidden_exit == 0
        and build_exit == 0
        and not dependency_changed
        and not generated_artifacts
        and not network_attempts
    )
    return {
        "outcome": "solved" if solved else "unsolved",
        "pi_exit": pi_exit,
        "regression_exit": regression,
        "hidden_exit": hidden_exit,
        "build_exit": build_exit,
        "dependency_changed": dependency_changed,
        "generated_artifacts": list(generated_artifacts),
        "network_attempts": list(network_attempts),
        "patch_sha256": patch_sha256,
        "changed_paths": [path for _, path in entries],
    }


def grade_review(
    attempt: PreparedAttempt, *, pi_exit: int, network_attempts: Sequence[str]
) -> Mapping[str, object]:
    valid, invalid_reason = _validate_agent_tree(attempt.fixture)
    if not valid:
        return {"outcome": "invalid", "reason": invalid_reason, "pi_exit": pi_exit}
    patch_sha256, entries = _capture_patch(attempt)
    changed = [path for _, path in entries]
    unexpected = [path for path in changed if path != attempt.task.answer]
    answer = attempt.fixture / attempt.task.answer
    try:
        answer_text = answer.read_text()
    except OSError as error:
        answer_text = ""
        answer_error = str(error)
    else:
        answer_error = ""
    words = len(answer_text.split())
    cited_files = sorted(set(re.findall(r"\b[A-Za-z0-9_./-]+\.go\b", answer_text)))
    recorded = (
        pi_exit == 0
        and not unexpected
        and not network_attempts
        and not answer_error
        and REVIEW_MIN_WORDS <= words <= REVIEW_MAX_WORDS
        and len(cited_files) >= 2
    )
    try:
        (attempt.root / "answer.md").write_text(answer_text)
    except OSError as error:
        raise LauncherError("cannot preserve coding review answer: {}".format(error))
    return {
        "outcome": "review-pending" if recorded else "invalid",
        "pi_exit": pi_exit,
        "answer_error": answer_error,
        "answer_words": words,
        "cited_files": cited_files,
        "unexpected_paths": unexpected,
        "network_attempts": list(network_attempts),
        "patch_sha256": patch_sha256,
        "changed_paths": changed,
    }


def _walk_command_strings(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("command", "cmd") and isinstance(child, str):
                yield child
            yield from _walk_command_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_command_strings(child)


def transcript_network_attempts(path: Path) -> Tuple[str, ...]:
    attempts = []
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ()
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        for command in _walk_command_strings(value):
            if _NETWORK_COMMAND.search(command):
                attempts.append(command[:500])
    return tuple(attempts)


def transcript_usage(path: Path) -> Mapping[str, int]:
    totals: Dict[str, int] = {
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache_read": 0,
        "cache_write": 0,
    }
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return totals
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict) or value.get("type") not in (
            "message_end",
            "assistant_message_end",
        ):
            continue
        message = value.get("message", value)
        if not isinstance(message, dict):
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        for source, destination in (
            ("input", "input"),
            ("output", "output"),
            ("reasoning", "reasoning"),
            ("cacheRead", "cache_read"),
            ("cacheWrite", "cache_write"),
        ):
            amount = usage.get(source, 0)
            if isinstance(amount, int) and amount >= 0:
                totals[destination] += amount
    return totals


def _evaluation_sandbox_environment(
    environ: Mapping[str, str],
) -> Mapping[str, str]:
    child = dict(environ)
    child.update(_EVALUATION_GIT_IDENTITY)
    child["TERM"] = "dumb"
    return child


def _run_pi(
    attempt: PreparedAttempt,
    options: AgentEvaluationOptions,
    catalog: Catalog,
) -> Mapping[str, object]:
    provider = DWARFSTAR_PROVIDER_ID if options.dwarfstar else PROVIDER_ID
    model = DWARFSTAR_MODEL if options.dwarfstar else options.preset
    arguments = (
        "--provider",
        provider,
        "--model",
        model,
        "--thinking",
        options.thinking,
        "--print",
        "--mode",
        "json",
        "--no-session",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--tools",
        "read,bash,edit,write",
        attempt.task.prompt,
    )
    plan = create_pi_launch_plan(
        catalog,
        options.data_dir,
        options.port if not options.dwarfstar else DEFAULT_PORT,
        arguments,
        os.environ,
        dwarfstar_port=options.port if options.dwarfstar else 8000,
    )
    paths = pi_sandbox_paths(options.data_dir)
    prepare_pi_state(plan, paths, options.data_dir)
    sandbox_environment = _evaluation_sandbox_environment(os.environ)
    read_only_mounts: Tuple[Tuple[Path, Path], ...] = ()
    if attempt.task.toolchain == "go":
        module_cache = _evaluation_root(options.data_dir) / "cache" / "go-mod"
        sandbox_module_cache = SANDBOX_RUNTIME / "go-mod"
        read_only_mounts = ((module_cache, sandbox_module_cache),)
        toolchain_environment = {
            "GOMODCACHE": str(sandbox_module_cache),
            "GOCACHE": "/tmp/go-build",
            "GOPROXY": "off",
            "GOSUMDB": "off",
            "GOFLAGS": "-buildvcs=false",
        }
    else:
        toolchain_environment = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(attempt.fixture / "src"),
        }
    sandbox = create_pi_sandbox_plan(
        plan,
        options.data_dir,
        attempt.fixture,
        sandbox_environment,
        read_only_mounts=read_only_mounts,
        extra_environment=toolchain_environment,
    )
    stdout_path = attempt.root / "pi.jsonl"
    stderr_path = attempt.root / "pi.stderr.log"
    started = time.monotonic()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                list(sandbox.command),
                cwd=str(attempt.fixture),
                env=dict(sandbox.environment),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                exit_code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    except OSError as error:
        raise LauncherError("cannot run Pi coding evaluation: {}".format(error))
    wall = time.monotonic() - started
    return {
        "exit": exit_code,
        "wall_seconds": round(wall, 3),
        "usage": transcript_usage(stdout_path),
        "network_attempts": list(transcript_network_attempts(stdout_path)),
    }


def _server_command(options: AgentEvaluationOptions) -> Tuple[str, ...]:
    command = [str(PROJECT_ROOT / "rocmplete"), "run"]
    if options.dwarfstar:
        command.extend(("dwarfstar", "server"))
    else:
        command.extend(
            (
                "llama-cpp",
                "server",
                "--preset",
                options.preset,
                "--backend",
                options.backend,
            )
        )
    command.extend(
        (
            "--profile",
            options.profile,
            "--data-dir",
            str(options.data_dir),
            "--context",
            str(options.context),
            "--listen",
            "127.0.0.1",
            "--port",
            str(options.port),
            "--detach",
        )
    )
    for render_node in options.render_nodes:
        command.extend(("--render-node", render_node))
    return tuple(command)


def _container_name(options: AgentEvaluationOptions) -> str:
    return APPLICATIONS["dwarfstar" if options.dwarfstar else "llama-cpp"].container_name


def _capture_container_logs(name: str) -> bytes:
    result = _run_capture(("podman", "logs", name), "capture coding evaluation server logs")
    return result.stdout + (b"\n[stderr]\n" + result.stderr if result.stderr else b"")


def _container_running(name: str) -> bool:
    result = _run_capture(
        ("podman", "inspect", "--format", "{{.State.Running}}", name),
        "inspect coding evaluation server",
    )
    return result.returncode == 0 and result.stdout.strip() == b"true"


def _server_readiness_url(options: AgentEvaluationOptions) -> str:
    path = "/v1/models" if options.dwarfstar else "/health"
    return "http://127.0.0.1:{}{}".format(options.port, path)


def _wait_for_server(options: AgentEvaluationOptions) -> None:
    url = _server_readiness_url(options)
    name = _container_name(options)
    while True:
        if not _container_running(name):
            detail = _capture_container_logs(name).decode("utf-8", errors="replace")[-4000:]
            raise LauncherError(
                "coding evaluation model server stopped during startup:\n{}".format(
                    detail
                )
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)


def parse_server_metrics(log: bytes) -> Mapping[str, object]:
    text = log.decode("utf-8", errors="replace")
    prompt = _PROMPT_METRIC.findall(text)
    generation = _GENERATION_METRIC.findall(text)
    result: Dict[str, object] = {}
    if prompt:
        result["prompt_tokens"] = sum(int(tokens) for tokens, _ in prompt)
        result["prompt_tokens_per_second"] = float(prompt[-1][1])
    if generation:
        result["generation_tokens"] = sum(int(tokens) for tokens, _ in generation)
        result["generation_tokens_per_second"] = float(generation[-1][1])
    return result


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
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise LauncherError("cannot checkpoint coding evaluation {}: {}".format(path, error))
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
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise LauncherError("coding evaluation result already exists: {}".format(path))
    except OSError as error:
        raise LauncherError("cannot create coding evaluation result {}: {}".format(path, error))


def _project_revision() -> str:
    result = _run_capture(
        ("git", "rev-parse", "HEAD"),
        "inspect ROCmplete revision",
        cwd=PROJECT_ROOT,
    )
    if result.returncode == 0:
        return result.stdout.decode("ascii", errors="replace").strip()
    return "unknown"


def _pi_version() -> str:
    result = _run_capture(("pi", "--version"), "inspect Pi version")
    if result.returncode == 0:
        return result.stdout.decode("utf-8", errors="replace").strip()
    return "unknown"


def _model_identity(
    catalog: Catalog, options: AgentEvaluationOptions
) -> Mapping[str, object]:
    if options.dwarfstar:
        bundle = catalog.bundle(DWARFSTAR_DEFAULT_MODEL_BUNDLE)
        artifacts = catalog.bundle_artifacts(bundle)
        return {
            "provider": DWARFSTAR_PROVIDER_ID,
            "identifier": DWARFSTAR_MODEL,
            "bundle": bundle.identifier,
            "artifacts": [
                {
                    "identifier": artifact.identifier,
                    "destination": artifact.destination,
                    "size": artifact.size,
                    "sha256": artifact.sha256,
                }
                for artifact in artifacts
            ],
            "context": options.context,
            "thinking": options.thinking,
            "backend": options.backend,
            "sampling": {},
        }
    preset = catalog.llama_preset(options.preset)
    native_reasoning = reasoning_native_value(preset, options.thinking)
    identifiers = [preset.artifact]
    if preset.draft_artifact:
        identifiers.append(preset.draft_artifact)
    return {
        "provider": PROVIDER_ID,
        "identifier": preset.identifier,
        "bundle": preset.bundle,
        "artifacts": [
            {
                "identifier": identifier,
                "destination": catalog.artifact(identifier).destination,
                "size": catalog.artifact(identifier).size,
                "sha256": catalog.artifact(identifier).sha256,
            }
            for identifier in identifiers
        ],
        "context": options.context,
        "thinking": options.thinking,
        "reasoning": {
            "control": preset.reasoning_control or "none",
            "client_level": options.thinking,
            "native_value": native_reasoning,
        },
        "backend": options.backend,
        "sampling": dict(
            agent_sampling_parameters(preset.identifier, options.thinking)
        ),
        "speculative_type": preset.speculative_type,
        "draft_tokens": preset.draft_tokens_for_backend(options.backend),
        "draft_tokens_by_backend": dict(preset.draft_tokens_by_backend),
        "flash_attention": dict(preset.flash_attention),
        "kv_cache": dict(preset.kv_cache),
    }


def default_result_path(data_dir: Path, model: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", model)
    return _evaluation_root(data_dir) / "results" / "{}-{}.json".format(timestamp, safe_model)


def render_agent_evaluation_markdown(result: Mapping[str, object]) -> str:
    model = result.get("model", {})
    tasks = result.get("tasks", [])
    lines = [
        "# Coding-agent evaluation",
        "",
        "- Suite: `{}`".format(result.get("suite", "unknown")),
        "- Model: `{}`".format(
            model.get("identifier", "unknown")
            if isinstance(model, dict)
            else "unknown"
        ),
        "- Harness: Pi `{}`".format(
            result.get("harness", {}).get("version", "unknown")
            if isinstance(result.get("harness"), dict)
            else "unknown"
        ),
        "- Status: `{}`".format(result.get("status", "unknown")),
        "",
        "| Task | Kind | Difficulty | Outcome | Wall time | Output tokens |",
        "|---|---|---:|---|---:|---:|",
    ]
    solved = 0
    implementations = 0
    if isinstance(tasks, list):
        for task in tasks:
            if not isinstance(task, dict):
                continue
            attempts = task.get("attempts", [])
            for attempt in attempts if isinstance(attempts, list) else []:
                if not isinstance(attempt, dict):
                    continue
                grade = attempt.get("grade", {})
                harness = attempt.get("harness", {})
                outcome = grade.get("outcome", "unknown") if isinstance(grade, dict) else "unknown"
                if task.get("kind") == "implementation":
                    implementations += 1
                    if outcome == "solved":
                        solved += 1
                usage = harness.get("usage", {}) if isinstance(harness, dict) else {}
                lines.append(
                    "| `{}` | {} | {} | **{}** | {:.1f}s | {} |".format(
                        task.get("identifier", "unknown"),
                        task.get("kind", "unknown"),
                        task.get("difficulty", "unknown"),
                        outcome,
                        float(harness.get("wall_seconds", 0)) if isinstance(harness, dict) else 0,
                        usage.get("output", 0) if isinstance(usage, dict) else 0,
                    )
                )
    lines.extend(("", "Implementation solve rate: **{}/{}**.".format(solved, implementations), ""))
    return "\n".join(lines)


def _write_agent_report(path: Path, result: Mapping[str, object]) -> None:
    try:
        with path.open("x") as handle:
            handle.write(render_agent_evaluation_markdown(result))
    except FileExistsError:
        raise LauncherError("coding evaluation report already exists: {}".format(path))
    except OSError as error:
        raise LauncherError("cannot write coding evaluation report: {}".format(error))


def run_agent_evaluation(
    catalog: Catalog, options: AgentEvaluationOptions
) -> Tuple[Path, Mapping[str, object]]:
    suite = load_coding_suite()
    tasks = select_coding_tasks(suite, options.tasks)
    if options.repetitions < 1:
        raise LauncherError("--repetitions must be at least 1")
    if options.context < 4096:
        raise LauncherError("--context must be at least 4096")
    if options.dwarfstar == bool(options.preset):
        raise LauncherError("choose exactly one of --preset or --dwarfstar")
    if options.dwarfstar and options.backend != "rocm":
        raise LauncherError("DwarfStar coding evaluation supports only ROCm")
    if options.dwarfstar and options.thinking not in ("off", "high"):
        raise LauncherError(
            "DwarfStar coding evaluation supports --thinking off or high"
        )
    if options.preset:
        preset = catalog.llama_preset(options.preset)
        if not preset.agent_tools:
            raise LauncherError(
                "llama.cpp preset is not reviewed for agent tools: {}".format(
                    options.preset
                )
            )
        reasoning_native_value(preset, options.thinking)
    model = DWARFSTAR_MODEL if options.dwarfstar else options.preset
    if options.dry_run:
        print("Coding-agent evaluation")
        print("  Suite       {} ({})".format(suite.identifier, suite.fingerprint))
        print("  Model       {}".format(model))
        print("  Harness     Pi")
        print("  Context     {}".format(options.context))
        if options.dwarfstar:
            print("  Thinking    {} (DwarfStar mode)".format(options.thinking))
        else:
            preset = catalog.llama_preset(options.preset)
            print(
                "  Thinking    {} ({}={})".format(
                    options.thinking,
                    preset.reasoning_control or "none",
                    reasoning_native_value(preset, options.thinking),
                )
            )
        print("  Tasks       {}".format(", ".join(task.identifier for task in tasks)))
        print("  Repetitions {}".format(options.repetitions))
        print("  Server      {}".format(" ".join(_server_command(options))))
        return Path(), {"status": "dry-run"}
    name = _container_name(options)
    if podman.container_exists(name):
        raise LauncherError(
            "coding evaluation requires stopped container {!r}".format(name)
        )
    root = _evaluation_root(options.data_dir)
    result_path = options.output or default_result_path(options.data_dir, model)
    if result_path.suffix != ".json":
        raise LauncherError("coding evaluation result must use a .json suffix")
    validate_managed_parent(result_path, root, options.data_dir, "coding evaluation result")
    report_path = result_path.with_suffix(".md")
    if result_path.exists() or result_path.is_symlink():
        raise LauncherError("coding evaluation result already exists: {}".format(result_path))
    if report_path.exists() or report_path.is_symlink():
        raise LauncherError("coding evaluation report already exists: {}".format(report_path))
    for path, description in (
        (root, "coding evaluation data"),
        (root / "runs", "coding evaluation runs"),
        (root / "results", "coding evaluation results"),
        (root / "cache", "coding evaluation cache"),
    ):
        _prepare_directory(path, options.data_dir, description)
    suite_id = uuid.uuid4().hex
    suite_root = root / "runs" / suite_id
    _prepare_directory(suite_root, options.data_dir, "coding evaluation run")
    result: Dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "suite": suite.identifier,
        "suite_fingerprint": suite.fingerprint,
        "suite_id": suite_id,
        "status": "preparing",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "project_revision": _project_revision(),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "harness": {"name": "pi", "version": _pi_version()},
        "conditions": "custom",
        "model": dict(_model_identity(catalog, options)),
        "runtime": {
            "application": "dwarfstar" if options.dwarfstar else "llama-cpp",
            "image": APPLICATIONS[
                "dwarfstar" if options.dwarfstar else "llama-cpp"
            ].image,
            "profile": options.profile,
            "render_nodes": list(options.render_nodes),
            "port": options.port,
        },
        "tasks": [],
        "run_root": str(suite_root),
    }
    task_results: List[Dict[str, object]] = []
    for task in tasks:
        task_result: Dict[str, object] = {
            "identifier": task.identifier,
            "kind": task.kind,
            "toolchain": task.toolchain,
            "repository": task.repository,
            "base_commit": task.base_commit,
            "reference_commit": task.reference_commit,
            "difficulty": task.difficulty,
            "safety_critical": task.safety_critical,
            "attempts": [],
        }
        task_results.append(task_result)
    result["tasks"] = task_results
    _create_json(result_path, result)
    prepared: List[PreparedAttempt] = []
    try:
        for task, task_result in zip(tasks, task_results):
            attempts = task_result["attempts"]
            if not isinstance(attempts, list):
                raise LauncherError(
                    "coding evaluation task has invalid checkpoint state"
                )
            for repetition in range(1, options.repetitions + 1):
                attempt = prepare_attempt(
                    suite, task, repetition, suite_root, options.data_dir
                )
                prepared.append(attempt)
                attempts.append(
                    {
                        "repetition": repetition,
                        "status": "prepared",
                        "path": str(attempt.root),
                    }
                )
                _atomic_json(result_path, result)
    except LauncherError as error:
        result.update(
            {
                "status": "infrastructure-failed",
                "error": str(error),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(result_path, result)
        _write_agent_report(report_path, result)
        raise
    server_start = _run_capture(
        _server_command(options), "start coding evaluation model server"
    )
    try:
        (suite_root / "server-start.log").write_bytes(
            server_start.stdout + b"\n[stderr]\n" + server_start.stderr
        )
    except OSError as error:
        if podman.container_exists(name):
            podman.remove_container(name, stop_timeout=10)
        result.update(
            {
                "status": "infrastructure-failed",
                "error": str(error),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(result_path, result)
        _write_agent_report(report_path, result)
        raise LauncherError(
            "cannot write coding evaluation server-start log: {}".format(
                error
            )
        )
    if server_start.returncode != 0:
        error = "cannot start coding evaluation model server"
        if podman.container_exists(name):
            podman.remove_container(name, stop_timeout=10)
        result.update(
            {
                "status": "infrastructure-failed",
                "error": error,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _atomic_json(result_path, result)
        _write_agent_report(report_path, result)
        raise LauncherError(error)
    prior_logs = b""
    interrupted = False
    infrastructure_error = ""
    result["status"] = "running"
    _atomic_json(result_path, result)
    try:
        _wait_for_server(options)
        runtime = result["runtime"]
        if not isinstance(runtime, dict):
            raise LauncherError(
                "coding evaluation has invalid runtime checkpoint state"
            )
        runtime["image_id"] = podman.image_id(str(runtime["image"]))
        _atomic_json(result_path, result)
        prior_logs = _capture_container_logs(name)
        prepared_by_key = {
            (item.task.identifier, item.repetition): item for item in prepared
        }
        for task_result in task_results:
            attempts = task_result["attempts"]
            if not isinstance(attempts, list):
                raise LauncherError(
                    "coding evaluation task has invalid checkpoint state"
                )
            for attempt_result in attempts:
                if not isinstance(attempt_result, dict):
                    raise LauncherError(
                        "coding evaluation attempt has invalid checkpoint state"
                    )
                key = (
                    str(task_result["identifier"]),
                    int(attempt_result["repetition"]),
                )
                attempt = prepared_by_key[key]
                attempt_result["status"] = "running"
                _atomic_json(result_path, result)
                try:
                    harness = _run_pi(attempt, options, catalog)
                    current_logs = _capture_container_logs(name)
                    delta = (
                        current_logs[len(prior_logs) :]
                        if current_logs.startswith(prior_logs)
                        else current_logs
                    )
                    prior_logs = current_logs
                    (attempt.root / "server.log").write_bytes(delta)
                    harness["server_metrics"] = parse_server_metrics(delta)
                    network_attempts = harness.get("network_attempts", [])
                    if not isinstance(network_attempts, list):
                        raise LauncherError(
                            "Pi returned invalid network audit state"
                        )
                    if attempt.task.kind == "implementation":
                        grade = grade_implementation(
                            attempt,
                            pi_exit=int(harness["exit"]),
                            network_attempts=network_attempts,
                            data_dir=options.data_dir,
                        )
                    else:
                        grade = grade_review(
                            attempt,
                            pi_exit=int(harness["exit"]),
                            network_attempts=network_attempts,
                        )
                    attempt_result.update(
                        {
                            "status": "complete",
                            "harness": harness,
                            "grade": grade,
                        }
                    )
                except KeyboardInterrupt:
                    attempt_result["status"] = "interrupted"
                    interrupted = True
                    raise
                except (LauncherError, OSError) as error:
                    attempt_result.update(
                        {
                            "status": "infrastructure-failed",
                            "error": str(error),
                        }
                    )
                    infrastructure_error = str(error)
                    if not options.keep_going:
                        raise LauncherError(str(error))
                finally:
                    _atomic_json(result_path, result)
    except KeyboardInterrupt:
        interrupted = True
    except LauncherError as error:
        infrastructure_error = str(error)
    finally:
        if podman.container_exists(name):
            try:
                all_logs = _capture_container_logs(name)
                (suite_root / "server.log").write_bytes(all_logs)
            except (LauncherError, OSError):
                pass
            podman.remove_container(name, stop_timeout=10)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    if interrupted:
        result["status"] = "interrupted"
    elif infrastructure_error:
        result["status"] = "infrastructure-failed"
        result["error"] = infrastructure_error
    else:
        result["status"] = "complete"
    _atomic_json(result_path, result)
    _write_agent_report(report_path, result)
    if interrupted:
        raise KeyboardInterrupt
    if infrastructure_error:
        raise LauncherError(infrastructure_error)
    return result_path, result
