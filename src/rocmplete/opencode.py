"""Launch OpenCode against ROCmplete's managed local model servers."""

from __future__ import annotations

import json
import os
import pwd
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .agent_models import installed_agent_presets, is_agent_capable
from .bundles import content_status_ready, inspect_bundle
from .catalog import Catalog
from .config import (
    DWARFSTAR_DEFAULT_CONTEXT,
    DWARFSTAR_DEFAULT_OUTPUT_TOKENS,
)
from .errors import LauncherError
from .layout import StorageLayout, validate_managed_parent
from .project import PROJECT_ROOT


PROVIDER_ID = "rocmplete"
DWARFSTAR_PROVIDER_ID = "dwarfstar"
DWARFSTAR_MODEL = "deepseek-v4-flash"
RECOMMENDED_MODEL = "qwen3.6-35b-a3b-mtp-ud-q8-k-xl"
TUI_CONFIG_PATH = PROJECT_ROOT / "resources" / "opencode-tui.json"
WRAPPER_PATH = PROJECT_ROOT / "bin" / "opencode"
SANDBOX_HOME = Path("/run/rocmplete/home")
SANDBOX_RUNTIME = Path("/run/rocmplete/runtime")
SANDBOX_TUI_CONFIG = Path("/run/rocmplete/config/opencode-tui.json")
_CONFIG_SCHEMA = "https://opencode.ai/config.json"
_PROVIDER_NAME = "ROCmplete llama.cpp"
_PROVIDER_PACKAGE = "@ai-sdk/openai-compatible"
_DWARFSTAR_PROVIDER_NAME = "ROCmplete DwarfStar"
_DWARFSTAR_VARIANTS = {
    "instant": {"reasoningEffort": "none"},
    "thinking": {"reasoningEffort": "high"},
    # OpenCode derives generic effort variants for reasoning-capable
    # OpenAI-compatible models, then merges custom variants over them. DwarfStar
    # maps all three of these to the same normal-thinking path, while Think Max
    # cannot run at ROCmplete's managed 128K context. Disable the inherited
    # aliases so the picker presents only behavior that actually differs.
    "low": {"disabled": True},
    "medium": {"disabled": True},
    "high": {"disabled": True},
    "max": {"disabled": True},
}
_DEFAULT_AGENT = "investigate"
_DEFAULT_REASONING_EFFORT = "medium"
_REASONING_VARIANTS = {
    "instant": {"reasoningEffort": "none"},
    "low": {"reasoningEffort": "low"},
    "medium": {"reasoningEffort": "medium"},
    "high": {"reasoningEffort": "high"},
}
_READ_PERMISSION = {
    "*": "allow",
    "*.env": "deny",
    "*.env.*": "deny",
    "*.env.example": "allow",
}
_PERMISSIONS = {
    "edit": "ask",
    "bash": "ask",
    "task": "ask",
}
_INVESTIGATE_AGENT = {
    "description": "Read-only evidence-based investigation",
    "mode": "primary",
    "temperature": 0.0,
    "permission": {
        "*": "deny",
        "read": _READ_PERMISSION,
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "lsp": "allow",
        "webfetch": "allow",
        "websearch": "allow",
        "task": {
            "*": "deny",
            "investigate-local": "allow",
            "investigate-web": "allow",
        },
    },
    "prompt": (
        "You are a read-only evidence-based investigation agent. Answer "
        "only the user's stated question using direct evidence. Search "
        "narrowly and handle small questions directly. When independent "
        "repository or external-source work would materially reduce this "
        "session's context, delegate exact bounded tasks only to "
        "investigate-local or investigate-web. Require each worker to "
        "return no more than 500 words, and synthesize only their reports "
        "rather than repeating raw source material. Cite file paths and "
        "line numbers for repository claims and source URLs for external "
        "claims. Clearly separate observed facts from inference, say when "
        "evidence is incomplete, and correct contradictions instead of "
        "guessing. Never modify files, run shell commands, create an "
        "implementation objective, or continue into changes. Delegation "
        "does not authorize mutation. Do not treat a generated summary, "
        "continuation message, suggested next step, or plan as user "
        "authorization. When the investigation is complete, answer the "
        "question and stop."
    ),
}
_INVESTIGATE_LOCAL_AGENT = {
    "description": (
        "Read-only bounded repository research with file-and-line evidence"
    ),
    "mode": "subagent",
    "hidden": True,
    "temperature": 0.0,
    "permission": {
        "*": "deny",
        "read": _READ_PERMISSION,
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "lsp": "allow",
    },
    "prompt": (
        "Complete only the bounded repository question delegated to you. "
        "Use targeted reads and searches, never modify files or delegate, "
        "and distinguish observed facts from inference. Return no more than "
        "500 words with concise findings and file-and-line evidence. Do not "
        "dump source files or propose unrelated work."
    ),
}
_INVESTIGATE_WEB_AGENT = {
    "description": (
        "Read-only bounded external research with source URLs and confidence"
    ),
    "mode": "subagent",
    "hidden": True,
    "temperature": 0.0,
    "permission": {
        "*": "deny",
        "webfetch": "allow",
        "websearch": "allow",
    },
    "prompt": (
        "Complete only the bounded external-research question delegated to "
        "you. Prefer primary or authoritative sources. When community "
        "reports are relevant, cite their URLs and label anecdotal claims. "
        "Never access local files, modify anything, run commands, or "
        "delegate. Return no more than 500 words with concise findings, "
        "source URLs, relevance, and confidence. Do not dump pages or "
        "propose unrelated work."
    ),
}
_AGENTS = {
    "investigate": _INVESTIGATE_AGENT,
    "investigate-local": _INVESTIGATE_LOCAL_AGENT,
    "investigate-web": _INVESTIGATE_WEB_AGENT,
}


@dataclass(frozen=True)
class OpenCodeLaunchPlan:
    command: Tuple[str, ...]
    default_provider: str
    default_model: str
    endpoint: str
    dwarfstar_endpoint: str
    config_content: str
    tui_config: Path


@dataclass(frozen=True)
class OpenCodeSandboxPaths:
    root: Path
    config: Path
    data: Path
    state: Path
    cache: Path


@dataclass(frozen=True)
class OpenCodeSandboxPlan:
    command: Tuple[str, ...]
    environment: Mapping[str, str]
    workdir: Path
    state_root: Path


def _output_limit(context: int) -> int:
    # OpenCode reserves the advertised output allowance before deciding when
    # to compact. These local agent models do not need more than 16K for one
    # tool turn, and a larger allowance would discard useful session context.
    return min(16384, max(4096, context // 4))


def render_config(
    catalog: Catalog,
    default_model: str,
    endpoint: str,
    dwarfstar_endpoint: str = "http://127.0.0.1:8000/v1",
    default_provider: str = PROVIDER_ID,
) -> bytes:
    models = {}
    for identifier, preset in catalog.llama_presets.items():
        if not is_agent_capable(preset):
            continue
        model = {
            "name": identifier,
            "limit": {
                "context": preset.default_context,
                "output": _output_limit(preset.default_context),
            },
        }
        if preset.reasoning_effort_budget:
            model["reasoning"] = True
            # OpenCode merges an explicitly selected variant over these model
            # options. Keep a useful fallback without overriding a choice the
            # user has already made for this model.
            model["options"] = {
                "reasoningEffort": _DEFAULT_REASONING_EFFORT,
            }
            model["variants"] = _REASONING_VARIANTS
        models[identifier] = model
    contents = {
        "$schema": _CONFIG_SCHEMA,
        "model": "{}/{}".format(default_provider, default_model),
        "default_agent": _DEFAULT_AGENT,
        "agent": _AGENTS,
        "permission": _PERMISSIONS,
        "provider": {
            PROVIDER_ID: {
                "npm": _PROVIDER_PACKAGE,
                "name": _PROVIDER_NAME,
                "options": {"baseURL": endpoint},
                "models": models,
            },
            DWARFSTAR_PROVIDER_ID: {
                "npm": _PROVIDER_PACKAGE,
                "name": _DWARFSTAR_PROVIDER_NAME,
                "options": {"baseURL": dwarfstar_endpoint},
                "models": {
                    DWARFSTAR_MODEL: {
                        "name": "DeepSeek V4 Flash 0731",
                        "limit": {
                            "context": DWARFSTAR_DEFAULT_CONTEXT,
                            "output": DWARFSTAR_DEFAULT_OUTPUT_TOKENS,
                        },
                        "reasoning": True,
                        "options": {"reasoningEffort": "high"},
                        "variants": _DWARFSTAR_VARIANTS,
                    }
                },
            },
        },
    }
    return (json.dumps(contents, indent=2) + "\n").encode("utf-8")


def _default_model(catalog: Catalog, data_dir: Path) -> Tuple[str, str]:
    installed = installed_agent_presets(catalog, data_dir)
    if RECOMMENDED_MODEL in installed:
        return PROVIDER_ID, RECOMMENDED_MODEL
    if installed:
        return PROVIDER_ID, installed[0]
    dwarfstar = catalog.bundle(
        "dwarfstar-deepseek-v4-flash-0731-q2-imatrix"
    )
    if all(
        content_status_ready(status)
        for status in inspect_bundle(catalog, dwarfstar, data_dir)
    ):
        return DWARFSTAR_PROVIDER_ID, DWARFSTAR_MODEL
    raise LauncherError(
        "no installed model is maintained for OpenCode"
        "\n  llama.cpp: ./rocmplete content install llama-cpp qwen3.6"
        "\n  DwarfStar: ./rocmplete content install dwarfstar "
        "flash-0731-q2-imatrix"
    )


def _real_opencode(environ: Mapping[str, str]) -> str:
    path_value = environ.get("PATH", os.defpath)
    try:
        wrapper = WRAPPER_PATH.resolve(strict=False)
    except OSError as error:
        raise LauncherError("cannot resolve OpenCode wrapper: {}".format(error))
    search = []
    for entry in path_value.split(os.pathsep):
        directory = Path(entry or os.curdir)
        try:
            candidate = (directory / "opencode").resolve(strict=False)
        except OSError:
            candidate = directory / "opencode"
        if candidate == wrapper:
            continue
        search.append(entry)
    executable = shutil.which("opencode", path=os.pathsep.join(search))
    if executable is None:
        raise LauncherError(
            "OpenCode executable not found outside ROCmplete's bin directory"
        )
    return executable


def _validate_tui_config(path: Path) -> None:
    try:
        status = path.stat()
    except OSError as error:
        raise LauncherError(
            "cannot inspect OpenCode TUI configuration {}: {}".format(
                path, error
            )
        )
    if not stat.S_ISREG(status.st_mode):
        raise LauncherError(
            "OpenCode TUI configuration is not a regular file: {}".format(
                path
            )
        )


def create_launch_plan(
    catalog: Catalog,
    data_dir: Path,
    port: int,
    arguments: Sequence[str],
    environ: Optional[Mapping[str, str]] = None,
    *,
    dwarfstar_port: int = 8000,
) -> OpenCodeLaunchPlan:
    env = os.environ if environ is None else environ
    _validate_tui_config(TUI_CONFIG_PATH)
    default_provider, default_model = _default_model(catalog, data_dir)
    endpoint = "http://127.0.0.1:{}/v1".format(port)
    dwarfstar_endpoint = "http://127.0.0.1:{}/v1".format(dwarfstar_port)
    forwarded = tuple(arguments)
    if forwarded[:1] == ("--",):
        forwarded = forwarded[1:]
    return OpenCodeLaunchPlan(
        command=(_real_opencode(env),) + forwarded,
        default_provider=default_provider,
        default_model=default_model,
        endpoint=endpoint,
        dwarfstar_endpoint=dwarfstar_endpoint,
        config_content=render_config(
            catalog,
            default_model,
            endpoint,
            dwarfstar_endpoint,
            default_provider,
        ).decode("utf-8"),
        tui_config=TUI_CONFIG_PATH,
    )


def launch_environment(
    plan: OpenCodeLaunchPlan,
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, str]:
    child = dict(os.environ if environ is None else environ)
    # An inherited explicit file would layer stale ROCmplete settings with the
    # freshly rendered runtime configuration.
    child.pop("OPENCODE_CONFIG", None)
    child["OPENCODE_CONFIG_CONTENT"] = plan.config_content
    child["OPENCODE_TUI_CONFIG"] = str(plan.tui_config)
    return child


def sandbox_paths(data_dir: Path) -> OpenCodeSandboxPaths:
    root = StorageLayout(data_dir).application("opencode") / "sandbox"
    return OpenCodeSandboxPaths(
        root=root,
        config=root / "config",
        data=root / "data",
        state=root / "state",
        cache=root / "cache",
    )


def prepare_sandbox_paths(
    paths: OpenCodeSandboxPaths, data_dir: Path
) -> None:
    deepest = paths.cache / ".boundary"
    validate_managed_parent(
        deepest,
        paths.root,
        data_dir,
        "OpenCode sandbox state",
    )
    shared_parent = paths.root.parent.parent
    for path in (
        shared_parent,
        paths.root.parent,
        paths.root,
        paths.config,
        paths.data,
        paths.state,
        paths.cache,
    ):
        try:
            status = path.lstat()
        except FileNotFoundError:
            try:
                path.mkdir(mode=0o700)
            except OSError as error:
                raise LauncherError(
                    "cannot create OpenCode sandbox directory {}: {}".format(
                        path, error
                    )
                )
            continue
        except OSError as error:
            raise LauncherError(
                "cannot inspect OpenCode sandbox directory {}: {}".format(
                    path, error
                )
            )
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
            raise LauncherError(
                "OpenCode sandbox path is not a real directory: {}".format(
                    path
                )
            )
        if path != shared_parent:
            try:
                path.chmod(0o700)
            except OSError as error:
                raise LauncherError(
                    "cannot secure OpenCode sandbox directory {}: {}".format(
                        path, error
                    )
                )


def _resolved_executable(name: str, environ: Mapping[str, str]) -> Path:
    executable = shutil.which(name, path=environ.get("PATH", os.defpath))
    if executable is None:
        if name == "bwrap":
            raise LauncherError(
                "bubblewrap executable 'bwrap' not found on PATH; install "
                "the bubblewrap package or rerun with --no-sandbox"
            )
        raise LauncherError(
            "{} executable not found on PATH".format(name)
        )
    try:
        resolved = Path(executable).resolve(strict=True)
        status = resolved.stat()
    except OSError as error:
        raise LauncherError(
            "cannot inspect {} executable {}: {}".format(
                name, executable, error
            )
        )
    if not stat.S_ISREG(status.st_mode) or not os.access(resolved, os.X_OK):
        raise LauncherError(
            "{} executable is not an executable regular file: {}".format(
                name, resolved
            )
        )
    return resolved


def _linuxbrew_prefix(executable: Path) -> Optional[Path]:
    for parent in executable.parents:
        if parent.name == ".linuxbrew":
            return parent
    return None


def _sandbox_path(executable: Path) -> str:
    entries = []
    prefix = _linuxbrew_prefix(executable)
    if prefix is not None:
        entries.extend((str(prefix / "bin"), str(prefix / "sbin")))
    entries.extend(
        (
            "/usr/local/sbin",
            "/usr/local/bin",
            "/usr/sbin",
            "/usr/bin",
            "/sbin",
            "/bin",
        )
    )
    return os.pathsep.join(entries)


def _git_identity(environ: Mapping[str, str]) -> Mapping[str, str]:
    result: Dict[str, str] = {}
    configured = (
        ("GIT_AUTHOR_NAME", "user.name"),
        ("GIT_AUTHOR_EMAIL", "user.email"),
    )
    git = shutil.which("git", path=environ.get("PATH", os.defpath))
    for variable, key in configured:
        value = environ.get(variable)
        if value is None and git is not None:
            try:
                probe = subprocess.run(
                    [git, "config", "--global", "--get", key],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                probe = None
            if probe is not None and probe.returncode == 0:
                value = probe.stdout.rstrip("\r\n")
        if value and "\x00" not in value and "\n" not in value:
            result[variable] = value
    if "GIT_AUTHOR_NAME" in result:
        result["GIT_COMMITTER_NAME"] = result["GIT_AUTHOR_NAME"]
    if "GIT_AUTHOR_EMAIL" in result:
        result["GIT_COMMITTER_EMAIL"] = result["GIT_AUTHOR_EMAIL"]
    return result


def _terminal_environment(environ: Mapping[str, str]) -> Mapping[str, str]:
    allowed = {
        "COLORTERM",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "NO_COLOR",
        "TERM",
        "TZ",
    }
    result = {
        key: value
        for key, value in environ.items()
        if key in allowed or key.startswith("LC_")
    }
    return result


def _directory_arguments(paths: Sequence[Path]) -> Tuple[str, ...]:
    directories = set()
    for path in paths:
        for parent in reversed(path.parents):
            if parent != Path("/"):
                directories.add(parent)
        directories.add(path)
    arguments = []
    for directory in sorted(
        directories, key=lambda item: (len(item.parts), str(item))
    ):
        if directory == Path("/usr") or Path("/usr") in directory.parents:
            continue
        if directory == Path("/etc") or Path("/etc") in directory.parents:
            continue
        arguments.extend(("--dir", str(directory)))
    return tuple(arguments)


def _home_alias_arguments(
    home: Path = Path("/home"),
    expected_target: Path = Path("/var/home"),
) -> Tuple[str, ...]:
    """Preserve Fedora's stable home alias without exposing either tree."""
    try:
        if not stat.S_ISLNK(home.lstat().st_mode):
            return ()
        target = os.readlink(str(home))
        if home.resolve(strict=True) != expected_target.resolve(strict=True):
            return ()
    except OSError:
        return ()
    return ("--symlink", target, str(home))


def create_sandbox_plan(
    plan: OpenCodeLaunchPlan,
    data_dir: Path,
    workdir: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> OpenCodeSandboxPlan:
    env = os.environ if environ is None else environ
    paths = sandbox_paths(data_dir)
    try:
        working = workdir.resolve(strict=True)
        status = working.stat()
    except OSError as error:
        raise LauncherError(
            "cannot inspect OpenCode working directory {}: {}".format(
                workdir, error
            )
        )
    if not stat.S_ISDIR(status.st_mode):
        raise LauncherError(
            "OpenCode working directory is not a directory: {}".format(
                working
            )
        )
    try:
        host_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
        host_home.relative_to(working)
        contains_home = True
    except ValueError:
        contains_home = False
    except OSError as error:
        raise LauncherError(
            "cannot inspect the host home directory: {}".format(error)
        )
    if contains_home:
        raise LauncherError(
            "refusing an OpenCode sandbox working directory that contains "
            "the host home: {}".format(working)
        )
    try:
        working.relative_to(paths.root)
        overlaps = True
    except ValueError:
        try:
            paths.root.relative_to(working)
            overlaps = True
        except ValueError:
            overlaps = False
    if overlaps:
        raise LauncherError(
            "OpenCode working directory overlaps its private sandbox state: "
            "{}".format(paths.root)
        )

    bwrap = _resolved_executable("bwrap", env)
    opencode = Path(plan.command[0]).resolve(strict=True)
    prefix = _linuxbrew_prefix(opencode)
    mount_paths = [
        SANDBOX_HOME,
        SANDBOX_HOME / ".config",
        SANDBOX_HOME / ".local",
        SANDBOX_HOME / ".local" / "share",
        SANDBOX_HOME / ".local" / "state",
        SANDBOX_HOME / ".cache",
        SANDBOX_RUNTIME,
        SANDBOX_TUI_CONFIG.parent,
        working.parent,
    ]
    if prefix is not None:
        mount_paths.append(prefix.parent)
    else:
        mount_paths.append(opencode.parent)

    arguments = [
        str(bwrap),
        "--unshare-all",
        "--share-net",
        "--die-with-parent",
        "--new-session",
        "--hostname",
        "rocmplete",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/run",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/etc",
        "/etc",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/sbin",
        "/sbin",
        "--symlink",
        "usr/lib",
        "/lib",
    ]
    if Path("/usr/lib64").exists():
        arguments.extend(("--symlink", "usr/lib64", "/lib64"))
    arguments.extend(_home_alias_arguments())
    arguments.extend(_directory_arguments(mount_paths))
    if prefix is not None:
        arguments.extend(("--ro-bind", str(prefix), str(prefix)))
    elif not (
        opencode == Path("/usr") or Path("/usr") in opencode.parents
    ):
        arguments.extend(
            ("--ro-bind", str(opencode), str(opencode))
        )
    arguments.extend(
        (
            "--ro-bind",
            str(plan.tui_config),
            str(SANDBOX_TUI_CONFIG),
            "--bind",
            str(working),
            str(working),
        )
    )
    state_mounts = (
        (paths.config, SANDBOX_HOME / ".config"),
        (paths.data, SANDBOX_HOME / ".local" / "share"),
        (paths.state, SANDBOX_HOME / ".local" / "state"),
        (paths.cache, SANDBOX_HOME / ".cache"),
    )
    for source, destination in state_mounts:
        arguments.extend(("--bind", str(source), str(destination)))

    child = {
        "HOME": str(SANDBOX_HOME),
        "PATH": _sandbox_path(opencode),
        "SHELL": "/bin/sh",
        "USER": pwd.getpwuid(os.getuid()).pw_name,
        "LOGNAME": pwd.getpwuid(os.getuid()).pw_name,
        "TMPDIR": "/tmp",
        "XDG_CONFIG_HOME": str(SANDBOX_HOME / ".config"),
        "XDG_DATA_HOME": str(SANDBOX_HOME / ".local" / "share"),
        "XDG_STATE_HOME": str(SANDBOX_HOME / ".local" / "state"),
        "XDG_CACHE_HOME": str(SANDBOX_HOME / ".cache"),
        "XDG_RUNTIME_DIR": str(SANDBOX_RUNTIME),
        "OPENCODE_CONFIG_CONTENT": plan.config_content,
        "OPENCODE_TUI_CONFIG": str(SANDBOX_TUI_CONFIG),
        "OPENCODE_DISABLE_AUTOUPDATE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
    }
    child.update(_terminal_environment(env))
    child.update(_git_identity(env))
    for key, value in child.items():
        arguments.extend(("--setenv", key, value))
    arguments.extend(("--chdir", str(working), "--"))
    arguments.extend((str(opencode), "--pure"))
    arguments.extend(plan.command[1:])
    return OpenCodeSandboxPlan(
        command=tuple(arguments),
        environment={"PATH": env.get("PATH", os.defpath)},
        workdir=working,
        state_root=paths.root,
    )
