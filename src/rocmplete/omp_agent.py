"""Launch Oh My Pi against ROCmplete's managed local model servers."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

from .agent_models import (
    DWARFSTAR_MODEL,
    DWARFSTAR_PROVIDER_ID,
    PROVIDER_ID,
    agent_output_limit,
    agent_sampling_parameters,
    default_agent_model,
    is_agent_capable,
)
from .agent_sandbox import (
    AgentSandboxPaths as OmpSandboxPaths,
    AgentSandboxPlan as OmpSandboxPlan,
    SANDBOX_HOME,
    create_sandbox_plan as create_agent_sandbox_plan,
    find_real_executable,
    prepare_sandbox_paths as prepare_agent_sandbox_paths,
    sandbox_paths as agent_sandbox_paths,
)
from .catalog import Catalog
from .config import (
    DWARFSTAR_DEFAULT_CONTEXT,
    DWARFSTAR_DEFAULT_OUTPUT_TOKENS,
)
from .errors import LauncherError
from .layout import validate_managed_parent
from .project import PROJECT_ROOT


WRAPPER_PATH = PROJECT_ROOT / "bin" / "omp"
SANDBOX_AGENT_DIR = SANDBOX_HOME / ".local" / "share" / "omp" / "agent"
SANDBOX_OVERLAY_PATH = SANDBOX_AGENT_DIR / "rocmplete.json"
_COST = {
    "input": 0,
    "output": 0,
    "cacheRead": 0,
    "cacheWrite": 0,
}
_REASONING = {
    "mode": "effort",
    "efforts": ["low", "medium", "high"],
    "defaultLevel": "medium",
}
_DWARFSTAR_REASONING = {
    "mode": "effort",
    "efforts": ["high"],
    "defaultLevel": "high",
}
_INFORMATION_ARGUMENTS = frozenset(("--help", "-h", "--version", "-v"))
_PASSTHROUGH_COMMANDS = frozenset(("completions", "update"))
_MANAGEMENT_COMMANDS = frozenset(
    (
        "acp",
        "agents",
        "auth-broker",
        "auth-gateway",
        "bench",
        "browser-relay",
        "cleanse",
        "commit",
        "config",
        "dry-balance",
        "gallery",
        "gc",
        "grep",
        "grievances",
        "install",
        "join",
        "models",
        "plugin",
        "read",
        "say",
        "search",
        "setup",
        "share",
        "shell",
        "ssh",
        "stats",
        "tiny-models",
        "token",
        "ttsr",
        "usage",
        "worktree",
    )
)
_PROFILE_ARGUMENTS = ("--profile", "--alias")


@dataclass(frozen=True)
class OmpLaunchPlan:
    command: Tuple[str, ...]
    default_provider: Optional[str]
    default_model: Optional[str]
    default_thinking: Optional[str]
    endpoint: str
    dwarfstar_endpoint: str
    models_content: bytes
    overlay_content: bytes
    mode: str


def _compat(identifier: str) -> Mapping[str, object]:
    return {
        "supportsDeveloperRole": False,
        # OMP's ordinary sampling configuration is global. extraBody is
        # applied last and preserves the exact reviewed policy per model.
        "extraBody": dict(agent_sampling_parameters(identifier)),
    }


def render_models(
    catalog: Catalog,
    endpoint: str,
    dwarfstar_endpoint: str = "http://127.0.0.1:8000/v1",
) -> bytes:
    """Render OMP's custom local-provider catalog as JSON-compatible YAML."""

    models = []
    for identifier, preset in catalog.llama_presets.items():
        if not is_agent_capable(preset):
            continue
        model = {
            "id": identifier,
            "name": identifier,
            "reasoning": preset.reasoning_effort_budget,
            "input": ["text"],
            "supportsTools": True,
            "cost": _COST,
            "contextWindow": preset.default_context,
            "maxTokens": agent_output_limit(preset.default_context),
            "compat": _compat(identifier),
        }
        if preset.reasoning_effort_budget:
            model["thinking"] = _REASONING
            model["compat"]["supportsReasoningEffort"] = True
        models.append(model)

    dwarfstar_model = {
        "id": DWARFSTAR_MODEL,
        "name": "DeepSeek V4 Flash 0731",
        "reasoning": True,
        "thinking": _DWARFSTAR_REASONING,
        "input": ["text"],
        "supportsTools": True,
        "cost": _COST,
        "contextWindow": DWARFSTAR_DEFAULT_CONTEXT,
        "maxTokens": DWARFSTAR_DEFAULT_OUTPUT_TOKENS,
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": True,
        },
    }
    contents = {
        "providers": {
            PROVIDER_ID: {
                "name": "ROCmplete llama.cpp",
                "baseUrl": endpoint,
                "api": "openai-completions",
                "auth": "none",
                # This enables OMP's llama.cpp/Qwen chat and reasoning
                # compatibility without accepting arbitrary discovered IDs.
                "discovery": {"type": "llama.cpp"},
                "models": models,
            },
            DWARFSTAR_PROVIDER_ID: {
                "name": "ROCmplete DwarfStar",
                "baseUrl": dwarfstar_endpoint,
                "api": "openai-completions",
                "auth": "none",
                "models": [dwarfstar_model],
            },
        }
    }
    return (json.dumps(contents, indent=2) + "\n").encode("utf-8")


def render_overlay(
    catalog: Catalog, default_provider: str, default_model: str
) -> bytes:
    reference = "{}/{}".format(default_provider, default_model)
    enabled = [
        "{}/{}".format(PROVIDER_ID, identifier)
        for identifier, preset in catalog.llama_presets.items()
        if is_agent_capable(preset)
    ]
    enabled.append("{}/{}".format(DWARFSTAR_PROVIDER_ID, DWARFSTAR_MODEL))
    contents = {
        "modelRoles": {
            "default": reference,
            "smol": reference,
            "slow": reference,
            "plan": reference,
        },
        "enabledModels": enabled,
        # OMP otherwise adds its implicit localhost:8080 llama.cpp provider
        # beside the reviewed ROCmplete catalog.
        "disabledProviders": ["llama.cpp"],
        "tools": {"approvalMode": "yolo"},
        "startup": {"checkUpdate": False, "setupWizard": False},
        "marketplace": {"autoUpdate": "off"},
    }
    return (json.dumps(contents, indent=2) + "\n").encode("utf-8")


def _reject_profile_arguments(arguments: Sequence[str]) -> None:
    for argument in arguments:
        if any(
            argument == option or argument.startswith(option + "=")
            for option in _PROFILE_ARGUMENTS
        ):
            raise LauncherError(
                "OMP profiles cannot be combined with ROCmplete's private "
                "managed state; run the real omp executable directly to "
                "use --profile or --alias"
            )


def _configuration_default(
    catalog: Catalog, data_dir: Path
) -> Tuple[str, str, str]:
    # Model-independent commands still need a complete provider overlay.
    return default_agent_model(
        catalog, data_dir, "OMP", require_installed=False
    )


def create_launch_plan(
    catalog: Catalog,
    data_dir: Path,
    port: int,
    arguments: Sequence[str],
    environ: Optional[Mapping[str, str]] = None,
    *,
    dwarfstar_port: int = 8000,
) -> OmpLaunchPlan:
    env = os.environ if environ is None else environ
    endpoint = "http://127.0.0.1:{}/v1".format(port)
    dwarfstar_endpoint = "http://127.0.0.1:{}/v1".format(dwarfstar_port)
    forwarded = tuple(arguments)
    if forwarded[:1] == ("--",):
        forwarded = forwarded[1:]
    executable = find_real_executable("omp", WRAPPER_PATH, env, "OMP")
    _reject_profile_arguments(forwarded)

    passthrough = (
        bool(forwarded) and forwarded[0] in _INFORMATION_ARGUMENTS
    ) or (forwarded[:1] and forwarded[0] in _PASSTHROUGH_COMMANDS)
    management = forwarded[:1] and forwarded[0] in _MANAGEMENT_COMMANDS
    if passthrough or management:
        provider, model, thinking = _configuration_default(
            catalog, data_dir
        )
        defaults = (None, None, None)
        mode = "passthrough" if passthrough else "management"
        command = (executable, *forwarded)
    else:
        provider, model, thinking = default_agent_model(
            catalog, data_dir, "OMP"
        )
        defaults = (provider, model, thinking)
        mode = "session"
        command = (
            executable,
            "--model",
            "{}/{}".format(provider, model),
            "--thinking",
            thinking,
            *forwarded,
        )

    return OmpLaunchPlan(
        command=command,
        default_provider=defaults[0],
        default_model=defaults[1],
        default_thinking=defaults[2],
        endpoint=endpoint,
        dwarfstar_endpoint=dwarfstar_endpoint,
        models_content=render_models(
            catalog, endpoint, dwarfstar_endpoint
        ),
        overlay_content=render_overlay(catalog, provider, model),
        mode=mode,
    )


def sandbox_paths(data_dir: Path) -> OmpSandboxPaths:
    return agent_sandbox_paths(data_dir, "omp")


def _agent_dir(paths: OmpSandboxPaths) -> Path:
    return paths.data / "omp" / "agent"


def _secure_directory(path: Path, display_name: str) -> None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir(mode=0o700)
        except OSError as error:
            raise LauncherError(
                "cannot create {} directory {}: {}".format(
                    display_name, path, error
                )
            )
        return
    except OSError as error:
        raise LauncherError(
            "cannot inspect {} directory {}: {}".format(
                display_name, path, error
            )
        )
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise LauncherError(
            "{} path is not a real directory: {}".format(display_name, path)
        )
    try:
        path.chmod(0o700)
    except OSError as error:
        raise LauncherError(
            "cannot secure {} directory {}: {}".format(
                display_name, path, error
            )
        )


def _write_private_file(
    path: Path, content: bytes, root: Path, data_dir: Path, display_name: str
) -> None:
    validate_managed_parent(path, root, data_dir, display_name)
    try:
        status = path.lstat()
    except FileNotFoundError:
        status = None
    except OSError as error:
        raise LauncherError(
            "cannot inspect {} {}: {}".format(display_name, path, error)
        )
    if status is not None:
        if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise LauncherError(
                "{} is not a private regular file: {}".format(
                    display_name, path
                )
            )
        try:
            if path.read_bytes() == content:
                path.chmod(0o600)
                return
        except OSError as error:
            raise LauncherError(
                "cannot read {} {}: {}".format(display_name, path, error)
            )

    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".{}-".format(path.name), suffix=".tmp", dir=str(path.parent)
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise LauncherError(
            "cannot write {} {}: {}".format(display_name, path, error)
        )
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def prepare_state(
    plan: OmpLaunchPlan, paths: OmpSandboxPaths, data_dir: Path
) -> Path:
    """Prepare OMP's private state and refresh ROCmplete-owned config."""

    prepare_agent_sandbox_paths(paths, data_dir, "OMP")
    agent_dir = _agent_dir(paths)
    _secure_directory(agent_dir.parent, "OMP state")
    _secure_directory(agent_dir, "OMP state")
    _write_private_file(
        agent_dir / "models.yml",
        plan.models_content,
        paths.root,
        data_dir,
        "OMP model configuration",
    )
    _write_private_file(
        agent_dir / "rocmplete.json",
        plan.overlay_content,
        paths.root,
        data_dir,
        "OMP runtime overlay",
    )
    return agent_dir


def _runtime_environment(
    agent_dir: Path, overlay_path: Path
) -> Mapping[str, str]:
    return {
        "PI_CODING_AGENT_DIR": str(agent_dir),
        "PI_CONFIG_FILES": str(overlay_path),
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
    }


def launch_environment(
    paths: OmpSandboxPaths,
    environ: Optional[Mapping[str, str]] = None,
) -> Mapping[str, str]:
    child = dict(os.environ if environ is None else environ)
    child.pop("OMP_PROFILE", None)
    child.pop("PI_PROFILE", None)
    child.update(
        {
            "XDG_CONFIG_HOME": str(paths.config),
            "XDG_DATA_HOME": str(paths.data),
            "XDG_STATE_HOME": str(paths.state),
            "XDG_CACHE_HOME": str(paths.cache),
        }
    )
    agent_dir = _agent_dir(paths)
    child.update(
        _runtime_environment(agent_dir, agent_dir / "rocmplete.json")
    )
    return child


def create_sandbox_plan(
    plan: OmpLaunchPlan,
    data_dir: Path,
    workdir: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> OmpSandboxPlan:
    return create_agent_sandbox_plan(
        plan.command,
        data_dir,
        workdir,
        "omp",
        "OMP",
        _runtime_environment(SANDBOX_AGENT_DIR, SANDBOX_OVERLAY_PATH),
        environ,
    )
