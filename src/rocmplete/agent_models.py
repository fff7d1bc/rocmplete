"""Managed llama.cpp presets suitable for tool-using agent harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Tuple

from .bundles import content_status_ready, inspect_bundle
from .catalog import Catalog, LlamaPreset
from .content_verification import VerificationStore
from .errors import LauncherError


PROVIDER_ID = "rocmplete"
DWARFSTAR_PROVIDER_ID = "dwarfstar"
DWARFSTAR_MODEL = "deepseek-v4-flash-0731-q2-imatrix"
RECOMMENDED_MODEL = "qwen3.8-27b-mtp-ud-q8-k-xl"
# Agent sampling is caller policy, not model runtime policy, so it stays out of
# the catalog presets. Values use llama.cpp's Chat Completions field names;
# notably, upstream repetition_penalty maps to repeat_penalty here.
_QWEN36_PRECISE_CODING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
_QWEN38_THINKING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
_MUSE_GLIMMER_CODING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repeat_penalty": 1.0,
}
_AGENT_SAMPLING_PARAMETERS = {
    "kat-coder-v2.5-dev-q8-0": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
    },
    "qwen3.6-27b-q8-0": _QWEN36_PRECISE_CODING,
    "qwen3.6-27b-mtp-q8-0": _QWEN36_PRECISE_CODING,
    "qwen3.8-27b-ud-q8-k-xl": _QWEN38_THINKING,
    "qwen3.8-27b-mtp-ud-q8-k-xl": _QWEN38_THINKING,
    "gemma4-31b-it-q8-0-mtp": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 64,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
    },
    "muse-glimmer-30b-kquant-dynamic-q4-k-xl": _MUSE_GLIMMER_CODING,
    "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash": _MUSE_GLIMMER_CODING,
    "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash-256k": (
        _MUSE_GLIMMER_CODING
    ),
}


def agent_output_limit(context: int) -> int:
    """Return the maintained per-turn output ceiling for agent clients."""

    # Clients reserve this advertised allowance before deciding when to
    # compact. These local models do not need more than 16K for one tool turn,
    # and a larger allowance would discard useful session context.
    return min(16384, max(4096, context // 4))


def is_agent_capable(preset: LlamaPreset) -> bool:
    """Return whether a preset has passed the managed agent contract."""

    # Do not infer tool-use suitability from model size or Jinja alone. This
    # claim records a reviewed Chat Completions function-tool contract.
    return preset.agent_tools


def agent_sampling_parameters(identifier: str) -> Mapping[str, object]:
    """Return reviewed llama.cpp request defaults for a coding-agent model."""

    try:
        return dict(_AGENT_SAMPLING_PARAMETERS[identifier])
    except KeyError as error:
        raise LauncherError(
            "llama.cpp agent preset {} has no reviewed sampling policy".format(
                identifier
            )
        ) from error


def reasoning_client_levels(preset: LlamaPreset) -> Tuple[str, ...]:
    """Return the truthful selector surface used by generic agent clients."""

    if not preset.reasoning_control:
        return ("off",)
    levels = ("high",) if preset.reasoning_control == "toggle" else (
        preset.reasoning_levels
    )
    if preset.reasoning_off:
        return ("off", *levels)
    return tuple(levels)


def reasoning_client_default(preset: LlamaPreset) -> str:
    """Map a model-native default onto the shared client vocabulary."""

    if preset.reasoning_control == "toggle":
        return "high"
    return preset.reasoning_default


def reasoning_native_value(preset: LlamaPreset, level: str) -> str:
    """Validate a client selector and return the model-native control value."""

    supported = reasoning_client_levels(preset)
    if level in supported:
        if preset.reasoning_control == "toggle" and level == "high":
            return "on"
        return level
    raise LauncherError(
        "llama.cpp preset {} supports --thinking {}".format(
            preset.identifier, " or ".join(supported)
        )
    )


def installed_presets(catalog: Catalog, data_dir: Path) -> Tuple[str, ...]:
    ready = []
    verification_store = VerificationStore.load(data_dir)
    for identifier, preset in catalog.llama_presets.items():
        statuses = inspect_bundle(
            catalog,
            catalog.bundle(preset.bundle),
            data_dir,
            verification_store,
        )
        if statuses and all(content_status_ready(item) for item in statuses):
            ready.append(identifier)
    return tuple(ready)


def installed_agent_presets(
    catalog: Catalog, data_dir: Path
) -> Tuple[str, ...]:
    ready = set(installed_presets(catalog, data_dir))
    return tuple(
        identifier
        for identifier, preset in catalog.llama_presets.items()
        if identifier in ready and is_agent_capable(preset)
    )


def recommended_agent_model(catalog: Catalog) -> Tuple[str, str, str]:
    """Return the shared recommendation with its model-native default."""

    preset = catalog.llama_preset(RECOMMENDED_MODEL)
    return (
        PROVIDER_ID,
        RECOMMENDED_MODEL,
        reasoning_client_default(preset),
    )


def default_agent_model(
    catalog: Catalog,
    data_dir: Path,
    client_name: str,
) -> Tuple[str, str, str]:
    """Select the common managed default for a coding-agent client."""

    recommended = recommended_agent_model(catalog)
    installed = installed_agent_presets(catalog, data_dir)
    if RECOMMENDED_MODEL in installed:
        return recommended
    if installed:
        preset = catalog.llama_preset(installed[0])
        thinking = reasoning_client_default(preset)
        return PROVIDER_ID, installed[0], thinking
    dwarfstar = catalog.bundle(
        "dwarfstar-deepseek-v4-flash-0731-q2-imatrix"
    )
    if all(
        content_status_ready(status)
        for status in inspect_bundle(catalog, dwarfstar, data_dir)
    ):
        return DWARFSTAR_PROVIDER_ID, DWARFSTAR_MODEL, "high"
    raise LauncherError(
        "no installed model is maintained for {}".format(client_name)
        + "\n  llama.cpp: ./rocmplete content install llama-cpp qwen3.8"
        + "\n  DwarfStar: ./rocmplete content install dwarfstar "
        "flash-0731-q2-imatrix"
    )
