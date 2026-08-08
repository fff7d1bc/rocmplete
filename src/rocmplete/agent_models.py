"""Managed llama.cpp presets suitable for tool-using agent harnesses."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from .bundles import content_status_ready, inspect_bundle
from .catalog import Catalog, LlamaPreset
from .content_verification import VerificationStore


def is_agent_capable(preset: LlamaPreset) -> bool:
    """Return whether a preset has passed the managed agent contract."""

    # Do not infer tool-use suitability from model size or Jinja alone. This
    # claim records a reviewed Chat Completions function-tool contract.
    return preset.agent_tools


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
