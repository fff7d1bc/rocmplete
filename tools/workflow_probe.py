#!/usr/bin/env python3
"""Summarize declared packages and asset references in ComfyUI workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


ASSET_SUFFIXES = (
    ".bin",
    ".ckpt",
    ".cube",
    ".engine",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".vae",
)


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _asset_references(value: object) -> Set[str]:
    return {
        item
        for item in _strings(value)
        if item.lower().endswith(ASSET_SUFFIXES)
    }


def _ui_nodes(workflow: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    groups: List[Mapping[str, Any]] = [workflow]
    definitions = workflow.get("definitions")
    if isinstance(definitions, dict):
        subgraphs = definitions.get("subgraphs")
        if isinstance(subgraphs, list):
            groups.extend(
                item for item in subgraphs if isinstance(item, dict)
            )
    for group in groups:
        nodes = group.get("nodes")
        if not isinstance(nodes, list):
            raise RuntimeError("UI workflow has an invalid node collection")
        for node in nodes:
            if not isinstance(node, dict):
                raise RuntimeError("UI workflow contains an invalid node")
            yield node


def _package_summary(
    packages: Mapping[Tuple[str, str], Mapping[str, Set[str]]]
) -> List[Mapping[str, object]]:
    result = []
    for (source, identifier), metadata in sorted(packages.items()):
        result.append(
            {
                "identifier": identifier,
                "source": source,
                "versions": sorted(metadata["versions"]),
                "node_types": sorted(metadata["node_types"]),
            }
        )
    return result


def _probe_ui(workflow: Mapping[str, Any]) -> Mapping[str, object]:
    packages: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
    core_types: Set[str] = set()
    unattributed_types: Set[str] = set()
    assets: Set[str] = set()
    modes = {"active": 0, "bypassed": 0, "other": 0}
    nodes = list(_ui_nodes(workflow))
    for node in nodes:
        node_type = node.get("type")
        if not isinstance(node_type, str) or not node_type:
            raise RuntimeError("UI workflow node has no valid type")
        mode = node.get("mode", 0)
        if mode == 0:
            modes["active"] += 1
        elif mode == 4:
            modes["bypassed"] += 1
        else:
            modes["other"] += 1
        properties = node.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        cnr_id = properties.get("cnr_id")
        aux_id = properties.get("aux_id")
        version = properties.get("ver")
        if cnr_id == "comfy-core":
            core_types.add(node_type)
        elif isinstance(cnr_id, str) and cnr_id:
            key = ("registry", cnr_id)
            package = packages.setdefault(
                key, {"versions": set(), "node_types": set()}
            )
            package["node_types"].add(node_type)
            if isinstance(version, str) and version:
                package["versions"].add(version)
        elif isinstance(aux_id, str) and aux_id:
            key = ("repository", aux_id)
            package = packages.setdefault(
                key, {"versions": set(), "node_types": set()}
            )
            package["node_types"].add(node_type)
            if isinstance(version, str) and version:
                package["versions"].add(version)
        else:
            unattributed_types.add(node_type)
        assets.update(_asset_references(node.get("widgets_values", [])))
    return {
        "format": "ui",
        "node_count": len(nodes),
        "node_modes": modes,
        "declared_packages": _package_summary(packages),
        "core_node_types": sorted(core_types),
        "unattributed_node_types": sorted(unattributed_types),
        "asset_references": sorted(assets),
    }


def _probe_api(workflow: Mapping[str, Any]) -> Mapping[str, object]:
    node_types: Set[str] = set()
    assets: Set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            raise RuntimeError("API workflow contains an invalid node")
        node_type = node.get("class_type")
        inputs = node.get("inputs")
        if not isinstance(node_type, str) or not isinstance(inputs, dict):
            raise RuntimeError("API workflow node is missing type or inputs")
        node_types.add(node_type)
        assets.update(_asset_references(inputs))
    return {
        "format": "api",
        "node_count": len(workflow),
        "node_types": sorted(node_types),
        "asset_references": sorted(assets),
    }


def probe(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cannot read workflow {}: {}".format(path, error))
    if not isinstance(value, dict):
        raise RuntimeError("workflow root must be an object")
    result = (
        _probe_ui(value)
        if isinstance(value.get("nodes"), list)
        else _probe_api(value)
    )
    return {"path": str(path), **result}


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ComfyUI workflow dependencies."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    arguments = parse_arguments(argv or sys.argv[1:])
    try:
        value = {"workflows": [probe(path) for path in arguments.paths]}
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
