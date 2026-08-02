#!/usr/bin/env python3
"""Inspect Civitai model metadata without exposing authentication tokens."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence

API_ROOT = "https://civitai.com/api/v1"


def _request_json(
    endpoint: str, token: Optional[str] = None
) -> Mapping[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "ROCmplete-catalog-probe/1",
    }
    if token:
        headers["Authorization"] = "Bearer {}".format(token)
    request = urllib.request.Request(
        "{}/{}".format(API_ROOT, endpoint), headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.load(response)
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Civitai API request failed for {}: {}".format(endpoint, error)
        )
    if not isinstance(value, dict):
        raise RuntimeError(
            "Civitai API returned a non-object for {}".format(endpoint)
        )
    return value


def _file_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "id",
            "name",
            "type",
            "sizeKB",
            "primary",
            "hashes",
            "metadata",
            "downloadUrl",
        )
    }


def _image_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "id",
            "url",
            "type",
            "width",
            "height",
            "nsfwLevel",
            "meta",
        )
    }


def _version_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    files = value.get("files")
    images = value.get("images")
    return {
        "id": value.get("id"),
        "modelId": value.get("modelId"),
        "name": value.get("name"),
        "description": value.get("description"),
        "baseModel": value.get("baseModel"),
        "baseModelType": value.get("baseModelType"),
        "publishedAt": value.get("publishedAt"),
        "availability": value.get("availability"),
        "files": [
            _file_summary(item)
            for item in files
            if isinstance(item, dict)
        ]
        if isinstance(files, list)
        else [],
        "images": [
            _image_summary(item)
            for item in images
            if isinstance(item, dict)
        ]
        if isinstance(images, list)
        else [],
    }


def _model_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    versions = value.get("modelVersions")
    creator = value.get("creator")
    return {
        "id": value.get("id"),
        "name": value.get("name"),
        "description": value.get("description"),
        "type": value.get("type"),
        "nsfw": value.get("nsfw"),
        "creator": (
            {"username": creator.get("username")}
            if isinstance(creator, dict)
            else None
        ),
        "permissions": {
            key: value.get(key)
            for key in (
                "allowNoCredit",
                "allowCommercialUse",
                "allowDerivatives",
                "allowDifferentLicense",
            )
        },
        "versions": [
            _version_summary(item)
            for item in versions
            if isinstance(item, dict)
        ]
        if isinstance(versions, list)
        else [],
    }


def probe(
    resource: str,
    identifier: int,
    token: Optional[str] = None,
    raw: bool = False,
) -> Mapping[str, Any]:
    endpoint = (
        "models/{}".format(identifier)
        if resource == "model"
        else "model-versions/{}".format(identifier)
    )
    value = _request_json(endpoint, token)
    if raw:
        return value
    return (
        _model_summary(value)
        if resource == "model"
        else _version_summary(value)
    )


def search(query: str, token: Optional[str] = None) -> Mapping[str, Any]:
    endpoint = "models?{}".format(
        urllib.parse.urlencode({"query": query, "limit": 20})
    )
    value = _request_json(endpoint, token)
    items = value.get("items")
    return {
        "items": [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type"),
                "creator": (
                    item.get("creator", {}).get("username")
                    if isinstance(item.get("creator"), dict)
                    else None
                ),
                "versions": [
                    {
                        "id": version.get("id"),
                        "name": version.get("name"),
                        "baseModel": version.get("baseModel"),
                        "files": [
                            _file_summary(file)
                            for file in version.get("files", [])
                            if isinstance(file, dict)
                        ],
                    }
                    for version in item.get("modelVersions", [])
                    if isinstance(version, dict)
                ],
            }
            for item in items
            if isinstance(item, dict)
        ]
        if isinstance(items, list)
        else []
    }


def probe_hash(digest: str, token: Optional[str] = None) -> Mapping[str, Any]:
    value = _request_json(
        "model-versions/by-hash/{}".format(
            urllib.parse.quote(digest, safe="")
        ),
        token,
    )
    return _version_summary(value)


def _positive_integer(value: str) -> int:
    identifier = int(value)
    if identifier <= 0:
        raise argparse.ArgumentTypeError("identifier must be positive")
    return identifier


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Civitai metadata for catalog research."
    )
    parser.add_argument(
        "resource", choices=("model", "version", "search", "hash")
    )
    parser.add_argument("identifier")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="print the complete API object instead of a stable summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    arguments = parse_arguments(argv or sys.argv[1:])
    try:
        token = os.environ.get("CIVITAI_TOKEN")
        if arguments.resource == "search":
            value = search(arguments.identifier, token=token)
        elif arguments.resource == "hash":
            value = probe_hash(arguments.identifier, token=token)
        else:
            value = probe(
                arguments.resource,
                _positive_integer(arguments.identifier),
                token=token,
                raw=arguments.raw,
            )
    except (RuntimeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
