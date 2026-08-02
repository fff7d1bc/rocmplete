#!/usr/bin/env python3
"""Inspect immutable Hugging Face model metadata without downloading weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Mapping, Optional, Sequence


API_ROOT = "https://huggingface.co/api"


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
            "Hugging Face API request failed for {}: {}".format(
                endpoint, error
            )
        )
    if not isinstance(value, dict):
        raise RuntimeError(
            "Hugging Face API returned a non-object for {}".format(endpoint)
        )
    return value


def _endpoint(repository: str, revision: Optional[str] = None) -> str:
    repository_path = urllib.parse.quote(repository, safe="/")
    if revision is None:
        return "models/{}".format(repository_path)
    return "models/{}/revision/{}?blobs=true".format(
        repository_path,
        urllib.parse.quote(revision, safe=""),
    )


def _license_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    card = value.get("cardData")
    if not isinstance(card, dict):
        card = {}
    return {
        "id": card.get("license"),
        "name": card.get("license_name"),
        "url": card.get("license_link"),
    }


def _file_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    lfs = value.get("lfs")
    if not isinstance(lfs, dict):
        lfs = {}
    return {
        "path": value.get("rfilename"),
        "size": lfs.get("size", value.get("size")),
        "sha256": lfs.get("sha256"),
        "lfs": bool(lfs),
        "blob_id": value.get("blobId"),
    }


def _repository_summary(
    value: Mapping[str, Any], include_files: bool
) -> Dict[str, Any]:
    result = {
        "repository": value.get("id") or value.get("modelId"),
        "revision": value.get("sha"),
        "private": value.get("private"),
        "gated": value.get("gated"),
        "disabled": value.get("disabled"),
        "license": _license_summary(value),
    }
    if include_files:
        siblings = value.get("siblings")
        result["files"] = sorted(
            (
                _file_summary(item)
                for item in siblings
                if isinstance(item, dict)
            ),
            key=lambda item: str(item.get("path")),
        ) if isinstance(siblings, list) else []
    return result


def probe_repository(
    repository: str, token: Optional[str] = None
) -> Mapping[str, Any]:
    return _repository_summary(
        _request_json(_endpoint(repository), token), include_files=False
    )


def probe_revision(
    repository: str,
    revision: str,
    token: Optional[str] = None,
) -> Mapping[str, Any]:
    return _repository_summary(
        _request_json(_endpoint(repository, revision), token),
        include_files=True,
    )


def probe_file(
    repository: str,
    revision: str,
    path: str,
    token: Optional[str] = None,
) -> Mapping[str, Any]:
    value = _request_json(_endpoint(repository, revision), token)
    siblings = value.get("siblings")
    matches = [
        item
        for item in siblings
        if isinstance(item, dict) and item.get("rfilename") == path
    ] if isinstance(siblings, list) else []
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one file {!r} at {}@{}; found {}".format(
                path, repository, revision, len(matches)
            )
        )
    result = _repository_summary(value, include_files=False)
    result["file"] = _file_summary(matches[0])
    return result


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face metadata for catalog research."
    )
    commands = parser.add_subparsers(dest="resource", required=True)

    repository = commands.add_parser(
        "repository", help="inspect the current repository head for discovery"
    )
    repository.add_argument("repository")

    revision = commands.add_parser(
        "revision", help="inspect every file at one immutable revision"
    )
    revision.add_argument("repository")
    revision.add_argument("revision")

    file_parser = commands.add_parser(
        "file", help="inspect one file at one immutable revision"
    )
    file_parser.add_argument("repository")
    file_parser.add_argument("revision")
    file_parser.add_argument("path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = ()) -> int:
    arguments = parse_arguments(argv or sys.argv[1:])
    token = os.environ.get("HF_TOKEN")
    try:
        if arguments.resource == "repository":
            value = probe_repository(arguments.repository, token)
        elif arguments.resource == "revision":
            value = probe_revision(
                arguments.repository, arguments.revision, token
            )
        else:
            value = probe_file(
                arguments.repository,
                arguments.revision,
                arguments.path,
                token,
            )
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
