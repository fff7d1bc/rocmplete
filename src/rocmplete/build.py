"""Podman command construction for locally built application images."""

import os
from pathlib import Path
from typing import List, Mapping, Optional

from .errors import LauncherError


PIP_BUILD_CACHE_CONTAINER_PATH = Path("/var/cache/rocmplete/pip")


def build_cache_dir(environ: Mapping[str, str] = os.environ) -> Path:
    cache_home_value = environ.get("XDG_CACHE_HOME")
    if cache_home_value:
        cache_home = Path(cache_home_value)
        if not cache_home.is_absolute():
            raise LauncherError(
                "XDG_CACHE_HOME must be an absolute path: {}".format(
                    cache_home
                )
            )
    else:
        home_value = environ.get("HOME")
        if not home_value:
            raise LauncherError(
                "cannot locate the build cache: HOME is not set"
            )
        home = Path(home_value)
        if not home.is_absolute():
            raise LauncherError(
                "HOME must be an absolute path: {}".format(home)
            )
        cache_home = home / ".cache"
    try:
        return cache_home.resolve(strict=False) / "rocmplete" / "build"
    except OSError as error:
        raise LauncherError(
            "cannot resolve build cache location {}: {}".format(
                cache_home, error
            )
        )


def prepare_pip_build_cache(
    environ: Mapping[str, str] = os.environ,
) -> Path:
    root = build_cache_dir(environ)
    cache = root / "pip"
    for path in (root.parent, root, cache):
        if path.is_symlink():
            raise LauncherError(
                "refusing symlinked build cache path: {}".format(path)
            )
        if path.exists() and not path.is_dir():
            raise LauncherError(
                "build cache path is not a directory: {}".format(path)
            )
    try:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as error:
        raise LauncherError(
            "cannot create pip build cache {}: {}".format(cache, error)
        )
    return cache


def build_command(
    script_dir: Path,
    image: str,
    no_layer_cache: bool = False,
    target: str = "comfyui",
    base_image: Optional[str] = None,
    runtime_image: Optional[str] = None,
    pip_cache_dir: Optional[Path] = None,
    volume_suffix: str = ":rw",
) -> List[str]:
    command = [
        "podman",
        "build",
        "--tag",
        image,
        "--file",
        str(script_dir / "Containerfile"),
        "--target",
        target,
    ]
    if base_image is not None:
        command.extend(
            [
                "--build-arg",
                "ROCM_BASE_IMAGE={}".format(base_image),
                "--pull=never",
            ]
        )
    if runtime_image is not None:
        command.extend(
            [
                "--build-arg",
                "ROCM_RUNTIME_IMAGE={}".format(runtime_image),
                "--pull=never",
            ]
        )
    if pip_cache_dir is not None:
        if not pip_cache_dir.is_absolute():
            raise LauncherError(
                "pip build cache path must be absolute: {}".format(
                    pip_cache_dir
                )
            )
        if ":" in str(pip_cache_dir) or "\n" in str(pip_cache_dir):
            raise LauncherError(
                "pip build cache path cannot contain ':' or a newline: "
                "{}".format(pip_cache_dir)
            )
        if volume_suffix not in (":rw", ":rw,Z"):
            raise LauncherError(
                "unsupported build-cache volume suffix: {}".format(
                    volume_suffix
                )
            )
        command.extend(
            [
                "--build-arg",
                "PIP_NO_CACHE_DIR=",
                "--build-arg",
                "PIP_CACHE_DIR={}".format(
                    PIP_BUILD_CACHE_CONTAINER_PATH
                ),
                "--volume",
                "{}:{}{}".format(
                    pip_cache_dir,
                    PIP_BUILD_CACHE_CONTAINER_PATH,
                    volume_suffix,
                ),
            ]
        )
    if no_layer_cache:
        command.append("--no-cache")
    command.append(str(script_dir))
    return command
