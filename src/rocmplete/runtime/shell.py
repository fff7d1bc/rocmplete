"""Interactive shell command construction for managed application images."""

from pathlib import Path
from typing import List

from .. import podman
from ..layout import StorageLayout
from .common import read_only_shared_suffix


def shell_command(
    image: str,
    data_dir: Path,
    volume_suffix: str,
    application: str = "comfyui",
) -> List[str]:
    layout = StorageLayout(data_dir)
    read_only_suffix = read_only_shared_suffix(volume_suffix)
    volumes: List[str] = [
        "--volume",
        "{}:/data{}".format(layout.application(application), volume_suffix),
    ]
    if application == "comfyui":
        volumes.extend(
            [
                "--volume",
                "{}:/content/models{}".format(
                    layout.comfy_models, read_only_suffix
                ),
            ]
        )
    elif application == "llama-cpp":
        volumes.extend(
            [
                "--volume",
                "{}:/content/models{}".format(
                    layout.llama_models, read_only_suffix
                ),
            ]
        )
    elif application == "dwarfstar":
        volumes.extend(
            [
                "--volume",
                "{}:/content/models{}".format(
                    layout.dwarfstar_models, read_only_suffix
                ),
            ]
        )
    return [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        "--umask",
        podman.current_umask(),
        "-it",
        *podman.managed_container_arguments(
            application=application, role="shell"
        ),
        "--read-only",
        "--cap-drop",
        "all",
        "--security-opt",
        "no-new-privileges",
        "--ulimit",
        "core=0:0",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=8g",
        *volumes,
        "--entrypoint",
        "/bin/bash",
        image,
    ]
