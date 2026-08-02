"""Runtime command construction for managed web applications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

from .. import podman
from ..config import APPLICATIONS
from ..layout import StorageLayout
from .common import (
    container_listen_address,
    gpu_device_arguments,
    publication_network_arguments,
    published_port,
    read_only_shared_suffix,
)


@dataclass(frozen=True)
class WebOptions:
    image: str
    profile: str
    listen: str
    port: int
    data_dir: Path
    render_nodes: Sequence[str] = field(default_factory=tuple)
    detach: bool = False
    unconfined: bool = False
    disable_bundled_extensions: bool = False
    comfy_args: Sequence[str] = field(default_factory=tuple)
    container_name: str = APPLICATIONS["comfyui"].container_name
    application: str = "comfyui"
    memory_policy: str = "balanced"
    kernel_policy: str = "default"
    environment: Sequence[str] = field(default_factory=tuple)
    publish: bool = True
    network_none: bool = False
    container_role: str = "application"


def web_command(options: WebOptions, volume_suffix: str) -> List[str]:
    layout = StorageLayout(options.data_dir)
    app_data = layout.application(options.application)
    read_only_suffix = read_only_shared_suffix(volume_suffix)
    content_volumes: List[str] = []
    if options.application == "comfyui":
        content_volumes.extend(
            [
                "--volume",
                "{}:/content/models{}".format(
                    layout.comfy_models, read_only_suffix
                ),
            ]
        )
    command = [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        "--umask",
        podman.current_umask(),
        "--name",
        options.container_name,
        *podman.managed_container_arguments(
            application=options.application,
            role=options.container_role,
        ),
    ]
    if options.publish:
        command.extend(publication_network_arguments(options.listen))
        command.extend(
            [
                "--publish",
                published_port(options.listen, options.port),
            ]
        )
    command.extend(
        [
            "--read-only",
            "--cap-drop",
            "all",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "2048",
            "--ulimit",
            "core=0:0",
            "--shm-size",
            "8g",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=8g",
            "--volume",
            "{}:/data{}".format(app_data, volume_suffix),
            *content_volumes,
            "--env",
            "ROCMLETE_PROFILE={}".format(options.profile),
            "--env",
            "ROCMLETE_LISTEN={}".format(
                container_listen_address(options.listen)
            ),
            "--env",
            "ROCMLETE_HOST_LISTEN={}".format(options.listen),
            "--env",
            "ROCMLETE_PORT={}".format(options.port),
            "--env",
            "ROCMLETE_KERNEL_POLICY={}".format(options.kernel_policy),
            "--env",
            "ROCMLETE_DISABLE_BUNDLED_EXTENSIONS={}".format(
                "1" if options.disable_bundled_extensions else "0"
            ),
        ]
    )
    if options.application == "comfyui":
        command.extend(
            [
                "--env",
                "ROCMLETE_MEMORY_POLICY={}".format(options.memory_policy),
            ]
        )
    if options.network_none:
        command.extend(["--network", "none"])
    for value in options.environment:
        command.extend(["--env", value])
    if options.profile != "cpu":
        command.extend(gpu_device_arguments(options.render_nodes))
    if options.unconfined:
        command.extend(["--security-opt", "seccomp=unconfined"])
    if options.detach:
        command.append("--detach")
    if options.kernel_policy == "experimental":
        command.extend(
            [
                "--env",
                "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1",
                "--env",
                "TORCH_BLAS_PREFER_HIPBLASLT=1",
            ]
        )
    command.append(options.image)
    command.extend(options.comfy_args)
    return command
