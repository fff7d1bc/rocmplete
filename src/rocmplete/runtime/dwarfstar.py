"""Runtime command construction for the managed DwarfStar engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from .. import podman
from ..config import (
    APPLICATIONS,
    DWARFSTAR_DEFAULT_CONTEXT,
    DWARFSTAR_DEFAULT_OUTPUT_TOKENS,
)
from ..layout import StorageLayout
from .common import (
    container_listen_address,
    gpu_device_arguments,
    publication_network_arguments,
    published_port,
    read_only_shared_suffix,
)


@dataclass(frozen=True)
class DwarfStarOptions:
    image: str
    mode: str
    data_dir: Path
    model: Path
    render_nodes: Sequence[str] = field(default_factory=tuple)
    profile: str = "auto"
    listen: str = "127.0.0.1"
    port: int = 8000
    context: int = DWARFSTAR_DEFAULT_CONTEXT
    output_tokens: int = DWARFSTAR_DEFAULT_OUTPUT_TOKENS
    prompt: Optional[str] = None
    no_thinking: bool = False
    detach: bool = False
    interactive: bool = False
    unconfined: bool = False


def dwarfstar_command(
    options: DwarfStarOptions, volume_suffix: str
) -> List[str]:
    layout = StorageLayout(options.data_dir)
    read_only_suffix = read_only_shared_suffix(volume_suffix)
    model_root = options.model.parent
    container_model = "/content/models/{}".format(options.model.name)
    application = APPLICATIONS["dwarfstar"]
    command = [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        "--umask",
        podman.current_umask(),
        "--name",
        application.container_name,
        *podman.managed_container_arguments(
            application="dwarfstar", role="application"
        ),
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
        "/tmp:rw,nosuid,nodev,size=1g",
        "--volume",
        "{}:/data{}".format(
            layout.application("dwarfstar"), volume_suffix
        ),
        "--volume",
        "{}:/content/models{}".format(model_root, read_only_suffix),
        "--env",
        "ROCMLETE_PROFILE={}".format(options.profile),
        "--env",
        "ROCMLETE_DWARFSTAR_MODE={}".format(options.mode),
        "--env",
        "ROCMLETE_DWARFSTAR_MODEL={}".format(container_model),
        "--env",
        "ROCMLETE_DWARFSTAR_CONTEXT={}".format(options.context),
        "--env",
        "ROCMLETE_DWARFSTAR_OUTPUT_TOKENS={}".format(
            options.output_tokens
        ),
        "--env",
        "ROCMLETE_DWARFSTAR_NO_THINKING={}".format(
            "1" if options.no_thinking else "0"
        ),
        "--env",
        "ROCMLETE_LISTEN={}".format(
            container_listen_address(options.listen)
        ),
        "--env",
        "ROCMLETE_HOST_LISTEN={}".format(options.listen),
        "--env",
        "ROCMLETE_PORT={}".format(options.port),
    ]
    if options.prompt is not None:
        command.extend(
            ["--env", "ROCMLETE_DWARFSTAR_PROMPT={}".format(options.prompt)]
        )
    if options.mode == "server":
        command.extend(publication_network_arguments(options.listen))
        command.extend(
            ["--publish", published_port(options.listen, options.port)]
        )
    else:
        command.extend(["--network", "none"])
    command.extend(gpu_device_arguments(options.render_nodes))
    if options.unconfined:
        command.extend(["--security-opt", "seccomp=unconfined"])
    if options.detach:
        command.append("--detach")
    if options.interactive:
        command.extend(["--interactive", "--tty"])
    command.append(options.image)
    return command
