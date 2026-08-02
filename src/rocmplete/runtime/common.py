"""Shared details for confined application runtime commands."""

from typing import List, Sequence


def container_listen_address(host_address: str) -> str:
    """Bind the service on the same address family as its host publication."""
    return "::" if ":" in host_address else "0.0.0.0"


def publication_network_arguments(host_address: str) -> List[str]:
    """Keep pasta's IPv4 wildcard publication from half-binding IPv6.

    Without ``-4``, pasta accepts host IPv6 connections for an unqualified
    IPv4 wildcard port and then resets them because the container service is
    listening on IPv4. A refused IPv6 connection can correctly fall back to
    IPv4; an accepted and reset connection cannot.
    """
    if host_address == "0.0.0.0":
        return ["--network", "pasta:-4"]
    return []


def published_port(address: str, port: int) -> str:
    host = "[{}]".format(address) if ":" in address else address
    return "{}:{}:{}/tcp".format(host, port, port)


def read_only_shared_suffix(volume_suffix: str) -> str:
    suffix = volume_suffix.replace(":rw", ":ro", 1)
    return suffix.replace(",Z", ",z")


def gpu_device_arguments(render_nodes: Sequence[str]) -> List[str]:
    arguments = ["--device", "/dev/kfd"]
    for render_node in render_nodes:
        arguments.extend(["--device", render_node])
    return arguments
