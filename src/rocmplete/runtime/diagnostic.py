"""Constrained runtime probes used by host diagnostics and acceptance."""

from typing import List, Mapping, Sequence

from .. import podman
from ..errors import LauncherError
from ..hardware_profiles import SUPPORTED_ARCHITECTURES
from .common import gpu_device_arguments


GPU_DIAGNOSTIC_FIELDS = (
    "PyTorch",
    "ROCm/HIP",
    "Device",
    "Architecture",
    "GPU operation",
    "GPU devices",
)


def parse_gpu_diagnostic_output(output: str) -> Mapping[str, str]:
    fields = {}
    for line in output.splitlines():
        label, separator, value = line.partition(": ")
        if separator and label in GPU_DIAGNOSTIC_FIELDS:
            fields[label] = value
    missing = [label for label in GPU_DIAGNOSTIC_FIELDS if label not in fields]
    if missing:
        raise LauncherError(
            "GPU diagnostics returned incomplete output; missing {}".format(
                ", ".join(missing)
            )
        )
    failed = [
        label
        for label in ("GPU operation", "GPU devices")
        if fields[label] != "passed"
    ]
    if failed:
        raise LauncherError(
            "GPU diagnostics did not pass: {}".format(", ".join(failed))
        )
    return fields


def gpu_diagnostic_command(
    image: str, render_nodes: Sequence[str]
) -> List[str]:
    probe = (
        "import glob, pathlib, sys, torch; "
        "expected={!r}; supported={!r}; ".format(
            tuple(render_nodes), SUPPORTED_ARCHITECTURES
        )
        + "nodes=glob.glob('/dev/dri/renderD*'); "
        "assert pathlib.Path('/dev/kfd').exists(), '/dev/kfd missing'; "
        "assert sorted(nodes) == sorted(expected), nodes; "
        "count=torch.cuda.device_count(); "
        "assert count == len(expected), (count, expected); "
        "props=[torch.cuda.get_device_properties(i) for i in range(count)]; "
        "architectures=[getattr(p, 'gcnArchName', 'unknown').split(':', 1)[0] "
        "for p in props]; "
        "assert len(set(architectures)) == 1, architectures; "
        "all(architecture in supported for architecture in architectures) or "
        "sys.exit('unsupported GPU architecture(s) {}; this image contains {}'"
        ".format(', '.join(architectures), ', '.join(supported))); "
        "names=[torch.cuda.get_device_name(i) for i in range(count)]; "
        "actual=[torch.arange(1024, device='cuda:%d' % i).sum().item() "
        "for i in range(count)]; "
        "assert actual == [523776] * count, actual; "
        'print("PyTorch:", torch.__version__); '
        'print("ROCm/HIP:", torch.version.hip); '
        'print("Device:", "; ".join(names)); '
        'print("Architecture:", architectures[0]); '
        'print("GPU operation: passed"); '
        'print("GPU devices: passed")'
    )
    return [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        *podman.managed_container_arguments(role="diagnostic"),
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "all",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--ulimit",
        "core=0:0",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m",
        *gpu_device_arguments(render_nodes),
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "-c",
        probe,
    ]


def cpu_isolation_diagnostic_command(image: str) -> List[str]:
    probe = (
        "import glob, pathlib; "
        "assert not pathlib.Path('/dev/kfd').exists(), '/dev/kfd exposed'; "
        "nodes=glob.glob('/dev/dri/renderD*'); "
        "assert not nodes, nodes; "
        "print('CPU device isolation: passed')"
    )
    return [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        *podman.managed_container_arguments(role="diagnostic"),
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "all",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--ulimit",
        "core=0:0",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "-c",
        probe,
    ]
