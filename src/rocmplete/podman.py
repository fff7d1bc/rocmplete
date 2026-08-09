"""Minimal rootless Podman adapter."""

from __future__ import annotations

import os
import signal
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Callable, List, Optional

from .errors import LauncherError


MANAGED_CONTAINER_LABEL = "io.github.fff7d1bc.rocmplete.managed"
MANAGED_APPLICATION_LABEL = "io.github.fff7d1bc.rocmplete.application"
MANAGED_ROLE_LABEL = "io.github.fff7d1bc.rocmplete.role"
SELINUX_CONTAINER_DEVICE_COMMAND = (
    "sudo setsebool -P container_use_devices 1"
)


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def require_rootless() -> None:
    if not command_exists("podman"):
        raise LauncherError("podman is not installed")
    result = subprocess.run(
        ["podman", "info", "--format", "{{.Host.Security.Rootless}}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(
            "cannot query Podman; verify the rootless Podman service "
            "and runtime directory"
        )
    if result.stdout.strip() != "true":
        raise LauncherError("rootful Podman is not supported by this launcher")


def selinux_container_device_access() -> Optional[bool]:
    """Report an enforcing SELinux host's container-device policy."""
    getenforce = shutil.which("getenforce")
    getsebool = shutil.which("getsebool")
    if getenforce is None or getsebool is None:
        return None
    enforcing = subprocess.run(
        [getenforce],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if enforcing.returncode != 0 or enforcing.stdout.strip() != "Enforcing":
        return None
    device_policy = subprocess.run(
        [getsebool, "container_use_devices"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if device_policy.returncode != 0:
        return None
    value = device_policy.stdout.rsplit(None, 1)[-1:]
    if value == ["on"]:
        return True
    if value == ["off"]:
        return False
    return None


def require_container_device_access() -> None:
    if selinux_container_device_access() is False:
        raise LauncherError(
            "SELinux blocks GPU memory mapping for containers because "
            "container_use_devices is off; run {!r} and retry".format(
                SELINUX_CONTAINER_DEVICE_COMMAND
            )
        )


def image_exists(image: str) -> bool:
    if not command_exists("podman"):
        raise LauncherError("podman is not installed")
    return (
        subprocess.run(
            ["podman", "image", "exists", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def image_id(image: str) -> str:
    value = capture(
        ["podman", "image", "inspect", "--format", "{{.Id}}", image],
        "cannot inspect image {}".format(image),
    )
    if len(value) == 64:
        value = "sha256:{}".format(value)
    if (
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise LauncherError(
            "Podman returned an invalid image ID for {}".format(image)
        )
    return value


def current_umask() -> str:
    """Return the launcher's current file-creation mask as four octal digits."""
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("Umask:"):
                    return "{:04o}".format(int(line.split()[1], 8))
    except (OSError, ValueError, IndexError):
        pass
    mask = os.umask(0)
    os.umask(mask)
    return "{:04o}".format(mask)


def container_exists(name: str) -> bool:
    if not command_exists("podman"):
        raise LauncherError("podman is not installed")
    return (
        subprocess.run(
            ["podman", "container", "exists", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def managed_container_arguments(
    *, application: str = "", role: str
) -> List[str]:
    """Label a container so scoped cleanup can discover owned transients."""
    arguments = [
        "--label",
        "{}=true".format(MANAGED_CONTAINER_LABEL),
        "--label",
        "{}={}".format(MANAGED_ROLE_LABEL, role),
    ]
    if application:
        arguments.extend(
            [
                "--label",
                "{}={}".format(MANAGED_APPLICATION_LABEL, application),
            ]
        )
    return arguments


def managed_container_names(application: str = "") -> List[str]:
    """Return containers explicitly labelled as owned by ROCmplete."""
    command = [
        "podman",
        "ps",
        "--all",
        "--filter",
        "label={}={}".format(MANAGED_CONTAINER_LABEL, "true"),
    ]
    if application:
        command.extend(
            [
                "--filter",
                "label={}={}".format(
                    MANAGED_APPLICATION_LABEL, application
                ),
            ]
        )
    command.extend(["--format", "{{.Names}}"])
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LauncherError(
            "cannot inspect managed containers: {}".format(error)
        )
    if result.returncode != 0:
        raise LauncherError(
            "cannot inspect managed containers: {}".format(
                result.stderr.strip()
                or "podman ps exited with status {}".format(
                    result.returncode
                )
            )
        )
    return sorted(set(result.stdout.splitlines()))


def capture(command: List[str], error: str) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        if detail:
            raise LauncherError("{}: {}".format(error, detail))
        raise LauncherError(error)
    return result.stdout.strip()


def capture_bytes(command: List[str], error: str) -> bytes:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        if detail:
            raise LauncherError("{}: {}".format(error, detail))
        raise LauncherError(error)
    return result.stdout


def capture_stdout(command: List[str], error: str) -> str:
    """Capture structured stdout while leaving progress stderr attached."""
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise LauncherError(
            "{} (exit status {})".format(error, result.returncode)
        )
    return result.stdout


def run(command: List[str]) -> int:
    return subprocess.run(command, check=False).returncode


def run_quiet_stdout(command: List[str]) -> int:
    """Run a command while reserving stdout for ROCmplete's own summary."""
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        check=False,
    ).returncode


def run_managed_foreground(
    command: List[str], container_name: str, error: str
) -> int:
    """Run one attached managed container and report nonzero child status."""
    if command[:2] != ["podman", "run"]:
        raise LauncherError(
            "managed foreground execution requires a podman run command"
        )
    if "--detach" in command:
        raise LauncherError(
            "managed foreground execution does not accept --detach"
        )
    try:
        name_index = command.index("--name")
        command_name = command[name_index + 1]
    except (ValueError, IndexError):
        raise LauncherError(
            "managed foreground podman command requires a container name"
        )
    if command_name != container_name:
        raise LauncherError(
            "managed foreground container name does not match command"
        )

    # Keep the client in the terminal process group so interactive containers
    # retain normal stdin and signal behavior. The launcher owns the cleanup
    # race when Ctrl-C reaches both it and the Podman client.
    process = subprocess.Popen(command)
    try:
        status = process.wait()
    except BaseException as interruption:
        with _ignore_terminal_interrupts():
            _request_container_removal(container_name, stop_timeout=1)
            _reap_process(process)
            cleanup_error = _force_remove_container(
                container_name, stop_timeout=1
            )
        if cleanup_error:
            raise LauncherError(
                "cannot remove interrupted container {!r}: {}".format(
                    container_name, cleanup_error
                )
            ) from interruption
        raise
    if status != 0:
        raise LauncherError(
            "{} (exit status {})".format(error, status)
        )
    return 0


def _monitored_container_command(
    command: List[str], container_name: str
) -> List[str]:
    if command[:2] != ["podman", "run"]:
        raise LauncherError(
            "monitored execution requires a podman run command"
        )
    if "--name" in command:
        raise LauncherError(
            "monitored podman command already has a container name"
        )
    return (
        command[:2]
        + [
            "--name",
            container_name,
            *managed_container_arguments(role="download"),
        ]
        + command[2:]
    )


_CONTAINER_REMOVAL_WAIT = 15.0
_CONTAINER_REMOVAL_POLL_INTERVAL = 0.25


def _request_container_removal(
    container_name: str, *, stop_timeout: int
) -> str:
    try:
        result = subprocess.run(
            [
                "podman",
                "rm",
                "--force",
                "--time",
                str(stop_timeout),
                "--ignore",
                container_name,
            ],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)
    return "" if result.returncode == 0 else (
        result.stderr.strip()
        or "podman rm exited with status {}".format(result.returncode)
    )


def _wait_for_container_removal(container_name: str) -> str:
    deadline = time.monotonic() + _CONTAINER_REMOVAL_WAIT
    last_verification_error = ""
    while True:
        try:
            remaining = subprocess.run(
                ["podman", "container", "exists", container_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            last_verification_error = str(error)
        else:
            if remaining.returncode == 1:
                return ""
            if remaining.returncode == 0:
                last_verification_error = ""
            else:
                last_verification_error = "exit status {}".format(
                    remaining.returncode
                )
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            if last_verification_error:
                return "cannot verify container removal: {}".format(
                    last_verification_error
                )
            return "container still exists {} seconds after forced removal".format(
                int(_CONTAINER_REMOVAL_WAIT)
            )
        time.sleep(
            min(_CONTAINER_REMOVAL_POLL_INTERVAL, remaining_time)
        )


def _force_remove_container(
    container_name: str, *, stop_timeout: int
) -> str:
    # A nonzero rm status can mean Podman has already moved the container into
    # Stopping. Container absence, rather than that transient client status,
    # is the cleanup contract.
    removal_error = _request_container_removal(
        container_name, stop_timeout=stop_timeout
    )
    wait_error = _wait_for_container_removal(container_name)
    if not wait_error:
        return ""
    if removal_error:
        return "{}; {}".format(removal_error, wait_error)
    return wait_error


def remove_container(container_name: str, *, stop_timeout: int = 0) -> None:
    """Remove one exact managed container with a forced bounded fallback."""
    if stop_timeout < 0:
        raise LauncherError("container stop timeout must not be negative")
    with _ignore_terminal_interrupts():
        cleanup_error = _force_remove_container(
            container_name, stop_timeout=stop_timeout
        )
    if cleanup_error:
        raise LauncherError(
            "cannot remove container {!r}: {}".format(
                container_name, cleanup_error
            )
        )


@contextmanager
def _ignore_terminal_interrupts():
    """Let critical child-container cleanup survive repeated Ctrl-C."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _reap_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        process.wait()


def _managed_download_containers() -> List[str]:
    try:
        result = subprocess.run(
            ["podman", "ps", "--all", "--format", "{{.Names}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LauncherError(
            "cannot inspect existing downloader containers: {}".format(error)
        )
    if result.returncode != 0:
        raise LauncherError(
            "cannot inspect existing downloader containers: {}".format(
                result.stderr.strip()
                or "podman ps exited with status {}".format(
                    result.returncode
                )
            )
        )
    return sorted(
        name
        for name in result.stdout.splitlines()
        if name.startswith("rocmplete-download-")
    )


def managed_download_container_names() -> List[str]:
    """Return legacy and current generated downloader container names."""
    return _managed_download_containers()


def run_with_progress(
    command: List[str], update: Callable[[], None], interval: float = 1.0
) -> int:
    """Run an attached command while periodically reporting host-side state."""
    existing = _managed_download_containers()
    if existing:
        name = existing[0]
        raise LauncherError(
            "another ROCmplete downloader container exists: {}; wait for it "
            "to finish, or if it is left from an interruption remove it with "
            "'podman rm --force {}'".format(name, name)
        )
    container_name = "rocmplete-download-{}".format(uuid.uuid4().hex[:12])
    process = subprocess.Popen(
        _monitored_container_command(command, container_name),
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        update()
        while True:
            try:
                return process.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                update()
    except BaseException as error:
        # The client is isolated from terminal signals so the launcher owns
        # cleanup. Ignore repeated Ctrl-C only across this bounded critical
        # section, and remove before and after reaping to cover interruption
        # while Podman is still creating the named container.
        with _ignore_terminal_interrupts():
            if isinstance(error, KeyboardInterrupt):
                print(
                    "\nStopping downloader; partial data will be kept for "
                    "resume.",
                    file=sys.stderr,
                    flush=True,
                )
            _request_container_removal(container_name, stop_timeout=0)
            _reap_process(process)
            cleanup_error = _force_remove_container(
                container_name, stop_timeout=0
            )
        if cleanup_error:
            raise LauncherError(
                "cannot remove interrupted downloader container {!r}: "
                "{}".format(container_name, cleanup_error)
            ) from error
        raise


def replace_process(command: List[str]) -> None:
    os.execvp(command[0], command)


def selinux_volume_suffix() -> str:
    getenforce = shutil.which("getenforce")
    if getenforce is None:
        return ":rw"
    result = subprocess.run(
        [getenforce],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return ":rw,Z" if result.stdout.strip() == "Enforcing" else ":rw"


def shared_selinux_volume_suffix() -> str:
    """Writable suffix for persistent storage reused by many containers."""
    return selinux_volume_suffix().replace(",Z", ",z")
