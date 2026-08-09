"""Curated artifact and bundle installation."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from . import podman
from .catalog import Artifact, Bundle, Catalog
from .content_verification import VerificationStore
from .errors import LauncherError
from .layout import StorageLayout, validate_managed_parent
from .ui import finish_rewrite, rewrite_line, style, terminal_output


@dataclass(frozen=True)
class ArtifactStatus:
    artifact: Artifact
    path: Path
    state: str
    actual_size: int = 0
    integrity: str = "unverified"


ContentStatus = ArtifactStatus


@dataclass(frozen=True)
class MirrorMatch:
    source: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class _DownloadTarget:
    path: Path
    size: int
    sha256: str


_REDIRECTED_PROGRESS_PERCENT = 5
_REDIRECTED_PROGRESS_SECONDS = 60.0
_TERMINAL_PROGRESS_SECONDS = 1.0
_SHARED_CONTENT_TARGETS = frozenset(
    ("models", "llama-models", "dwarfstar-models")
)


def _content_lock_directory() -> Path:
    runtime_value = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_value:
        runtime = Path(runtime_value)
        create_runtime = False
    else:
        runtime = (
            Path(tempfile.gettempdir())
            / "rocmplete-runtime-{}".format(os.geteuid())
        )
        create_runtime = True
    try:
        if create_runtime:
            runtime.mkdir(mode=0o700, exist_ok=True)
        status = runtime.lstat()
    except OSError as error:
        raise LauncherError(
            "cannot prepare content lock directory {}: {}".format(
                runtime, error
            )
        )
    if (
        stat.S_ISLNK(status.st_mode)
        or not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.geteuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise LauncherError(
            "content lock runtime directory is not a private owned "
            "directory: {}".format(runtime)
        )
    locks = runtime / "rocmplete-locks"
    try:
        locks.mkdir(mode=0o700, exist_ok=True)
        lock_status = locks.lstat()
    except OSError as error:
        raise LauncherError(
            "cannot prepare content lock directory {}: {}".format(
                locks, error
            )
        )
    if (
        stat.S_ISLNK(lock_status.st_mode)
        or not stat.S_ISDIR(lock_status.st_mode)
        or lock_status.st_uid != os.geteuid()
        or stat.S_IMODE(lock_status.st_mode) & 0o077
    ):
        raise LauncherError(
            "content lock directory is not private and owned: {}".format(
                locks
            )
        )
    return locks


@contextmanager
def content_install_lock(data_dir: Path):
    """Serialize all staging and installation mutations for one data root."""
    try:
        resolved_data = data_dir.resolve(strict=True)
    except OSError as error:
        raise LauncherError(
            "cannot resolve content data directory {}: {}".format(
                data_dir, error
            )
        )
    identity = hashlib.sha256(
        str(resolved_data).encode(
            "utf-8", "surrogateescape"
        )
    ).hexdigest()
    path = _content_lock_directory() / (
        "content-{}.lock".format(identity)
    )
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except OSError as error:
        raise LauncherError(
            "cannot open content installation lock {}: {}".format(
                path, error
            )
        )
    try:
        try:
            status = os.fstat(descriptor)
            if (
                not stat.S_ISREG(status.st_mode)
                or status.st_uid != os.geteuid()
                or stat.S_IMODE(status.st_mode) & 0o077
            ):
                raise LauncherError(
                    "content lock is not a regular owned file: {}".format(
                        path
                    )
                )
            try:
                fcntl.flock(
                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError:
                raise LauncherError(
                    "another content installation is active for {}; "
                    "wait for it to finish before retrying".format(data_dir)
                )
        except OSError as error:
            raise LauncherError(
                "cannot acquire content installation lock {}: {}".format(
                    path, error
                )
            )
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def human_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(size)


def _compact_progress_item(item: str, available: int) -> str:
    if len(item) <= available:
        return item
    if available <= 4:
        return item[:available]
    prefix = max(1, (available - 3) // 3)
    suffix = available - prefix - 3
    return "{}...{}".format(item[:prefix], item[-suffix:])


def _apparent_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = tuple(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                else:
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def _nearest_existing_path(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_managed_destination(
    destination: Path,
    managed_root: Path,
    data_dir: Path,
    description: str,
) -> None:
    validate_managed_parent(
        destination, managed_root, data_dir, description
    )


def _staging_partition(destination: Path, data_dir: Path) -> Path:
    staging = StorageLayout(data_dir).staging
    try:
        relative = destination.relative_to(staging)
    except ValueError:
        raise LauncherError(
            "staging path escapes the managed staging root: {}".format(
                destination
            )
        )
    if not relative.parts or relative.parts[0] not in (
        "comfyui",
        "llama-cpp",
        "dwarfstar",
    ):
        raise LauncherError(
            "staging path has no managed application partition: {}".format(
                destination
            )
        )
    return staging / relative.parts[0]


def _validate_staging_destination(destination: Path, data_dir: Path) -> None:
    _validate_managed_destination(
        destination,
        _staging_partition(destination, data_dir),
        data_dir,
        "staging",
    )


def _prepare_staging_root(staging: Path, data_dir: Path) -> None:
    _validate_staging_destination(staging / ".rocmplete-path-check", data_dir)
    staging.mkdir(parents=True, exist_ok=True)
    _validate_staging_destination(staging / ".rocmplete-path-check", data_dir)


def _staged_file_matches(path: Path, expected_size: int) -> bool:
    """Return whether a reusable staging entry is one regular file of this size."""
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise LauncherError(
            "cannot inspect staging entry {}: {}".format(path, error)
        )
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LauncherError(
            "refusing unexpected staging entry: {}".format(path)
        )
    return status.st_size == expected_size


def _quarantine_staged_file(path: Path, actual_sha256: str) -> Path:
    """Preserve rejected staging bytes while making the expected path retryable."""
    quarantine = path.with_name(
        "{}.invalid-{}-{}".format(
            path.name, actual_sha256[:12], time.time_ns()
        )
    )
    try:
        os.replace(str(path), str(quarantine))
    except OSError as error:
        raise LauncherError(
            "cannot quarantine corrupt staging file {}: {}".format(path, error)
        )
    return quarantine


class LocalMirror:
    """Find exact catalog bytes in an existing local content directory."""

    def __init__(self, root: Path, move: bool = False) -> None:
        try:
            self.root = root.expanduser().resolve(strict=True)
        except OSError as error:
            raise LauncherError(
                "cannot access local mirror {}: {}".format(root, error)
            )
        if not self.root.is_dir():
            raise LauncherError(
                "local mirror is not a directory: {}".format(self.root)
            )
        self.move = move
        self._files_by_name = None  # type: Optional[dict]
        self._digests = {}  # type: dict
        self._reserved = set()  # type: set

    def validate_destination(self, data_dir: Path) -> None:
        destination = data_dir.expanduser().resolve(strict=False)
        if (
            destination == self.root
            or _path_is_within(destination, self.root)
            or _path_is_within(self.root, destination)
        ):
            raise LauncherError(
                "local mirror and active data directory must not overlap: "
                "{} and {}".format(self.root, destination)
            )

    def _index(self) -> dict:
        if self._files_by_name is not None:
            return self._files_by_name
        files = {}  # type: dict
        for directory, directories, names in os.walk(
            str(self.root), followlinks=False
        ):
            directories[:] = sorted(
                name
                for name in directories
                if not (Path(directory) / name).is_symlink()
            )
            for name in sorted(names):
                candidate = Path(directory) / name
                try:
                    if candidate.is_symlink():
                        continue
                    resolved = candidate.resolve(strict=True)
                    if (
                        not _path_is_within(resolved, self.root)
                        or not resolved.is_file()
                    ):
                        continue
                except OSError:
                    continue
                entries = files.setdefault(name, [])
                if resolved not in entries:
                    entries.append(resolved)
        self._files_by_name = files
        return files

    def find(
        self, names: Sequence[str], expected_size: int, expected_sha256: str
    ) -> Optional[MirrorMatch]:
        candidates = self._index()
        checked = set()
        for name in names:
            basename = Path(name).name
            for candidate in candidates.get(basename, ()):
                if candidate in checked or candidate in self._reserved:
                    continue
                checked.add(candidate)
                try:
                    stat = candidate.stat()
                except OSError:
                    continue
                if stat.st_size != expected_size:
                    continue
                signature = (
                    candidate,
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                )
                digest = self._digests.get(signature)
                if digest is None:
                    print(
                        "\n{} local mirror candidate {} ({})".format(
                            style("Checking", "heading"),
                            candidate,
                            human_size(expected_size),
                        ),
                        flush=True,
                    )
                    progress = _VerificationProgress(expected_size)
                    progress.update(candidate.name, 1, 1, 0)
                    try:
                        digest = sha256_file(
                            candidate,
                            lambda hashed: progress.update(
                                candidate.name, 1, 1, hashed
                            ),
                        )
                        progress.update(
                            candidate.name,
                            1,
                            1,
                            expected_size,
                            force=True,
                        )
                    except BaseException:
                        progress.finish()
                        raise
                    self._digests[signature] = digest
                if digest == expected_sha256:
                    if self.move:
                        self._reserved.add(candidate)
                    return MirrorMatch(candidate, expected_size, digest)
        return None

    def required_space(self, match: MirrorMatch, data_dir: Path) -> int:
        if not self.move:
            return match.size
        try:
            destination_device = _nearest_existing_path(data_dir).stat().st_dev
            source_device = match.source.stat().st_dev
        except OSError:
            return match.size
        return 0 if destination_device == source_device else match.size

    def materialize(
        self, match: MirrorMatch, destination: Path, data_dir: Path
    ) -> None:
        _validate_staging_destination(destination, data_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _validate_staging_destination(destination, data_dir)
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise LauncherError(
                "refusing unexpected staging entry: {}".format(destination)
            )
        if destination.is_file():
            destination.unlink()
        action = "Moving" if self.move else "Copying"
        print(
            "{} local mirror file: {}".format(
                style(action, "heading"), match.source
            )
        )
        try:
            if self.move:
                shutil.move(str(match.source), str(destination))
            else:
                shutil.copy2(str(match.source), str(destination))
        except OSError as error:
            raise LauncherError(
                "cannot {} local mirror file {}: {}".format(
                    "move" if self.move else "copy", match.source, error
                )
            )


class _RetryAwareDownloadSize:
    """Measure current payloads without counting abandoned HF retry files."""

    def __init__(
        self, staging: Path, targets: Sequence[_DownloadTarget]
    ) -> None:
        self.staging = staging
        self.targets = tuple(targets)
        self.active_partials = set()  # type: set
        self.initial_partials = self._partial_signatures()

    def _partial_paths(self, target: _DownloadTarget) -> Tuple[Path, ...]:
        try:
            relative = target.path.relative_to(self.staging)
        except ValueError:
            return ()
        cache = (
            self.staging
            / ".cache"
            / "huggingface"
            / "download"
            / relative.parent
        )
        marker = ".{}.".format(target.sha256)
        try:
            resolved_staging = self.staging.resolve(strict=True)
            resolved_cache = cache.resolve(strict=True)
        except OSError:
            return ()
        if not _path_is_within(resolved_cache, resolved_staging):
            return ()
        try:
            entries = tuple(os.scandir(cache))
        except OSError:
            return ()
        paths = []
        for entry in entries:
            if (
                marker not in entry.name
                or not entry.name.endswith(".incomplete")
            ):
                continue
            try:
                if entry.is_file(follow_symlinks=False):
                    paths.append(Path(entry.path))
            except OSError:
                continue
        return tuple(paths)

    def _partial_signatures(self) -> dict:
        signatures = {}
        for target in self.targets:
            for path in self._partial_paths(target):
                try:
                    status = path.stat()
                except OSError:
                    continue
                signatures[path] = (status.st_size, status.st_mtime_ns)
        return signatures

    def reusable_partial_size(self) -> int:
        """Return bytes from unambiguous exact-file partials HF can resume."""
        total = 0
        for target in self.targets:
            candidates = [
                signature[0]
                for path, signature in self.initial_partials.items()
                if path in self._partial_paths(target)
                and 0 <= signature[0] <= target.size
            ]
            # Multiple files for one target are ambiguous. Do not assume which
            # one the downloader will reuse when planning required space.
            if len(candidates) == 1:
                total += candidates[0]
        return total

    @staticmethod
    def _regular_size(path: Path) -> Optional[int]:
        try:
            status = path.lstat()
        except OSError:
            return None
        return status.st_size if stat.S_ISREG(status.st_mode) else None

    def __call__(self) -> int:
        current_partials = self._partial_signatures()
        self.active_partials.update(
            path
            for path, signature in current_partials.items()
            if self.initial_partials.get(path) != signature
        )
        total = 0
        for target in self.targets:
            payload_size = self._regular_size(target.path)
            if payload_size is not None:
                total += min(payload_size, target.size)
                continue
            partial_size = max(
                (
                    current_partials[path][0]
                    for path in self._partial_paths(target)
                    if path in self.active_partials
                    and path in current_partials
                ),
                default=0,
            )
            total += min(partial_size, target.size)
        return total


class _DownloadProgress:
    """Approximate progress based on the mounted staging tree."""

    def __init__(
        self,
        staging: Path,
        expected_size: int,
        measure: Optional[Callable[[], int]] = None,
        bounded: bool = False,
    ) -> None:
        self.staging = staging
        self.expected_size = expected_size
        self.measure = measure or (lambda: _apparent_size(self.staging))
        self.bounded = bounded
        self.last_percent = -_REDIRECTED_PROGRESS_PERCENT
        self.last_downloaded = -1
        self.last_report = 0.0
        self.line_width = 0
        self.active = False

    def update(self, force: bool = False) -> None:
        downloaded = min(self.measure(), self.expected_size)
        percent = (
            100.0
            if self.expected_size == 0
            else downloaded * 100.0 / self.expected_size
        )
        whole_percent = int(percent)
        now = time.monotonic()
        redirected_percent_due = (
            whole_percent
            >= self.last_percent + _REDIRECTED_PROGRESS_PERCENT
        )
        redirected_time_due = (
            now - self.last_report >= _REDIRECTED_PROGRESS_SECONDS
        )
        is_terminal = terminal_output()
        if force and not is_terminal and downloaded == self.last_downloaded:
            return
        if (
            force
            or is_terminal
            or self.last_percent < 0
            or redirected_percent_due
            or redirected_time_due
        ):
            if self.bounded:
                message = "  {} {} {}; {} {}".format(
                    style("Progress:", "info"),
                    style("staged", "label"),
                    human_size(downloaded),
                    style("limit", "label"),
                    human_size(self.expected_size),
                )
            else:
                message = "  {} {}  {} {}; {} {}".format(
                    style("Progress:", "info"),
                    style(
                        "{:.1f}%".format(percent),
                        "info",
                    ),
                    style("staged", "label"),
                    human_size(downloaded),
                    style("expected", "label"),
                    human_size(self.expected_size),
                )
            self.line_width = rewrite_line(
                message,
                previous_width=self.line_width,
                complete=force,
            )
            self.last_percent = whole_percent
            self.last_downloaded = downloaded
            self.last_report = now
            self.active = is_terminal and not force

    def finish(self) -> None:
        if self.active:
            finish_rewrite()
            self.active = False


class _VerificationProgress:
    """Exact byte progress for SHA-256 verification."""

    def __init__(self, expected_size: int) -> None:
        self.expected_size = expected_size
        self.last_percent = -_REDIRECTED_PROGRESS_PERCENT
        self.last_hashed = -1
        self.last_report = 0.0
        self.line_width = 0
        self.active = False

    def update(
        self,
        item: str,
        index: int,
        total: int,
        hashed: int,
        force: bool = False,
    ) -> None:
        hashed = min(hashed, self.expected_size)
        percent = (
            100.0
            if self.expected_size == 0
            else hashed * 100.0 / self.expected_size
        )
        whole_percent = int(percent)
        now = time.monotonic()
        is_terminal = terminal_output()
        terminal_time_due = (
            now - self.last_report >= _TERMINAL_PROGRESS_SECONDS
        )
        redirected_percent_due = (
            whole_percent
            >= self.last_percent + _REDIRECTED_PROGRESS_PERCENT
        )
        redirected_time_due = (
            now - self.last_report >= _REDIRECTED_PROGRESS_SECONDS
        )
        if force and not is_terminal and hashed == self.last_hashed:
            return
        if not (
            force
            or self.last_hashed < 0
            or (is_terminal and terminal_time_due)
            or (not is_terminal and redirected_percent_due)
            or (not is_terminal and redirected_time_due)
        ):
            return

        position = "[{}/{}] ".format(index, total) if total > 1 else ""
        percentage = "{:.1f}%".format(percent)
        hashed_size = human_size(hashed)
        expected_size = human_size(self.expected_size)
        if is_terminal:
            fixed_width = len(
                "  Verifying: {}  {}  hashed {} of {}".format(
                    position,
                    percentage,
                    hashed_size,
                    expected_size,
                )
            )
            item = _compact_progress_item(
                item,
                max(8, shutil.get_terminal_size().columns - fixed_width),
            )
        message = "  {} {}{}  {}  {} {} of {}".format(
            style("Verifying:", "info"),
            position,
            item,
            style(percentage, "info"),
            style("hashed", "label"),
            hashed_size,
            expected_size,
        )
        self.line_width = rewrite_line(
            message,
            previous_width=self.line_width,
            complete=force,
        )
        self.last_percent = whole_percent
        self.last_hashed = hashed
        self.last_report = now
        self.active = is_terminal and not force

    def finish(self) -> None:
        if self.active:
            finish_rewrite()
            self.active = False


def _artifact_root(data_dir: Path, artifact: Artifact) -> Path:
    layout = StorageLayout(data_dir)
    roots = {
        "models": layout.comfy_models,
        "llama-models": layout.llama_models,
        "dwarfstar-models": layout.dwarfstar_models,
        "workflows": layout.imported_workflows,
    }
    return roots[artifact.target]


def artifact_path(data_dir: Path, artifact: Artifact) -> Path:
    return _artifact_root(data_dir, artifact) / Path(artifact.destination)


def _prepare_runtime_label(artifact: Artifact, path: Path) -> None:
    if artifact.target in _SHARED_CONTENT_TARGETS:
        podman.prepare_shared_content_label(path)


def inspect_bundle(
    catalog: Catalog,
    bundle: Bundle,
    data_dir: Path,
    verification_store: Optional[VerificationStore] = None,
) -> Tuple[ContentStatus, ...]:
    store = verification_store or VerificationStore.load(data_dir)
    return inspect_artifacts(catalog.bundle_artifacts(bundle), data_dir, store)


def artifact_staging_root(data_dir: Path, artifact: Artifact) -> Path:
    application = {
        "llama-models": "llama-cpp",
        "dwarfstar-models": "dwarfstar",
    }.get(artifact.target, "comfyui")
    root = StorageLayout(data_dir).staging_for(application)
    if artifact.source.archive_member:
        # Civitai may replace a ZIP behind one model-version ID. Key shared
        # staging by the stable source identity, while the extracted member
        # hash remains the immutable content identity.
        identity = "\0".join(
            (
                artifact.source.provider,
                artifact.source.repository,
                artifact.source.revision,
                artifact.source.path,
                artifact.source.download_url,
            )
        )
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return root / ".archives" / key
    return root / artifact.identifier


def artifact_payload_path(data_dir: Path, artifact: Artifact) -> Path:
    staging = artifact_staging_root(data_dir, artifact)
    if artifact.source.archive_member:
        return (
            staging
            / ".extracted"
            / artifact.identifier
            / Path(artifact.destination).name
        )
    return staging / Path(artifact.source.path)


def artifact_download_path(data_dir: Path, artifact: Artifact) -> Path:
    return artifact_staging_root(data_dir, artifact) / Path(
        artifact.source.path
    )


def artifact_download_size(artifact: Artifact) -> int:
    """Return an exact direct size or the hard limit for an archive."""
    return artifact.source.archive_max_size or artifact.size


def _archive_member(
    artifact: Artifact, bundle: zipfile.ZipFile
) -> zipfile.ZipInfo:
    matches = [
        item
        for item in bundle.infolist()
        if item.filename == artifact.source.archive_member
    ]
    if len(matches) != 1:
        raise LauncherError(
            "source archive layout changed: expected exactly one member "
            "{!r}, found {}".format(
                artifact.source.archive_member, len(matches)
            )
        )
    member = matches[0]
    member_mode = (member.external_attr >> 16) & 0xFFFF
    member_type = stat.S_IFMT(member_mode)
    if (
        member.is_dir()
        or member_type not in (0, stat.S_IFREG)
    ):
        raise LauncherError(
            "source archive member is not a regular file: {!r}".format(
                artifact.source.archive_member
            )
        )
    return member


def _archive_member_changed(
    artifact: Artifact,
    archive: Path,
    actual_size: int,
    actual_sha256: str,
) -> LauncherError:
    actual = (
        "{} bytes and SHA-256 {}".format(actual_size, actual_sha256)
        if len(actual_sha256) == 64
        else "{} bytes ({})".format(actual_size, actual_sha256)
    )
    return LauncherError(
        "source archive member changed upstream: {!r} from {} @ {}; "
        "expected {} bytes and SHA-256 {}, got {}. "
        "The archive was kept at {} for review; update the catalog only "
        "after reviewing the new member.".format(
            artifact.source.archive_member,
            artifact.source.repository,
            artifact.source.revision,
            artifact.size,
            artifact.sha256,
            actual,
            archive,
        )
    )


def _copy_archive_member(
    artifact: Artifact,
    bundle: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    output,
) -> Tuple[int, str]:
    if member.file_size != artifact.size:
        raise _archive_member_changed(
            artifact,
            Path(bundle.filename),
            member.file_size,
            "not computed because the member size changed",
        )
    digest = hashlib.sha256()
    copied = 0
    with bundle.open(member) as source:
        while True:
            block = source.read(
                min(1024 * 1024, artifact.size - copied + 1)
            )
            if not block:
                break
            copied += len(block)
            if copied > artifact.size:
                raise _archive_member_changed(
                    artifact,
                    Path(bundle.filename),
                    copied,
                    "not computed because the member exceeded its pinned size",
                )
            digest.update(block)
            if output is not None:
                output.write(block)
    return copied, digest.hexdigest()


def _staged_archive_has_pinned_member(
    artifact: Artifact, archive: Path
) -> bool:
    try:
        status = archive.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise LauncherError(
            "cannot inspect staging entry {}: {}".format(archive, error)
        )
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise LauncherError(
            "refusing unexpected staging entry: {}".format(archive)
        )
    if (
        status.st_size <= 0
        or status.st_size > artifact.source.archive_max_size
    ):
        return False
    try:
        with zipfile.ZipFile(str(archive)) as bundle:
            member = _archive_member(artifact, bundle)
            actual_size, actual_digest = _copy_archive_member(
                artifact, bundle, member, None
            )
    except zipfile.BadZipFile:
        return False
    except (OSError, RuntimeError) as error:
        raise LauncherError(
            "cannot inspect source archive {}: {}".format(archive, error)
        )
    if (
        actual_size != artifact.size
        or actual_digest != artifact.sha256
    ):
        raise _archive_member_changed(
            artifact, archive, actual_size, actual_digest
        )
    return True


def _extract_archive_artifact(
    artifact: Artifact, archive: Path, destination: Path
) -> None:
    try:
        with zipfile.ZipFile(str(archive)) as bundle:
            member = _archive_member(artifact, bundle)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name + ".partial")
            try:
                with partial.open("wb") as output:
                    actual_size, actual_digest = _copy_archive_member(
                        artifact, bundle, member, output
                    )
                if (
                    actual_size != artifact.size
                    or actual_digest != artifact.sha256
                ):
                    raise _archive_member_changed(
                        artifact, archive, actual_size, actual_digest
                    )
                os.replace(str(partial), str(destination))
            finally:
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise LauncherError(
            "cannot extract source archive {}: {}".format(archive, error)
        )


def _prune_completed_staging(staging: Path, data_dir: Path) -> None:
    _validate_staging_destination(staging / ".rocmplete-path-check", data_dir)
    try:
        shutil.rmtree(str(staging))
        parent = staging.parent
        staging_root = StorageLayout(data_dir).staging
        while parent != staging_root.parent and parent != data_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            if parent == staging_root:
                break
            parent = parent.parent
    except FileNotFoundError:
        return
    except OSError as error:
        print(
            "{} could not remove completed staging {}: {}".format(
                style("WARNING:", "warning"), staging, error
            )
        )


def inspect_artifacts(
    artifacts: Iterable[Artifact],
    data_dir: Path,
    verification_store: Optional[VerificationStore] = None,
) -> Tuple[ArtifactStatus, ...]:
    store = verification_store or VerificationStore.load(data_dir)
    statuses = []
    for artifact in artifacts:
        destination = artifact_path(data_dir, artifact)
        try:
            status = destination.lstat()
        except FileNotFoundError:
            statuses.append(ArtifactStatus(artifact, destination, "missing"))
            continue
        except OSError as error:
            raise LauncherError("cannot inspect {}: {}".format(destination, error))
        size = status.st_size
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            state = "user-file"
        else:
            state = "installed" if size == artifact.size else "size-mismatch"
        integrity = (
            "verified"
            if state == "installed"
            and store.matches(destination, artifact.size, artifact.sha256)
            else "unverified"
        )
        statuses.append(
            ArtifactStatus(artifact, destination, state, size, integrity)
        )
    return tuple(statuses)


def content_status_ready(status: ContentStatus) -> bool:
    """Return whether managed bytes and their materialization are trusted."""
    return status.state == "installed" and status.integrity == "verified"


def content_status_state(status: ContentStatus) -> str:
    if status.state == "installed" and status.integrity != "verified":
        return "unverified"
    return status.state


def selection_artifacts(
    catalog: Catalog, bundles: Iterable[Bundle]
) -> Tuple[Artifact, ...]:
    """Return a stable, deduplicated artifact list for several bundles."""
    identifiers = set()
    artifacts = []
    for bundle in bundles:
        for artifact in catalog.bundle_artifacts(bundle):
            if artifact.identifier not in identifiers:
                identifiers.add(artifact.identifier)
                artifacts.append(artifact)
    return tuple(artifacts)


def sha256_file(
    path: Path, progress: Optional[Callable[[int], None]] = None
) -> str:
    digest = hashlib.sha256()
    hashed = 0
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                hashed += len(block)
                if progress is not None:
                    progress(hashed)
    except OSError as error:
        raise LauncherError("cannot hash {}: {}".format(path, error))
    return digest.hexdigest()


_FILE_IDENTITY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)

_FILE_OBJECT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
)


def _same_file_identity(
    first: os.stat_result, second: os.stat_result
) -> bool:
    return all(
        getattr(first, field) == getattr(second, field)
        for field in _FILE_IDENTITY_FIELDS
    )


def _same_file_object(
    first: os.stat_result, second: os.stat_result
) -> bool:
    """Compare identity fields that remain stable across a same-FS rename."""
    return all(
        getattr(first, field) == getattr(second, field)
        for field in _FILE_OBJECT_FIELDS
    )


def _stable_file_digest(
    path: Path,
    expected_size: int,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[str, os.stat_result]:
    """Hash one stable regular file and return its digest and identity."""
    try:
        before = path.lstat()
    except OSError as error:
        raise LauncherError("cannot inspect {}: {}".format(path, error))
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        raise LauncherError(
            "cannot verify unexpected managed content file: {}".format(path)
        )
    digest = sha256_file(path, progress)
    try:
        after = path.lstat()
    except OSError as error:
        raise LauncherError("cannot re-inspect {}: {}".format(path, error))
    if not stat.S_ISREG(after.st_mode) or not _same_file_identity(
        before, after
    ):
        raise LauncherError(
            "managed content changed while it was being verified: {}".format(path)
        )
    return digest, after


def _verified_file_status(
    path: Path,
    expected_size: int,
    expected_sha256: str,
    progress: Optional[Callable[[int], None]] = None,
) -> os.stat_result:
    """Hash one stable regular file and return the identity that was read."""
    digest, status = _stable_file_digest(path, expected_size, progress)
    if digest != expected_sha256:
        raise LauncherError(
            "SHA-256 mismatch for installed content {} (expected {}, got {}); "
            "the file was preserved for review".format(
                path, expected_sha256, digest
            )
        )
    return status


def verify_status(status: ContentStatus) -> str:
    if status.state != "installed":
        return status.state
    return (
        "verified"
        if sha256_file(status.path) == status.artifact.sha256
        else "hash-mismatch"
    )


def _status_size(status: ContentStatus) -> int:
    return status.artifact.size


def missing_size(statuses: Iterable[ContentStatus]) -> int:
    sizes = {}
    for status in statuses:
        if status.state == "missing":
            sizes[status.artifact.identifier] = status.artifact.size
    return sum(sizes.values())


def verification_size(statuses: Iterable[ContentStatus]) -> int:
    """Return bytes that must be hashed before managed content is ready."""
    sizes = {}
    for status in statuses:
        if status.state == "installed" and status.integrity != "verified":
            sizes[status.artifact.identifier] = status.artifact.size
    return sum(sizes.values())


def missing_download_size(statuses: Iterable[ContentStatus]) -> int:
    """Return network bytes, counting a shared source archive only once."""
    sizes = {}
    for status in statuses:
        if status.state != "missing":
            continue
        artifact = status.artifact
        if artifact.source.archive_member:
            key = (
                "archive",
                artifact.source.provider,
                artifact.source.repository,
                artifact.source.revision,
                artifact.source.path,
            )
            sizes[key] = artifact.source.archive_max_size
        else:
            sizes[("artifact", artifact.identifier)] = artifact.size
    return sum(sizes.values())


def missing_unverified(
    statuses: Iterable[ContentStatus],
) -> Tuple[ContentStatus, ...]:
    return tuple(
        status
        for status in statuses
        if status.state == "missing"
        and status.artifact.license.status == "unverified"
    )


def _status_warning(status: ContentStatus) -> str:
    return status.artifact.license.warning


def _download_container_command(
    data_dir: Path,
    volume_suffix: str,
    pass_hf_token: bool,
    pass_civitai_token: bool = False,
) -> List[str]:
    command = [
        "podman",
        "run",
        "--rm",
        "--userns",
        "keep-id",
        "--umask",
        podman.current_umask(),
        "--read-only",
        "--cap-drop",
        "all",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
        "--volume",
        "{}:/storage{}".format(data_dir, volume_suffix),
        "--env",
        "HOME=/storage/staging/.home",
        "--env",
        "XDG_CACHE_HOME=/storage/staging/.cache",
        "--env",
        "HF_HOME=/storage/staging/.cache/huggingface",
        "--env",
        "HF_HUB_DISABLE_TELEMETRY=1",
        "--env",
        "HF_HUB_DISABLE_PROGRESS_BARS=1",
        "--env",
        "HF_HUB_DISABLE_UPDATE_CHECK=1",
    ]
    if pass_hf_token:
        command.extend(["--env", "HF_TOKEN"])
    if pass_civitai_token:
        command.extend(["--env", "CIVITAI_TOKEN"])
    return command


def download_command(
    image: str,
    data_dir: Path,
    artifact: Artifact,
    volume_suffix: str,
    pass_hf_token: bool,
    pass_civitai_token: bool = False,
) -> List[str]:
    staging_root = artifact_staging_root(data_dir, artifact)
    try:
        relative_staging = staging_root.relative_to(data_dir)
    except ValueError as error:
        raise LauncherError(
            "artifact staging path escapes the persistent data directory"
        ) from error
    local_dir = "/storage/{}".format(relative_staging.as_posix())
    command = _download_container_command(
        data_dir,
        volume_suffix,
        pass_hf_token,
        pass_civitai_token,
    )
    if artifact.source.provider == "civitai":
        command.extend(
            [
                "--entrypoint",
                "/opt/venv/bin/python",
                image,
                "/opt/rocmplete/container_download.py",
                "--url",
                artifact.source.download_url
                or "https://civitai.com/api/download/models/{}".format(
                    artifact.source.model_version_id
                ),
                "--output",
                "{}/{}".format(local_dir, artifact.source.path),
            ]
        )
        if artifact.source.archive_member:
            command.extend(
                [
                    "--maximum-size",
                    str(artifact.source.archive_max_size),
                ]
            )
        else:
            command.extend(
                ["--expected-size", str(artifact.size)]
            )
        command.extend(["--token-env", "CIVITAI_TOKEN"])
        return command
    command.extend(
        [
            "--entrypoint",
            "/opt/venv/bin/hf",
            image,
            "download",
            artifact.source.repository,
            artifact.source.path,
            "--revision",
            artifact.source.revision,
            "--local-dir",
            local_dir,
            "--max-workers",
            "4",
        ]
    )
    return command


def print_plan(
    catalog: Catalog, bundle: Bundle, data_dir: Path
) -> Tuple[ContentStatus, ...]:
    statuses = inspect_bundle(catalog, bundle, data_dir)
    print(
        "{} — {}".format(
            style(bundle.identifier, "heading"), bundle.description
        )
    )
    print(
        "{} {}".format(
            style("Application:", "label"), bundle.application
        )
    )
    _print_content_plan(
        statuses,
        data_dir,
        total_label="Bundle size",
        total_size=catalog.bundle_size(bundle),
    )
    return statuses


def print_selection_plan(
    catalog: Catalog,
    bundles: Sequence[Bundle],
    data_dir: Path,
    target: str,
) -> Tuple[ContentStatus, ...]:
    artifacts = selection_artifacts(catalog, bundles)
    verification_store = VerificationStore.load(data_dir)
    statuses = inspect_artifacts(artifacts, data_dir, verification_store)
    print(
        "{} — {} {}, {} direct {}".format(
            style(target, "heading"),
            len(bundles),
            "bundle" if len(bundles) == 1 else "bundles",
            len(artifacts),
            "artifact" if len(artifacts) == 1 else "artifacts",
        )
    )
    print(style("Selected bundles:", "heading"))
    for bundle in bundles:
        print("  {}".format(style(bundle.identifier, "command")))
    applications = ", ".join(
        dict.fromkeys(bundle.application for bundle in bundles)
    )
    print(
        "{} {}".format(style("Applications:", "label"), applications)
    )
    unique_sizes = {item.identifier: item.size for item in artifacts}
    _print_content_plan(
        statuses,
        data_dir,
        total_label="Unique size",
        total_size=sum(unique_sizes.values()),
    )
    return statuses


def print_selection_summary(
    catalog: Catalog,
    bundles: Sequence[Bundle],
    data_dir: Path,
) -> Tuple[ContentStatus, ...]:
    """Print the decision-relevant subset of a full content audit."""
    artifacts = selection_artifacts(catalog, bundles)
    verification_store = VerificationStore.load(data_dir)
    statuses = inspect_artifacts(artifacts, data_dir, verification_store)
    installed = sum(content_status_ready(item) for item in statuses)
    required = missing_download_size(statuses)
    verify = verification_size(statuses)
    free = shutil.disk_usage(_nearest_existing_path(data_dir)).free
    print(style("Content state:", "heading"))
    print(
        "  {} {}/{} files".format(
            style("Ready:", "label"), installed, len(statuses)
        )
    )
    print(
        "  {} {}".format(
            style("Download:", "label"), human_size(required)
        )
    )
    print(
        "  {} {}".format(style("Verify:", "label"), human_size(verify))
    )
    print(
        "  {} {}".format(style("Disk free:", "label"), human_size(free))
    )
    conflicts = tuple(
        item
        for item in statuses
        if item.state in ("size-mismatch", "user-file")
    )
    for item in conflicts:
        print(
            "  {} {} ({})".format(
                style("Conflict:", "error"), item.path, item.state
            )
        )
    if required:
        print(
            style(
                "  Resumable staging and caches may temporarily need "
                "additional disk space.",
                "muted",
            )
        )
    return statuses


def _print_content_plan(
    statuses: Sequence[ContentStatus],
    data_dir: Path,
    total_label: str,
    total_size: int,
) -> None:
    required = missing_download_size(statuses)
    verify = verification_size(statuses)
    free = shutil.disk_usage(_nearest_existing_path(data_dir)).free
    if any(
        status.artifact.target == "models"
        for status in statuses
    ):
        print(
            "{} {}".format(
                style("ComfyUI models:", "label"),
                StorageLayout(data_dir).comfy_models,
            )
        )
    if any(
        status.artifact.target == "llama-models"
        for status in statuses
    ):
        print(
            "{} {}".format(
                style("llama.cpp models:", "label"),
                StorageLayout(data_dir).llama_models,
            )
        )
    if any(
        status.artifact.target == "dwarfstar-models"
        for status in statuses
    ):
        print(
            "{} {}".format(
                style("DwarfStar models:", "label"),
                StorageLayout(data_dir).dwarfstar_models,
            )
        )
    if any(
        status.artifact.target == "workflows"
        for status in statuses
    ):
        print(
            "{} {}".format(
                style("Imported workflows:", "label"),
                StorageLayout(data_dir).imported_workflows,
            )
        )
        print(
            style(
                "Note: exact imported workflows may require external custom "
                "nodes and assets that the catalog does not install.",
                "muted",
            )
        )
    for status in statuses:
        description = (
            status.artifact.destination
            if status.artifact.target in (
                "models",
                "llama-models",
                "dwarfstar-models",
            )
            else "workflow/{}".format(status.artifact.destination)
        )
        repository = status.artifact.source.repository
        revision = status.artifact.source.revision
        license_info = status.artifact.license
        effective_state = content_status_state(status)
        marker, marker_role = {
            "installed": ("installed", "success"),
            "unverified": ("verify", "warning"),
            "missing": ("download", "warning"),
            "size-mismatch": ("ERROR: size mismatch", "error"),
            "user-file": ("ERROR: user file", "error"),
        }[effective_state]
        license_marker = (
            "{} verified".format(license_info.spdx)
            if license_info.status == "verified"
            else "NOASSERTION / acknowledgment required"
        )
        print(
            "  {}  {:>12}  {}".format(
                style("{:>20}".format(marker), marker_role),
                human_size(_status_size(status)),
                description,
            )
        )
        print(
            "    source:  {} @ {}".format(
                repository, revision
            )
        )
        print(
            "    {} {} ({})".format(
                style("license:", "label"),
                style(
                    license_marker,
                    "success"
                    if license_info.status == "verified"
                    else "warning",
                ),
                license_info.url,
            )
        )
    print(
        "{} {}".format(
            style("{}:".format(total_label), "label"),
            human_size(total_size),
        )
    )
    print(
        "{} {}".format(
            style("Download:", "label"), human_size(required)
        )
    )
    print(
        "{} {}".format(style("Verify:", "label"), human_size(verify))
    )
    print(
        "{} {}".format(
            style("Disk free:", "label"), human_size(free)
        )
    )
    print(
        style(
            "Note: resumable downloader staging and caches may require "
            "additional disk space.",
            "muted",
        )
    )
    warnings = {
        _status_warning(item)
        for item in missing_unverified(statuses)
    }
    for warning in sorted(warnings):
        print(
            "{} {}".format(style("WARNING:", "warning"), warning)
        )


def install_bundle(
    catalog: Catalog,
    bundle: Bundle,
    data_dir: Path,
    image: str,
    acknowledge_license_risk: bool = False,
    local_mirror: Optional[LocalMirror] = None,
) -> int:
    result = install_artifacts(
        catalog.bundle_artifacts(bundle),
        data_dir,
        image,
        acknowledge_license_risk=acknowledge_license_risk,
        local_mirror=local_mirror,
    )
    if result == 0:
        print(
            "\n{} Bundle {} artifacts are ready.".format(
                style("Ready:", "success"), bundle.identifier
            )
        )
    return result


def install_artifacts(
    artifacts: Iterable[Artifact],
    data_dir: Path,
    image: str,
    acknowledge_license_risk: bool = False,
    local_mirror: Optional[LocalMirror] = None,
) -> int:
    artifacts = tuple(artifacts)
    verification_store = VerificationStore.load(data_dir)
    statuses = inspect_artifacts(artifacts, data_dir, verification_store)
    mismatched = [
        status
        for status in statuses
        if status.state in ("size-mismatch", "user-file")
    ]
    if mismatched:
        paths = ", ".join(str(status.path) for status in mismatched)
        raise LauncherError(
            "refusing to replace existing model files with unexpected "
            "sizes or types: {}; "
            "move or remove them explicitly first".format(paths)
        )
    risky = missing_unverified(statuses)
    if risky and not acknowledge_license_risk:
        raise LauncherError(
            "bundle contains missing artifacts with unverified licensing; "
            "explicit risk acknowledgment is required"
        )
    unverified = [
        status
        for status in statuses
        if status.state == "installed" and not content_status_ready(status)
    ]
    for status in unverified:
        artifact = status.artifact
        _prepare_runtime_label(artifact, status.path)
        print(
            "\n{} SHA-256 for existing {} ({})".format(
                style("Verifying", "heading"),
                artifact.destination,
                human_size(artifact.size),
            ),
            flush=True,
        )
        verification = _VerificationProgress(artifact.size)
        verification.update(artifact.destination, 1, 1, 0)
        try:
            verified = _verified_file_status(
                status.path,
                artifact.size,
                artifact.sha256,
                lambda hashed: verification.update(
                    artifact.destination, 1, 1, hashed
                ),
            )
            verification.update(
                artifact.destination, 1, 1, artifact.size, force=True
            )
        except BaseException:
            verification.finish()
            raise
        verification_store.record(
            status.path, artifact.size, artifact.sha256, verified
        )
    missing = [status for status in statuses if status.state == "missing"]
    if not missing:
        verification_store.save()
        print(
            style(
                "All selected artifacts are verified and ready.",
                "success",
            )
        )
        return 0
    mirror_matches = {}
    network_missing = []
    required_downloads = set()
    required = 0
    for status in missing:
        artifact = status.artifact
        staging_root = artifact_staging_root(data_dir, artifact)
        payload = artifact_payload_path(data_dir, artifact)
        download = artifact_download_path(data_dir, artifact)
        if _staged_file_matches(payload, artifact.size):
            continue
        match = (
            local_mirror.find(
                (
                    artifact.sha256,
                    Path(artifact.destination).name,
                    Path(artifact.source.path).name,
                ),
                artifact.size,
                artifact.sha256,
            )
            if local_mirror is not None
            else None
        )
        if match is None:
            if artifact.source.archive_member:
                staged_download_ready = _staged_archive_has_pinned_member(
                    artifact, download
                )
            else:
                staged_download_ready = _staged_file_matches(
                    download, artifact_download_size(artifact)
                )
            if not staged_download_ready:
                download_key = str(download)
                if download_key not in required_downloads:
                    network_missing.append(status)
                    download_size = artifact_download_size(artifact)
                    if artifact.source.archive_member:
                        remaining = download_size
                    else:
                        target = _DownloadTarget(
                            download,
                            download_size,
                            artifact.sha256,
                        )
                        remaining = (
                            download_size
                            - _RetryAwareDownloadSize(
                                staging_root, (target,)
                            ).reusable_partial_size()
                        )
                    required += remaining
                    required_downloads.add(download_key)
            if artifact.source.archive_member:
                required += artifact.size
        else:
            mirror_matches[artifact.identifier] = match
            required += local_mirror.required_space(match, data_dir)

    protected_civitai = [
        status.artifact.identifier
        for status in network_missing
        if status.artifact.source.provider == "civitai"
        and status.artifact.source.requires_auth
    ]
    if protected_civitai and not os.environ.get("CIVITAI_TOKEN"):
        raise LauncherError(
            "CIVITAI_TOKEN is required to download: {}".format(
                ", ".join(protected_civitai)
            )
        )

    free = shutil.disk_usage(_nearest_existing_path(data_dir)).free
    if free < required:
        raise LauncherError(
            "not enough free space: need at least {}, have {}".format(
                human_size(required), human_size(free)
            )
        )
    if network_missing:
        podman.require_rootless()
        if not podman.image_exists(image):
            raise LauncherError(
                "content tools image not found: {} "
                "(run './rocmplete build comfyui'; any application build "
                "provides it)".format(
                    image
                )
            )

    # Download staging is intentionally resumable across short-lived
    # containers. A shared label prevents an interrupted container's private
    # MCS category from making its partial files inaccessible to the retry.
    volume_suffix = podman.shared_selinux_volume_suffix()
    pass_hf_token = bool(os.environ.get("HF_TOKEN"))
    pass_civitai_token = bool(os.environ.get("CIVITAI_TOKEN"))
    completed_staging_roots = set()
    completed_downloads = set()
    for index, status in enumerate(missing, 1):
        artifact = status.artifact
        staging_root = artifact_staging_root(data_dir, artifact)
        _prepare_staging_root(staging_root, data_dir)
        payload = artifact_payload_path(data_dir, artifact)
        download = artifact_download_path(data_dir, artifact)
        _validate_staging_destination(payload, data_dir)
        _validate_staging_destination(download, data_dir)
        match = mirror_matches.get(artifact.identifier)
        if (
            not _staged_file_matches(payload, artifact.size)
            and match is not None
        ):
            local_mirror.materialize(match, payload, data_dir)
        download_key = str(download)
        if (
            not _staged_file_matches(payload, artifact.size)
            and match is None
            and download_key in required_downloads
            and download_key not in completed_downloads
        ):
            StorageLayout(data_dir).prepare_downloads()
            size_description = (
                "up to {}".format(human_size(artifact_download_size(artifact)))
                if artifact.source.archive_member
                else human_size(artifact_download_size(artifact))
            )
            print(
                "\n{} [{}/{}] {} ({})".format(
                    style("Downloading", "heading"),
                    index,
                    len(missing),
                    artifact.destination,
                    size_description,
                ),
                flush=True,
            )
            download_target = _DownloadTarget(
                download,
                artifact_download_size(artifact),
                (
                    staging_root.name
                    if artifact.source.archive_member
                    else artifact.sha256
                ),
            )
            progress = _DownloadProgress(
                staging_root,
                download_target.size,
                _RetryAwareDownloadSize(
                    staging_root, (download_target,)
                ),
                bounded=bool(artifact.source.archive_member),
            )
            try:
                result = podman.run_with_progress(
                    download_command(
                        image,
                        data_dir,
                        artifact,
                        volume_suffix,
                        pass_hf_token,
                        pass_civitai_token,
                    ),
                    progress.update,
                )
            except BaseException:
                progress.finish()
                raise
            progress.update(force=True)
            if result != 0:
                raise LauncherError(
                    "download failed for {} (exit status {})".format(
                        artifact.source.path, result
                    )
                )
            completed_downloads.add(download_key)
        if artifact.source.archive_member and (
            not _staged_file_matches(payload, artifact.size)
        ):
            print(
                "{} {} from {}".format(
                    style("Extracting:", "heading"),
                    artifact.source.archive_member,
                    artifact.source.path,
                ),
                flush=True,
            )
            _extract_archive_artifact(artifact, download, payload)
        if not _staged_file_matches(payload, artifact.size):
            raise LauncherError(
                "downloader did not create the expected regular file: {}".format(
                    payload
                )
            )
        _prepare_runtime_label(artifact, payload)
        print(
            "\n{} SHA-256 for {} ({})".format(
                style("Verifying", "heading"),
                artifact.destination,
                human_size(artifact.size),
            ),
            flush=True,
        )
        verification = _VerificationProgress(artifact.size)
        verification.update(artifact.destination, 1, 1, 0)
        try:
            actual_digest, verified_payload = _stable_file_digest(
                payload,
                artifact.size,
                lambda hashed: verification.update(
                    artifact.destination, 1, 1, hashed
                ),
            )
            verification.update(
                artifact.destination, 1, 1, artifact.size, force=True
            )
        except BaseException:
            verification.finish()
            raise
        if actual_digest != artifact.sha256:
            quarantine = _quarantine_staged_file(payload, actual_digest)
            raise LauncherError(
                "SHA-256 mismatch for {} (expected {}, got {}); corrupt "
                "staging was preserved at {}".format(
                    payload, artifact.sha256, actual_digest, quarantine
                )
            )
        destination = status.path
        _validate_managed_destination(
            destination,
            _artifact_root(data_dir, artifact),
            data_dir,
            "content",
        )
        if destination.exists():
            raise LauncherError(
                "destination appeared during download; refusing to overwrite: {}".format(
                    destination
                )
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _validate_managed_destination(
            destination,
            _artifact_root(data_dir, artifact),
            data_dir,
            "content",
        )
        try:
            preinstall_status = payload.lstat()
        except OSError as error:
            raise LauncherError(
                "cannot re-inspect verified staging {}: {}".format(
                    payload, error
                )
            )
        if not stat.S_ISREG(
            preinstall_status.st_mode
        ) or not _same_file_identity(verified_payload, preinstall_status):
            raise LauncherError(
                "managed content changed after it was verified: {}".format(
                    payload
                )
            )
        try:
            os.replace(str(payload), str(destination))
        except OSError as error:
            raise LauncherError("cannot install {}: {}".format(destination, error))
        _validate_managed_destination(
            destination,
            _artifact_root(data_dir, artifact),
            data_dir,
            "content",
        )
        try:
            installed_status = destination.lstat()
        except OSError as error:
            raise LauncherError(
                "cannot inspect installed content {}: {}".format(
                    destination, error
                )
            )
        if not stat.S_ISREG(installed_status.st_mode) or not _same_file_object(
            preinstall_status, installed_status
        ):
            raise LauncherError(
                "managed content changed after it was verified: {}".format(
                    destination
                )
            )
        verification_store.record(
            destination,
            artifact.size,
            artifact.sha256,
            installed_status,
        )
        print(
            "{} {}".format(style("Installed", "success"), destination)
        )
        completed_staging_roots.add(staging_root)
    verification_store.save()
    for staging_root in completed_staging_roots:
        _prune_completed_staging(staging_root, data_dir)
    print(
        "\n{}".format(
            style("All selected artifacts are ready.", "success")
        )
    )
    return 0
