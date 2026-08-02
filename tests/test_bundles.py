import hashlib
import io
import os
import stat
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rocmplete.bundles import (
    _DownloadProgress,
    _DownloadTarget,
    _RetryAwareDownloadSize,
    _stable_file_digest,
    _VerificationProgress,
    _compact_progress_item,
    _extract_archive_artifact,
    LocalMirror,
    artifact_download_path,
    artifact_path,
    artifact_payload_path,
    artifact_staging_root,
    content_install_lock,
    content_status_ready,
    download_command,
    human_size,
    inspect_artifacts,
    inspect_bundle,
    install_bundle,
    install_artifacts,
    missing_download_size,
    missing_unverified,
    sha256_file,
    verify_status,
)
from rocmplete.catalog import (
    Artifact,
    ArtifactSource,
    Bundle,
    Catalog,
    LicenseInfo,
)
from rocmplete.errors import LauncherError
from rocmplete.content_verification import VerificationStore


def fake_artifact(contents=b"model", status="verified"):
    unverified = status == "unverified"
    return Artifact(
        identifier="test-artifact",
        description="Test artifact",
        source=ArtifactSource(
            repository="owner/repository",
            revision="a" * 40,
            path="files/model.bin",
        ),
        destination="checkpoints/model.bin",
        size=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
        license=LicenseInfo(
            spdx="NOASSERTION" if unverified else "Apache-2.0",
            status=status,
            url="https://example.invalid/source",
            warning="No license declared." if unverified else "",
            upstream_repository="owner/upstream" if unverified else "",
            upstream_license="Apache-2.0" if unverified else "",
            upstream_license_url=(
                "https://example.invalid/upstream" if unverified else ""
            ),
        ),
    )


def fake_catalog(artifact):
    bundle = Bundle(
        identifier="test-bundle",
        description="Test bundle",
        application="comfyui",
        artifacts=(artifact.identifier,),
        workflow="test-workflow",
    )
    return (
        Catalog(
            agreements={},
            artifacts={artifact.identifier: artifact},
            bundles={bundle.identifier: bundle},
            workflow_packs={},
            benchmarks={},
        ),
        bundle,
    )


class BundleTests(unittest.TestCase):
    def test_content_install_lock_serializes_one_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            runtime.mkdir(mode=0o700)
            data_dir = root / "data"
            data_dir.mkdir()
            with patch.dict(
                os.environ,
                {"XDG_RUNTIME_DIR": str(runtime)},
            ):
                with content_install_lock(data_dir):
                    with self.assertRaisesRegex(
                        LauncherError,
                        "another content installation is active",
                    ):
                        with content_install_lock(data_dir):
                            self.fail("second lock unexpectedly acquired")
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    with content_install_lock(data_dir):
                        raise RuntimeError("test failure")
                with content_install_lock(data_dir):
                    pass

    def test_long_verification_item_is_compacted_to_available_width(self):
        item = "low_noise_model/diffusion_pytorch_model-00005.safetensors"
        compacted = _compact_progress_item(item, 24)
        self.assertEqual(len(compacted), 24)
        self.assertTrue(compacted.startswith("low_noi"))
        self.assertTrue(compacted.endswith("05.safetensors"))

    def test_download_progress_reports_staged_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "partial").write_bytes(b"x" * 50)
            progress = _DownloadProgress(staging, 100)
            with patch("builtins.print") as output:
                progress.update()
            text = output.call_args.args[0]
            self.assertIn("50.0%", text)
            self.assertNotIn("~", text)
            self.assertIn("staged", text)
            self.assertIn("expected", text)
            self.assertNotIn(" / ", text)

    def test_bounded_download_progress_reports_limit_without_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "partial").write_bytes(b"x" * 50)
            progress = _DownloadProgress(staging, 100, bounded=True)
            with patch("builtins.print") as output:
                progress.update()
            text = output.call_args.args[0]
            self.assertIn("staged", text)
            self.assertIn("limit", text)
            self.assertNotIn("%", text)
            self.assertNotIn("expected", text)

    def test_retry_progress_ignores_abandoned_huggingface_partials(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            target = _DownloadTarget(
                staging / "weights" / "model.bin",
                100,
                "a" * 64,
            )
            cache = (
                staging
                / ".cache"
                / "huggingface"
                / "download"
                / "weights"
            )
            cache.mkdir(parents=True)
            stale = cache / "source=.{}.old.incomplete".format(
                target.sha256
            )
            stale.write_bytes(b"x" * 80)
            measure = _RetryAwareDownloadSize(staging, (target,))

            current = cache / "source=.{}.current.incomplete".format(
                target.sha256
            )
            current.write_bytes(b"x" * 20)
            self.assertEqual(measure(), 20)

            current.write_bytes(b"x" * 25)
            self.assertEqual(measure(), 25)

    def test_retry_progress_recognizes_resumed_and_completed_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            target = _DownloadTarget(
                staging / "model.bin",
                100,
                "b" * 64,
            )
            cache = (
                staging / ".cache" / "huggingface" / "download"
            )
            cache.mkdir(parents=True)
            partial = cache / "source=.{}.resume.incomplete".format(
                target.sha256
            )
            partial.write_bytes(b"x" * 40)
            measure = _RetryAwareDownloadSize(staging, (target,))

            partial.write_bytes(b"x" * 50)
            self.assertEqual(measure(), 50)

            target.path.write_bytes(b"x" * 100)
            self.assertEqual(measure(), 100)

    def test_reusable_partial_size_requires_one_unambiguous_hf_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            target = _DownloadTarget(
                staging / "weights" / "model.bin",
                100,
                "c" * 64,
            )
            cache = (
                staging
                / ".cache"
                / "huggingface"
                / "download"
                / "weights"
            )
            cache.mkdir(parents=True)
            first = cache / "source=.{}.resume.incomplete".format(
                target.sha256
            )
            first.write_bytes(b"x" * 40)
            self.assertEqual(
                _RetryAwareDownloadSize(
                    staging, (target,)
                ).reusable_partial_size(),
                40,
            )

            second = cache / "mirror=.{}.retry.incomplete".format(
                target.sha256
            )
            second.write_bytes(b"x" * 25)
            self.assertEqual(
                _RetryAwareDownloadSize(
                    staging, (target,)
                ).reusable_partial_size(),
                0,
            )

    def test_download_progress_rewrites_one_terminal_line(self):
        class TerminalBuffer(io.StringIO):
            def isatty(self):
                return True

        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            partial = staging / "partial"
            partial.write_bytes(b"x" * 50)
            progress = _DownloadProgress(staging, 100)
            output = TerminalBuffer()
            with redirect_stdout(output):
                progress.update()
                partial.write_bytes(b"x" * 51)
                progress.update()
                partial.write_bytes(b"x" * 100)
                progress.update(force=True)
            rendered = output.getvalue()
            self.assertEqual(rendered.count("\r"), 3)
            self.assertEqual(rendered.count("\n"), 1)
            self.assertIn("50.0%", rendered)
            self.assertIn("51.0%", rendered)
            self.assertIn("100.0%", rendered)

    @patch("rocmplete.bundles.time.monotonic", side_effect=(100.0, 101.0))
    def test_download_progress_throttles_redirected_logs(self, monotonic):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            partial = staging / "partial"
            partial.write_bytes(b"x" * 50)
            progress = _DownloadProgress(staging, 100)
            output = io.StringIO()
            with redirect_stdout(output):
                progress.update()
                partial.write_bytes(b"x" * 51)
                progress.update()
            self.assertEqual(output.getvalue().count("\n"), 1)
            self.assertIn("50.0%", output.getvalue())
            self.assertNotIn("51.0%", output.getvalue())

    @patch("rocmplete.bundles.time.monotonic", side_effect=(100.0, 160.0))
    def test_download_progress_keeps_minute_heartbeat_in_logs(
        self, monotonic
    ):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            partial = staging / "partial"
            partial.write_bytes(b"x" * 50)
            progress = _DownloadProgress(staging, 100)
            output = io.StringIO()
            with redirect_stdout(output):
                progress.update()
                partial.write_bytes(b"x" * 51)
                progress.update()
            self.assertEqual(output.getvalue().count("\n"), 2)
            self.assertIn("50.0%", output.getvalue())
            self.assertIn("51.0%", output.getvalue())

    def test_download_progress_does_not_repeat_final_redirected_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            (staging / "complete").write_bytes(b"x" * 100)
            progress = _DownloadProgress(staging, 100)
            output = io.StringIO()
            with redirect_stdout(output):
                progress.update()
                progress.update(force=True)
            self.assertEqual(output.getvalue().count("\n"), 1)
            self.assertEqual(output.getvalue().count("100.0%"), 1)

    @patch(
        "rocmplete.bundles.time.monotonic",
        side_effect=(100.0, 100.5, 101.5, 101.6),
    )
    def test_verification_progress_reports_current_file_and_aggregate(
        self, monotonic
    ):
        class TerminalBuffer(io.StringIO):
            def isatty(self):
                return True

        progress = _VerificationProgress(100)
        output = TerminalBuffer()
        with redirect_stdout(output):
            progress.update("first.bin", 1, 2, 0)
            progress.update("first.bin", 1, 2, 25)
            progress.update("second.bin", 2, 2, 50)
            progress.update("second.bin", 2, 2, 100, force=True)
        rendered = output.getvalue()
        self.assertEqual(rendered.count("\r"), 3)
        self.assertEqual(rendered.count("\n"), 1)
        self.assertIn("[1/2] first.bin", rendered)
        self.assertIn("[2/2] second.bin", rendered)
        self.assertIn("hashed", rendered)
        self.assertIn("100.0%", rendered)

    def test_sha256_file_reports_exact_bytes_hashed(self):
        contents = b"model contents"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(contents)
            updates = []
            self.assertEqual(
                sha256_file(path, updates.append),
                hashlib.sha256(contents).hexdigest(),
            )
        self.assertEqual(updates[-1], len(contents))

    def test_status_and_hash_verification(self):
        contents = b"model"
        artifact = fake_artifact(contents)
        catalog, bundle = fake_catalog(artifact)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            missing = inspect_bundle(catalog, bundle, data_dir)[0]
            self.assertEqual(missing.state, "missing")

            destination = (
                data_dir
                / "content"
                / "comfyui"
                / "models"
                / artifact.destination
            )
            destination.parent.mkdir(parents=True)
            destination.write_bytes(contents)
            installed = inspect_bundle(catalog, bundle, data_dir)[0]
            self.assertEqual(installed.state, "installed")
            self.assertEqual(installed.integrity, "unverified")
            self.assertFalse(content_status_ready(installed))
            self.assertEqual(verify_status(installed), "verified")

            store = VerificationStore.load(data_dir)
            store.record(
                destination, artifact.size, artifact.sha256
            )
            store.save()
            installed = inspect_bundle(catalog, bundle, data_dir)[0]
            self.assertTrue(content_status_ready(installed))

            destination.write_bytes(b"wrong!")
            mismatch = inspect_bundle(catalog, bundle, data_dir)[0]
            self.assertEqual(mismatch.state, "size-mismatch")

    def test_same_size_mutation_invalidates_receipt_and_install_fails_closed(self):
        contents = b"good!"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            destination = artifact_path(data_dir, artifact)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(contents)
            store = VerificationStore.load(data_dir)
            store.record(destination, artifact.size, artifact.sha256)
            store.save()
            self.assertTrue(
                content_status_ready(
                    inspect_artifacts((artifact,), data_dir)[0]
                )
            )

            destination.write_bytes(b"evil!")
            status = inspect_artifacts((artifact,), data_dir)[0]
            self.assertEqual(status.state, "installed")
            self.assertEqual(status.integrity, "unverified")
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    LauncherError, "SHA-256 mismatch for installed content"
                ):
                    install_artifacts((artifact,), data_dir, "unused")
            self.assertEqual(destination.read_bytes(), b"evil!")

    def test_staged_payload_changed_after_hash_is_not_receipted(self):
        contents = b"good!"
        replacement = b"evil!"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            staged = artifact_payload_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            staged.write_bytes(contents)

            def change_after_hash(path, expected_size, progress=None):
                digest, status = _stable_file_digest(
                    path, expected_size, progress
                )
                path.write_bytes(replacement)
                return digest, status

            with patch(
                "rocmplete.bundles._stable_file_digest",
                side_effect=change_after_hash,
            ), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    LauncherError, "changed after it was verified"
                ):
                    install_artifacts((artifact,), data_dir, "unused")

            destination = artifact_path(data_dir, artifact)
            self.assertEqual(staged.read_bytes(), replacement)
            self.assertFalse(destination.exists())

    def test_install_migrates_matching_existing_file_without_downloading(self):
        contents = b"existing model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            destination = artifact_path(data_dir, artifact)
            destination.parent.mkdir(parents=True)
            destination.write_bytes(contents)

            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(
                    install_artifacts((artifact,), data_dir, "unused"), 0
                )

            status = inspect_artifacts((artifact,), data_dir)[0]
            self.assertTrue(content_status_ready(status))
            self.assertIn("existing", output.getvalue())
            self.assertTrue(
                (
                    data_dir
                    / "content"
                    / ".rocmplete"
                    / "verification.json"
                ).is_file()
            )

    def test_unsupported_verification_receipt_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            receipt = (
                data_dir / "content" / ".rocmplete" / "verification.json"
            )
            receipt.parent.mkdir(parents=True)
            receipt.write_text('{"schema": 99, "files": {}}\n')
            with self.assertRaisesRegex(LauncherError, "unsupported.*schema"):
                VerificationStore.load(data_dir)

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_direct_artifact_rejects_symlinked_destination(
        self, require_rootless
    ):
        contents = b"model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            external = data_dir / "external.bin"
            external.write_bytes(contents)
            destination = artifact_path(data_dir, artifact)
            destination.parent.mkdir(parents=True)
            destination.symlink_to(external)

            status = inspect_artifacts((artifact,), data_dir)[0]
            self.assertEqual(status.state, "user-file")
            with self.assertRaisesRegex(LauncherError, "sizes or types"):
                install_artifacts((artifact,), data_dir, "unused")
        require_rootless.assert_not_called()

    @patch(
        "rocmplete.bundles.podman.selinux_volume_suffix",
        return_value=":rw,Z",
    )
    @patch("rocmplete.bundles.podman.image_exists", return_value=True)
    @patch("rocmplete.bundles.podman.require_rootless")
    def test_corrupt_complete_staging_is_quarantined_and_retry_downloads(
        self, require_rootless, image_exists, volume_suffix
    ):
        expected = b"correct model"
        artifact = fake_artifact(expected)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            staged = artifact_download_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            staged.write_bytes(b"X" * len(expected))

            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    LauncherError, "corrupt staging was preserved"
                ):
                    install_artifacts((artifact,), data_dir, "test-image")
            self.assertFalse(staged.exists())
            quarantined = tuple(
                staged.parent.glob(staged.name + ".invalid-*")
            )
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(
                quarantined[0].read_bytes(), b"X" * len(expected)
            )

            def download(command, update):
                self.assertIn(
                    "{}:/storage:rw,z".format(data_dir),
                    command,
                )
                staged.write_bytes(expected)
                return 0

            with patch(
                "rocmplete.bundles.podman.run_with_progress",
                side_effect=download,
            ) as run_with_progress:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        install_artifacts(
                            (artifact,), data_dir, "test-image"
                        ),
                        0,
                    )
            run_with_progress.assert_called_once()
            self.assertEqual(
                artifact_path(data_dir, artifact).read_bytes(), expected
            )

    @patch("rocmplete.bundles.podman.image_exists", return_value=False)
    @patch("rocmplete.bundles.podman.require_rootless")
    def test_direct_install_space_check_credits_resumable_hf_partial(
        self, require_rootless, image_exists
    ):
        artifact = fake_artifact(b"x" * 100)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            staging = artifact_staging_root(data_dir, artifact)
            target = _DownloadTarget(
                artifact_download_path(data_dir, artifact),
                artifact.size,
                artifact.sha256,
            )
            cache = (
                staging
                / ".cache"
                / "huggingface"
                / "download"
                / "files"
            )
            cache.mkdir(parents=True)
            partial = cache / "source=.{}.resume.incomplete".format(
                target.sha256
            )
            partial.write_bytes(b"x" * 50)
            with patch(
                "rocmplete.bundles.shutil.disk_usage",
                return_value=SimpleNamespace(free=60),
            ):
                with self.assertRaisesRegex(
                    LauncherError, "content tools image not found"
                ):
                    install_artifacts((artifact,), data_dir, "missing-image")
        require_rootless.assert_called_once()
        image_exists.assert_called_once_with("missing-image")

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_artifact_install_rejects_symlinked_staging(
        self, require_rootless
    ):
        contents = b"model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            external = data_dir / "external.bin"
            external.write_bytes(contents)
            staged = artifact_download_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            staged.symlink_to(external)
            with self.assertRaisesRegex(
                LauncherError, "unexpected staging entry"
            ):
                install_artifacts((artifact,), data_dir, "unused")
        require_rootless.assert_not_called()

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_artifact_install_rejects_staging_redirected_into_app_state(
        self, require_rootless
    ):
        contents = b"model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            app_state = data_dir / "apps" / "comfyui" / "user"
            app_state.mkdir(parents=True)
            staging = artifact_staging_root(data_dir, artifact)
            staging.parent.mkdir(parents=True)
            staging.symlink_to(app_state, target_is_directory=True)
            staged = artifact_payload_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            staged.write_bytes(contents)

            with self.assertRaisesRegex(
                LauncherError, "symlinked staging path component"
            ):
                install_artifacts((artifact,), data_dir, "unused")

            self.assertEqual(staged.read_bytes(), contents)
            self.assertFalse(artifact_path(data_dir, artifact).exists())
        require_rootless.assert_not_called()

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_artifact_install_rejects_content_root_redirected_into_app_state(
        self, require_rootless
    ):
        contents = b"model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            app_state = data_dir / "apps" / "comfyui" / "user"
            app_state.mkdir(parents=True)
            content = data_dir / "content" / "comfyui"
            content.mkdir(parents=True)
            (content / "models").symlink_to(
                app_state, target_is_directory=True
            )
            staged = artifact_payload_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            staged.write_bytes(contents)

            with self.assertRaisesRegex(
                LauncherError, "symlinked content path component"
            ):
                install_artifacts((artifact,), data_dir, "unused")

            self.assertEqual(staged.read_bytes(), contents)
            self.assertEqual(tuple(app_state.iterdir()), ())
        require_rootless.assert_not_called()

    def test_download_command_is_constrained_and_artifact_pinned(self):
        artifact = fake_artifact()
        command = download_command(
            "localhost/image",
            Path("/data/content"),
            artifact,
            ":rw,Z",
            True,
        )
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--userns") + 1], "keep-id")
        self.assertRegex(command[command.index("--umask") + 1], r"^0[0-7]{3}$")
        self.assertIn("no-new-privileges", command)
        self.assertIn("--cap-drop", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("host", command)
        self.assertIn(artifact.source.revision, command)
        self.assertIn(artifact.source.repository, command)
        self.assertTrue(
            any(artifact.identifier in argument for argument in command)
        )
        self.assertIn("HF_HUB_DISABLE_PROGRESS_BARS=1", command)
        self.assertIn("HF_HUB_DISABLE_UPDATE_CHECK=1", command)
        token_index = command.index("HF_TOKEN")
        self.assertEqual(command[token_index - 1], "--env")

    def test_civitai_download_command_is_pinned_and_passes_token_by_name(self):
        contents = b"civitai model"
        artifact = Artifact(
            identifier="civitai-model",
            description="Civitai model",
            source=ArtifactSource(
                repository="civitai.com/models/123",
                revision="456",
                path="model.safetensors",
                provider="civitai",
                model_id=123,
                model_version_id=456,
                requires_auth=True,
            ),
            destination="checkpoints/model.safetensors",
            size=len(contents),
            sha256=hashlib.sha256(contents).hexdigest(),
            license=LicenseInfo(
                "LicenseRef-Test",
                "verified",
                "https://example.invalid/license",
            ),
        )
        command = download_command(
            "localhost/image",
            Path("/data/content"),
            artifact,
            ":rw",
            False,
            True,
        )
        self.assertIn(
            "https://civitai.com/api/download/models/456",
            command,
        )
        self.assertIn("--read-only", command)
        self.assertEqual(command[command.index("--userns") + 1], "keep-id")
        self.assertRegex(command[command.index("--umask") + 1], r"^0[0-7]{3}$")
        self.assertIn("no-new-privileges", command)
        self.assertIn("--cap-drop", command)
        self.assertNotIn("--device", command)
        self.assertNotIn("host", command)
        self.assertIn("/opt/rocmplete/container_download.py", command)
        self.assertIn("CIVITAI_TOKEN", command)
        self.assertNotIn("/opt/venv/bin/hf", command)
        self.assertNotIn("secret-token", command)

    def test_civitai_archive_download_uses_bounded_transport(self):
        artifact = Artifact(
            identifier="archived-workflow",
            description="Archived workflow",
            source=ArtifactSource(
                repository="civitai.com/models/123",
                revision="456",
                path="workflow.zip",
                provider="civitai",
                model_id=123,
                model_version_id=456,
                archive_member="pack/workflow.json",
                archive_max_size=4096,
            ),
            destination="family/workflow.json",
            size=100,
            sha256="b" * 64,
            license=LicenseInfo(
                "NOASSERTION",
                "unverified",
                "https://example.invalid/workflow",
                warning="No license declared.",
            ),
            target="workflows",
        )

        command = download_command(
            "localhost/image",
            Path("/data/content"),
            artifact,
            ":rw",
            False,
        )

        self.assertEqual(
            command[command.index("--maximum-size") + 1], "4096"
        )
        self.assertNotIn("--expected-size", command)
        self.assertTrue(
            any(argument.endswith("/workflow.zip") for argument in command)
        )
        self.assertTrue(
            any(
                "/staging/comfyui/.archives/" in argument
                and argument.endswith("/workflow.zip")
                for argument in command
            )
        )

    def test_civitai_download_uses_the_validated_provider_url(self):
        artifact = Artifact(
            identifier="red-model",
            description="Civitai red model",
            source=ArtifactSource(
                repository="civitai.red/models/123",
                revision="456",
                path="model.safetensors",
                provider="civitai",
                model_id=123,
                model_version_id=456,
                provider_host="civitai.red",
                download_url=(
                    "https://civitai.red/api/download/models/456"
                    "?type=Model&format=SafeTensor"
                ),
            ),
            destination="checkpoints/model.safetensors",
            size=1024,
            sha256="a" * 64,
            license=LicenseInfo(
                "NOASSERTION",
                "unverified",
                "https://civitai.red/models/123?modelVersionId=456",
                warning="Unverified.",
                upstream_repository="Civitai model 123",
                upstream_license="Civitai model-page permissions",
            ),
        )
        command = download_command(
            "localhost/image",
            Path("/data/content"),
            artifact,
            ":rw",
            False,
        )
        self.assertIn(artifact.source.download_url, command)

    def test_pinned_archive_member_is_extracted_exactly(self):
        contents = b'{"workflow": true}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "workflow.zip"
            with zipfile.ZipFile(str(archive), "w") as bundle:
                bundle.writestr("pack/workflow.json", contents)
            artifact = Artifact(
                identifier="archived-workflow",
                description="Archived workflow",
                source=ArtifactSource(
                    repository="civitai.com/models/123",
                    revision="456",
                    path="workflow.zip",
                    provider="civitai",
                    model_id=123,
                    model_version_id=456,
                    archive_member="pack/workflow.json",
                    archive_max_size=4096,
                ),
                destination="family/workflow.json",
                size=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
                license=LicenseInfo(
                    "NOASSERTION",
                    "unverified",
                    "https://example.invalid/workflow",
                    warning="No license declared.",
                ),
                target="workflows",
            )
            destination = root / "extracted/workflow.json"

            _extract_archive_artifact(artifact, archive, destination)

            self.assertEqual(destination.read_bytes(), contents)

    def test_archive_member_change_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "workflow.zip"
            with zipfile.ZipFile(str(archive), "w") as bundle:
                bundle.writestr("pack/workflow.json", b"changed!")
            artifact = Artifact(
                identifier="archived-workflow",
                description="Archived workflow",
                source=ArtifactSource(
                    repository="civitai.com/models/123",
                    revision="456",
                    path="workflow.zip",
                    provider="civitai",
                    model_id=123,
                    model_version_id=456,
                    archive_member="pack/workflow.json",
                    archive_max_size=4096,
                ),
                destination="family/workflow.json",
                size=8,
                sha256=hashlib.sha256(b"workflow").hexdigest(),
                license=LicenseInfo(
                    "NOASSERTION",
                    "unverified",
                    "https://example.invalid/workflow",
                    warning="No license declared.",
                ),
                target="workflows",
            )

            with self.assertRaisesRegex(
                LauncherError, "member changed upstream"
            ):
                _extract_archive_artifact(
                    artifact, archive, root / "workflow.json"
                )

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_changed_staged_archive_fails_before_podman(
        self, require_rootless
    ):
        expected = b"workflow"
        artifact = Artifact(
            identifier="archived-workflow",
            description="Archived workflow",
            source=ArtifactSource(
                repository="civitai.com/models/123",
                revision="456",
                path="workflow.zip",
                provider="civitai",
                model_id=123,
                model_version_id=456,
                archive_member="pack/workflow.json",
                archive_max_size=4096,
            ),
            destination="family/workflow.json",
            size=len(expected),
            sha256=hashlib.sha256(expected).hexdigest(),
            license=LicenseInfo(
                "MIT",
                "verified",
                "https://example.invalid/license",
            ),
            target="workflows",
        )
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            staged = artifact_download_path(data_dir, artifact)
            staged.parent.mkdir(parents=True)
            with zipfile.ZipFile(str(staged), "w") as bundle:
                bundle.writestr("pack/workflow.json", b"changed!")

            with self.assertRaisesRegex(
                LauncherError, "member changed upstream"
            ):
                install_artifacts((artifact,), data_dir, "unused-image")

            self.assertTrue(staged.exists())
            self.assertFalse(artifact_path(data_dir, artifact).exists())
        require_rootless.assert_not_called()

    def test_archive_symlink_member_is_rejected(self):
        contents = b"target.json"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "workflow.zip"
            member = zipfile.ZipInfo("pack/workflow.json")
            member.create_system = 3
            member.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(str(archive), "w") as bundle:
                bundle.writestr(member, contents)
            artifact = Artifact(
                identifier="archived-workflow",
                description="Archived workflow",
                source=ArtifactSource(
                    repository="civitai.com/models/123",
                    revision="456",
                    path="workflow.zip",
                    provider="civitai",
                    model_id=123,
                    model_version_id=456,
                    archive_member="pack/workflow.json",
                    archive_max_size=4096,
                ),
                destination="family/workflow.json",
                size=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
                license=LicenseInfo(
                    "MIT",
                    "verified",
                    "https://example.invalid/license",
                ),
                target="workflows",
            )

            with self.assertRaisesRegex(
                LauncherError, "not a regular file"
            ):
                _extract_archive_artifact(
                    artifact, archive, root / "workflow.json"
                )

    def test_archive_repack_with_same_member_is_accepted(self):
        contents = b'{"workflow": true}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.zip"
            second = root / "second.zip"
            with zipfile.ZipFile(
                str(first), "w", compression=zipfile.ZIP_STORED
            ) as bundle:
                bundle.writestr("pack/workflow.json", contents)
            with zipfile.ZipFile(
                str(second), "w", compression=zipfile.ZIP_DEFLATED
            ) as bundle:
                bundle.comment = b"repacked"
                bundle.writestr("pack/workflow.json", contents)
            self.assertNotEqual(
                hashlib.sha256(first.read_bytes()).hexdigest(),
                hashlib.sha256(second.read_bytes()).hexdigest(),
            )
            artifact = Artifact(
                identifier="archived-workflow",
                description="Archived workflow",
                source=ArtifactSource(
                    repository="civitai.com/models/123",
                    revision="456",
                    path="workflow.zip",
                    provider="civitai",
                    model_id=123,
                    model_version_id=456,
                    archive_member="pack/workflow.json",
                    archive_max_size=4096,
                ),
                destination="family/workflow.json",
                size=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
                license=LicenseInfo(
                    "MIT",
                    "verified",
                    "https://example.invalid/license",
                ),
                target="workflows",
            )

            for index, archive in enumerate((first, second), 1):
                destination = root / "extracted-{}.json".format(index)
                _extract_archive_artifact(artifact, archive, destination)
                self.assertEqual(destination.read_bytes(), contents)

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_archive_members_share_one_staged_download(
        self, require_rootless
    ):
        members = {
            "luts/first.cube": b"first LUT",
            "luts/second.cube": b"second LUT",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "source.zip"
            with zipfile.ZipFile(str(archive), "w") as bundle:
                for name, contents in members.items():
                    bundle.writestr(name, contents)
            archive_contents = archive.read_bytes()

            artifacts = []
            for index, (member, contents) in enumerate(members.items(), 1):
                artifacts.append(
                    Artifact(
                        identifier="shared-member-{}".format(index),
                        description="Shared archive member",
                        source=ArtifactSource(
                            repository="civitai.com/models/123",
                            revision="456",
                            path="shared.zip",
                            provider="civitai",
                            model_id=123,
                            model_version_id=456,
                            archive_member=member,
                            archive_max_size=4096,
                        ),
                        destination=member,
                        size=len(contents),
                        sha256=hashlib.sha256(contents).hexdigest(),
                        license=LicenseInfo(
                            "MIT",
                            "verified",
                            "https://example.invalid/license",
                        ),
                    )
                )

            data_dir = root / "data"
            data_dir.mkdir()
            self.assertEqual(
                artifact_staging_root(data_dir, artifacts[0]),
                artifact_staging_root(data_dir, artifacts[1]),
            )
            statuses = inspect_artifacts(artifacts, data_dir)
            self.assertEqual(
                missing_download_size(statuses), 4096
            )
            staged_archive = artifact_download_path(data_dir, artifacts[0])
            staged_archive.parent.mkdir(parents=True)
            staged_archive.write_bytes(archive_contents)

            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    install_artifacts(artifacts, data_dir, "unused-image"),
                    0,
                )

            for artifact, contents in zip(artifacts, members.values()):
                self.assertEqual(
                    artifact_path(data_dir, artifact).read_bytes(), contents
                )
            self.assertFalse(staged_archive.exists())
        require_rootless.assert_not_called()

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_protected_civitai_download_requires_token_before_podman(
        self, require_rootless
    ):
        artifact = Artifact(
            identifier="protected-model",
            description="Protected model",
            source=ArtifactSource(
                repository="civitai.com/models/123",
                revision="456",
                path="model.safetensors",
                provider="civitai",
                model_id=123,
                model_version_id=456,
                requires_auth=True,
            ),
            destination="checkpoints/model.safetensors",
            size=1,
            sha256="b" * 64,
            license=LicenseInfo(
                "LicenseRef-Test",
                "verified",
                "https://example.invalid/license",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "rocmplete.bundles.os.environ",
                {},
                clear=True,
            ):
                with self.assertRaisesRegex(
                    LauncherError, "CIVITAI_TOKEN is required"
                ):
                    install_artifacts(
                        (artifact,),
                        Path(directory),
                        "unused-image",
                    )
        require_rootless.assert_not_called()

    def test_workflow_artifact_uses_imported_workflow_directory(self):
        artifact = fake_artifact()
        artifact = Artifact(
            identifier=artifact.identifier,
            description=artifact.description,
            source=artifact.source,
            destination="krea/workflow.json",
            size=artifact.size,
            sha256=artifact.sha256,
            license=artifact.license,
            target="workflows",
        )
        self.assertEqual(
            artifact_path(Path("/data"), artifact),
            Path(
                "/data/apps/comfyui/user/default/workflows/imported/"
                "krea/workflow.json"
            ),
        )

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_unverified_missing_artifact_is_rejected_before_podman(
        self, require_rootless
    ):
        artifact = fake_artifact(status="unverified")
        catalog, bundle = fake_catalog(artifact)
        with tempfile.TemporaryDirectory() as directory:
            statuses = inspect_bundle(catalog, bundle, Path(directory))
            self.assertEqual(len(missing_unverified(statuses)), 1)
            with self.assertRaisesRegex(LauncherError, "acknowledgment"):
                install_bundle(
                    catalog,
                    bundle,
                    Path(directory),
                    "image",
                    acknowledge_license_risk=False,
                )
        require_rootless.assert_not_called()

    def test_human_size_uses_binary_units(self):
        self.assertEqual(human_size(1024**3), "1.00 GiB")

    @patch("rocmplete.bundles.podman.require_rootless")
    def test_artifact_install_copies_exact_local_mirror_file(
        self, require_rootless
    ):
        contents = b"local mirror model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            mirror_root = root / "old-data"
            source = mirror_root / "somewhere" / "model.bin"
            source.parent.mkdir(parents=True)
            source.write_bytes(contents)
            mirror = LocalMirror(mirror_root)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    install_artifacts(
                        (artifact,),
                        data_dir,
                        "unused-image",
                        local_mirror=mirror,
                    ),
                    0,
                )
            self.assertEqual(artifact_path(data_dir, artifact).read_bytes(), contents)
            self.assertEqual(source.read_bytes(), contents)
            self.assertFalse(
                artifact_staging_root(data_dir, artifact).exists()
            )
        require_rootless.assert_not_called()

    def test_local_mirror_rejects_wrong_hash_and_escaping_symlink(self):
        contents = b"expected"
        digest = hashlib.sha256(contents).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror_root = root / "mirror"
            mirror_root.mkdir()
            (mirror_root / "model.bin").write_bytes(b"not-right")
            external = root / digest
            external.write_bytes(contents)
            (mirror_root / digest).symlink_to(external)
            mirror = LocalMirror(mirror_root)
            with redirect_stdout(io.StringIO()):
                self.assertIsNone(
                    mirror.find(
                        (digest, "model.bin"),
                        len(contents),
                        digest,
                    )
                )

    def test_local_mirror_refuses_overlap_with_active_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mirror_root = root / "old-data"
            mirror_root.mkdir()
            mirror = LocalMirror(mirror_root)
            with self.assertRaisesRegex(LauncherError, "must not overlap"):
                mirror.validate_destination(mirror_root / "new-data")

    def test_local_mirror_move_refuses_symlinked_staging_parent(self):
        contents = b"model"
        artifact = fake_artifact(contents)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            external = root / "external"
            external.mkdir()
            (data_dir / "staging").symlink_to(
                external, target_is_directory=True
            )
            mirror_root = root / "old-data"
            source = mirror_root / "model.bin"
            source.parent.mkdir()
            source.write_bytes(contents)
            mirror = LocalMirror(mirror_root, move=True)
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    LauncherError, "symlinked staging path component"
                ):
                    install_artifacts(
                        (artifact,),
                        data_dir,
                        "unused-image",
                        local_mirror=mirror,
                    )
            self.assertEqual(source.read_bytes(), contents)
            self.assertEqual(tuple(external.iterdir()), ())

if __name__ == "__main__":
    unittest.main()
