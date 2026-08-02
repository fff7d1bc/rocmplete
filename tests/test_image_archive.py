import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocmplete.config import (
    APPLICATIONS,
    CONTENT_TOOLS_IMAGE,
    ROCM_BASE_IMAGE,
    ROCM_RUNTIME_IMAGE,
)
from rocmplete.errors import LauncherError
from rocmplete.image_archive import (
    inspect_archive,
    load_command,
    managed_image_references,
    save_command,
    selected_image_references,
    validate_managed_archive,
)


def _add_bytes(archive, name, contents):
    member = tarfile.TarInfo(name)
    member.size = len(contents)
    archive.addfile(member, io.BytesIO(contents))


def write_image_archive(path, references, architecture="amd64"):
    manifest = []
    image_ids = {}
    configs = {}
    for index, reference in enumerate(references):
        contents = json.dumps(
            {
                "architecture": architecture,
                "os": "linux",
                "config": {"Labels": {"test.reference": reference}},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest = hashlib.sha256(contents).hexdigest()
        config_name = "{}.json".format(digest)
        configs[config_name] = contents
        image_ids[reference] = "sha256:{}".format(digest)
        manifest.append(
            {
                "Config": config_name,
                "RepoTags": [reference],
                "Layers": [],
            }
        )
    with tarfile.open(str(path), "w") as archive:
        for name, contents in configs.items():
            _add_bytes(archive, name, contents)
        _add_bytes(
            archive,
            "manifest.json",
            json.dumps(manifest).encode(),
        )
    return image_ids


class ImageArchiveTests(unittest.TestCase):
    def test_managed_selections_include_only_required_base(self):
        self.assertEqual(
            selected_image_references("base"),
            (CONTENT_TOOLS_IMAGE, ROCM_RUNTIME_IMAGE, ROCM_BASE_IMAGE),
        )
        self.assertEqual(
            selected_image_references("comfyui"),
            (
                CONTENT_TOOLS_IMAGE,
                ROCM_RUNTIME_IMAGE,
                ROCM_BASE_IMAGE,
                APPLICATIONS["comfyui"].image,
            ),
        )
        self.assertEqual(
            selected_image_references("llama-cpp"),
            (
                CONTENT_TOOLS_IMAGE,
                ROCM_RUNTIME_IMAGE,
                APPLICATIONS["llama-cpp"].image,
            ),
        )
        self.assertEqual(
            selected_image_references("all"),
            managed_image_references(),
        )

    def test_save_and_load_commands_are_explicit(self):
        images = selected_image_references("comfyui")
        command = save_command(images, Path("/backup/images.tar"))
        self.assertEqual(command[:6], (
            "podman",
            "save",
            "--format",
            "docker-archive",
            "--output",
            "/backup/images.tar",
        ))
        self.assertIn("--multi-image-archive", command)
        self.assertEqual(command[-len(images):], images)
        self.assertEqual(
            load_command(Path("/backup/images.tar")),
            ("podman", "load", "--input", "/backup/images.tar"),
        )

    def test_inspection_uses_config_digest_as_image_identity(self):
        references = selected_image_references("comfyui")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.tar"
            expected_ids = write_image_archive(path, references)
            archive = inspect_archive(path)

        self.assertEqual(
            {item.reference: item.image_id for item in archive.images},
            expected_ids,
        )
        self.assertTrue(
            all(item.architecture == "amd64" for item in archive.images)
        )
        validate_managed_archive(archive, references)

    def test_archive_requires_base_and_current_managed_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing_base = root / "missing-base.tar"
            write_image_archive(
                missing_base,
                (
                    CONTENT_TOOLS_IMAGE,
                    ROCM_RUNTIME_IMAGE,
                    APPLICATIONS["comfyui"].image,
                ),
            )
            with self.assertRaisesRegex(LauncherError, "missing.*base"):
                validate_managed_archive(inspect_archive(missing_base))

            unmanaged = root / "unmanaged.tar"
            write_image_archive(
                unmanaged,
                (
                    CONTENT_TOOLS_IMAGE,
                    ROCM_RUNTIME_IMAGE,
                    ROCM_BASE_IMAGE,
                    "localhost/other:image",
                ),
            )
            with self.assertRaisesRegex(LauncherError, "unmanaged or obsolete"):
                validate_managed_archive(inspect_archive(unmanaged))

    def test_archive_requires_runtime_for_every_application(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-runtime.tar"
            write_image_archive(
                path,
                (CONTENT_TOOLS_IMAGE, APPLICATIONS["llama-cpp"].image),
            )
            with self.assertRaisesRegex(LauncherError, "missing.*runtime"):
                validate_managed_archive(inspect_archive(path))

    @patch("rocmplete.image_archive.platform.machine", return_value="x86_64")
    def test_archive_rejects_foreign_architecture(self, machine):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.tar"
            write_image_archive(path, (CONTENT_TOOLS_IMAGE,), "arm64")
            with self.assertRaisesRegex(LauncherError, "not host amd64"):
                validate_managed_archive(inspect_archive(path))

    def test_archive_rejects_duplicate_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.tar"
            with tarfile.open(str(path), "w") as archive:
                _add_bytes(archive, "manifest.json", b"[]")
                _add_bytes(archive, "manifest.json", b"[]")
            with self.assertRaisesRegex(LauncherError, "repeats members"):
                inspect_archive(path)

    def test_archive_accepts_only_resolved_internal_layer_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "linked-layer.tar"
            reference = CONTENT_TOOLS_IMAGE
            config = json.dumps(
                {"architecture": "amd64", "os": "linux"}
            ).encode()
            digest = hashlib.sha256(config).hexdigest()
            manifest = [
                {
                    "Config": "{}.json".format(digest),
                    "RepoTags": [reference],
                    "Layers": ["legacy/layer.tar"],
                }
            ]
            layer = tarfile.TarInfo("legacy/layer.tar")
            layer.type = tarfile.SYMTYPE
            layer.linkname = "../blob.tar"
            with tarfile.open(str(path), "w") as archive:
                _add_bytes(archive, "blob.tar", b"layer")
                _add_bytes(archive, "{}.json".format(digest), config)
                _add_bytes(
                    archive, "manifest.json", json.dumps(manifest).encode()
                )
                archive.addfile(layer)

            validate_managed_archive(inspect_archive(path))

            unsafe = root / "unsafe-link.tar"
            layer.linkname = "../../outside.tar"
            with tarfile.open(str(unsafe), "w") as archive:
                _add_bytes(archive, "blob.tar", b"layer")
                _add_bytes(archive, "{}.json".format(digest), config)
                _add_bytes(
                    archive, "manifest.json", json.dumps(manifest).encode()
                )
                archive.addfile(layer)
            with self.assertRaisesRegex(
                LauncherError, "unsupported member type"
            ):
                inspect_archive(unsafe)


if __name__ == "__main__":
    unittest.main()
