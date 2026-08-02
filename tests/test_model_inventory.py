import os
import tempfile
import unittest
from pathlib import Path

from rocmplete.catalog import (
    Artifact,
    ArtifactSource,
    Bundle,
    Catalog,
    LicenseInfo,
    LlamaPreset,
)
from rocmplete.errors import LauncherError
from rocmplete.content_verification import VerificationStore
from rocmplete.layout import StorageLayout
from rocmplete.model_inventory import llama_models


def _artifact(identifier, destination, size):
    return Artifact(
        identifier=identifier,
        description=identifier,
        source=ArtifactSource(
            repository="owner/repository",
            revision="0" * 40,
            path=Path(destination).name,
        ),
        destination=destination,
        size=size,
        sha256="0" * 64,
        license=LicenseInfo(
            spdx="MIT",
            status="verified",
            url="https://example.invalid/license",
        ),
        target="llama-models",
    )


def _catalog():
    first = _artifact(
        "known-first",
        "known/model-00001-of-00002.gguf",
        1,
    )
    second = _artifact(
        "known-second",
        "known/model-00002-of-00002.gguf",
        2,
    )
    return Catalog(
        agreements={},
        artifacts={first.identifier: first, second.identifier: second},
        bundles={
            "known-bundle": Bundle(
                identifier="known-bundle",
                description="Known split model",
                application="llama-cpp",
                artifacts=(first.identifier, second.identifier),
            )
        },
        workflow_packs={},
        benchmarks={},
        llama_presets={
            "known-preset": LlamaPreset(
                identifier="known-preset",
                bundle="known-bundle",
                artifact=first.identifier,
                default_context=4096,
            )
        },
    )


class LlamaModelInventoryTests(unittest.TestCase):
    def test_lists_catalog_loose_and_external_split_models(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            root = StorageLayout(data_dir).llama_models
            (root / "known").mkdir(parents=True)
            (root / "known/model-00001-of-00002.gguf").write_bytes(b"a")
            (root / "known/model-00002-of-00002.gguf").write_bytes(b"bc")
            store = VerificationStore.load(data_dir)
            catalog = _catalog()
            for artifact in catalog.artifacts.values():
                path = root / artifact.destination
                store.record(path, artifact.size, artifact.sha256)
            store.save()
            (root / "imported").mkdir()
            loose = root / "imported/manual.gguf"
            loose.write_bytes(b"loose")

            external = Path(directory) / "external"
            external.mkdir()
            shard = external / "other-00001-of-00002.gguf"
            shard.write_bytes(b"part")

            models = llama_models(catalog, data_dir, (external,))

        self.assertEqual(len(models), 3)
        known = next(item for item in models if item.presets)
        self.assertEqual(known.presets, ("known-preset",))
        self.assertEqual(known.state, "ready")
        self.assertEqual(known.size, 3)

        manual = next(item for item in models if item.path == loose)
        self.assertEqual(manual.state, "ready")
        self.assertEqual(manual.source, "local")

        split = next(item for item in models if item.path == shard)
        self.assertEqual(split.state, "partial")
        self.assertEqual(split.shard_count, 1)
        self.assertEqual(split.expected_shards, 2)

    def test_catalog_size_mismatch_is_not_reported_as_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            root = StorageLayout(data_dir).llama_models
            (root / "known").mkdir(parents=True)
            (root / "known/model-00001-of-00002.gguf").write_bytes(b"wrong")
            (root / "known/model-00002-of-00002.gguf").write_bytes(b"bc")

            models = llama_models(_catalog(), data_dir)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].state, "size-mismatch")

    def test_groups_a_complete_loose_split_model(self):
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "external"
            external.mkdir()
            first = external / "loose-00001-of-00002.gguf"
            first.write_bytes(b"a")
            (external / "loose-00002-of-00002.gguf").write_bytes(b"bc")

            models = llama_models(
                _catalog(),
                Path(directory) / "data",
                (external,),
            )

        self.assertEqual(len(models), 2)
        loose = next(item for item in models if item.path == first)
        self.assertEqual(loose.state, "ready")
        self.assertEqual(loose.size, 3)
        self.assertEqual(loose.shard_count, 2)

    def test_catalog_symlink_is_not_reported_as_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            root = StorageLayout(data_dir).llama_models
            (root / "known").mkdir(parents=True)
            target = Path(directory) / "target.gguf"
            target.write_bytes(b"a")
            os.symlink(
                str(target),
                str(root / "known/model-00001-of-00002.gguf"),
            )
            (root / "known/model-00002-of-00002.gguf").write_bytes(b"bc")

            models = llama_models(_catalog(), data_dir)

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].state, "user-file")

    def test_does_not_create_a_missing_model_root(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "missing"

            models = llama_models(_catalog(), data_dir)

            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].state, "missing")
            self.assertEqual(models[0].size, 3)
            self.assertEqual(models[0].presets, ("known-preset",))
            self.assertFalse(data_dir.exists())

    def test_recursive_scan_does_not_follow_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            root = StorageLayout(data_dir).llama_models
            root.mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            (outside / "hidden.gguf").write_bytes(b"model")
            os.symlink(str(outside), str(root / "linked"))

            models = llama_models(_catalog(), data_dir)

            self.assertEqual(len(models), 1)
            self.assertEqual(models[0].state, "missing")
            self.assertNotIn("hidden.gguf", str(models[0].path))

    def test_rejects_missing_explicit_scan_path(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaisesRegex(
                LauncherError, "model scan path does not exist"
            ):
                llama_models(_catalog(), Path(directory) / "data", (missing,))


if __name__ == "__main__":
    unittest.main()
