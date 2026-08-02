import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocmplete.catalog import load_catalog, load_content_packs
from rocmplete.errors import LauncherError
from rocmplete.remote_import import (
    IMPORT_KINDS,
    automatic_file,
    automatic_kind,
    build_import_plan,
    candidate_kinds,
    compatible_kinds,
    discover_remote,
    remote_provider,
    save_pack,
)
from rocmplete import remote_import


class RemoteImportTests(unittest.TestCase):
    @patch("rocmplete.remote_import.urllib.request.urlopen")
    def test_metadata_token_is_not_forwarded_across_redirects(self, urlopen):
        urlopen.return_value = io.BytesIO(b"{}")
        remote_import._request_json(
            "https://huggingface.co/api/models/owner/repo",
            "secret",
            "Hugging Face",
        )
        request = urlopen.call_args.args[0]
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(
            request.unredirected_hdrs["Authorization"],
            "Bearer secret",
        )

    def test_provider_allowlist_accepts_both_civitai_domains(self):
        self.assertEqual(
            remote_provider("https://civitai.com/models/1"), "civitai"
        )
        self.assertEqual(
            remote_provider("https://civitai.red/models/1"), "civitai"
        )
        self.assertEqual(
            remote_provider("https://huggingface.co/owner/model"),
            "huggingface",
        )
        with self.assertRaisesRegex(LauncherError, "unsupported import host"):
            remote_provider("https://example.invalid/model")

    @patch("rocmplete.remote_import._request_json")
    def test_civitai_red_lora_becomes_a_valid_pinned_pack(self, request):
        request.side_effect = (
            {
                "id": 456,
                "modelId": 123,
                "name": "v1",
                "availability": "Public",
                "files": [
                    {
                        "id": 789,
                        "name": "example.safetensors",
                        "sizeKB": 2.0,
                        "primary": True,
                        "downloadUrl": (
                            "https://civitai.red/api/download/models/456"
                        ),
                        "hashes": {"SHA256": "A" * 64},
                    }
                ],
            },
            {"id": 123, "name": "Example LoRA", "type": "LORA"},
        )
        discovery = discover_remote(
            "https://civitai.red/models/123/example?modelVersionId=456",
            civitai_token="secret",
        )
        selected = automatic_file(discovery)
        self.assertIsNotNone(selected)
        kind = automatic_kind(discovery, selected)
        self.assertEqual(kind.identifier, "comfyui:lora")
        plan = build_import_plan(discovery, selected, kind)
        self.assertEqual(plan.pack["schema_version"], 2)
        artifact = plan.pack["artifacts"][plan.artifact_identifier]
        self.assertEqual(artifact["size"], 2048)
        self.assertEqual(artifact["sha256"], "a" * 64)
        self.assertEqual(artifact["source"]["host"], "civitai.red")
        self.assertTrue(artifact["source"]["requires_auth"])
        self.assertEqual(
            artifact["destination"],
            "loras/imported/example.safetensors",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "import.json"
            path.write_text(json.dumps(plan.pack))
            catalog, selected_bundles = load_content_packs(
                load_catalog(), (path,)
            )
        self.assertEqual(selected_bundles, (plan.bundle_identifier,))
        loaded = catalog.artifact(plan.artifact_identifier)
        self.assertEqual(loaded.source.provider_host, "civitai.red")
        self.assertEqual(
            loaded.source.download_url,
            "https://civitai.red/api/download/models/456",
        )

    @patch("rocmplete.remote_import._request_json")
    def test_huggingface_file_resolves_to_full_commit_and_lfs_hash(
        self, request
    ):
        request.return_value = {
            "sha": "b" * 40,
            "cardData": {"license": "apache-2.0"},
            "siblings": [
                {
                    "rfilename": "models/example.gguf",
                    "lfs": {"size": 4096, "sha256": "c" * 64},
                }
            ],
        }
        discovery = discover_remote(
            "https://huggingface.co/owner/repo/blob/main/models/example.gguf"
        )
        selected = automatic_file(discovery)
        kind = automatic_kind(discovery, selected)
        plan = build_import_plan(discovery, selected, kind)
        artifact = plan.pack["artifacts"][plan.artifact_identifier]
        self.assertEqual(artifact["source"]["revision"], "b" * 40)
        self.assertEqual(artifact["source"]["path"], "models/example.gguf")
        self.assertEqual(artifact["target"], "llama-models")
        self.assertEqual(
            plan.pack["bundles"][plan.bundle_identifier]["groups"],
            ["all", "llama"],
        )

    @patch("rocmplete.remote_import._request_json")
    def test_huggingface_non_lfs_file_is_not_silently_imported(self, request):
        request.return_value = {
            "sha": "b" * 40,
            "siblings": [{"rfilename": "workflow.json", "size": 123}],
        }
        with self.assertRaisesRegex(LauncherError, "not an LFS object"):
            discover_remote(
                "https://huggingface.co/owner/repo/blob/main/workflow.json"
            )

    @patch("rocmplete.remote_import._request_json")
    def test_civitai_explicit_version_cannot_override_url_version(
        self, request
    ):
        with self.assertRaisesRegex(LauncherError, "different versions"):
            discover_remote(
                "https://civitai.com/models/123?modelVersionId=456",
                version_id=457,
            )
        request.assert_not_called()

    @patch("rocmplete.remote_import._request_json")
    def test_checkpoint_requires_explicit_destination_kind(self, request):
        request.side_effect = (
            {
                "id": 456,
                "modelId": 123,
                "files": [
                    {
                        "id": 789,
                        "name": "model.safetensors",
                        "sizeKB": 1,
                        "primary": True,
                        "downloadUrl": (
                            "https://civitai.com/api/download/models/456"
                        ),
                        "hashes": {"SHA256": "d" * 64},
                    }
                ],
            },
            {"id": 123, "name": "Model", "type": "Checkpoint"},
        )
        discovery = discover_remote(
            "https://civitai.com/models/123?modelVersionId=456"
        )
        selected = automatic_file(discovery)
        self.assertIsNone(automatic_kind(discovery, selected))
        identifiers = tuple(
            item.identifier for item in candidate_kinds(discovery, selected)
        )
        self.assertEqual(
            identifiers,
            (
                "comfyui:checkpoint",
                "comfyui:diffusion-model",
            ),
        )
        checkpoint = build_import_plan(
            discovery,
            selected,
            IMPORT_KINDS["comfyui:checkpoint"],
        )
        diffusion_model = build_import_plan(
            discovery,
            selected,
            IMPORT_KINDS["comfyui:diffusion-model"],
        )
        self.assertNotEqual(
            checkpoint.bundle_identifier,
            diffusion_model.bundle_identifier,
        )

    def test_civitai_metadata_narrows_weight_destinations(self):
        file = remote_import.RemoteFile(
            identifier="1",
            name="model.safetensors",
            size=1024,
            sha256="d" * 64,
        )
        expected = {
            "LORA": ("comfyui:lora",),
            "LoCon": ("comfyui:lora",),
            "DoRA": ("comfyui:lora",),
            "VAE": ("comfyui:vae",),
            "Controlnet": ("comfyui:controlnet",),
            "Upscaler": ("comfyui:upscaler",),
        }
        for model_type, identifiers in expected.items():
            with self.subTest(model_type=model_type):
                discovery = remote_import.RemoteDiscovery(
                    provider="civitai",
                    source_url="https://civitai.com/models/1",
                    title="Model",
                    repository="civitai.com/models/1",
                    revision="2",
                    files=(file,),
                    declared_license="unknown",
                    model_type=model_type,
                )
                self.assertEqual(
                    tuple(
                        kind.identifier
                        for kind in candidate_kinds(discovery, file)
                    ),
                    identifiers,
                )
                self.assertEqual(
                    automatic_kind(discovery, file).identifier,
                    identifiers[0],
                )

    def test_unsupported_civitai_type_has_no_candidate_destination(self):
        file = remote_import.RemoteFile(
            identifier="1",
            name="embedding.safetensors",
            size=1024,
            sha256="d" * 64,
        )
        discovery = remote_import.RemoteDiscovery(
            provider="civitai",
            source_url="https://civitai.com/models/1",
            title="Embedding",
            repository="civitai.com/models/1",
            revision="2",
            files=(file,),
            declared_license="unknown",
            model_type="TextualInversion",
        )
        self.assertEqual(candidate_kinds(discovery, file), ())
        self.assertIsNone(automatic_kind(discovery, file))
        self.assertGreater(len(compatible_kinds(file)), 1)

    def test_huggingface_pack_identity_retains_destination_kind(self):
        discovery = remote_import.RemoteDiscovery(
            provider="huggingface",
            source_url="https://huggingface.co/very-long-owner/repository",
            title="very-long-owner/repository",
            repository=(
                "very-long-owner/"
                "repository-name-that-consumes-the-slug-budget"
            ),
            revision="b" * 40,
            files=(),
            declared_license="unknown",
        )
        file = remote_import.RemoteFile(
            identifier="model.safetensors",
            name="model.safetensors",
            size=1024,
            sha256="c" * 64,
        )
        checkpoint = build_import_plan(
            discovery,
            file,
            IMPORT_KINDS["comfyui:checkpoint"],
        )
        diffusion_model = build_import_plan(
            discovery,
            file,
            IMPORT_KINDS["comfyui:diffusion-model"],
        )
        self.assertNotEqual(
            checkpoint.bundle_identifier,
            diffusion_model.bundle_identifier,
        )
        self.assertTrue(
            checkpoint.bundle_identifier.endswith("comfyui-checkpoint")
        )
        self.assertTrue(
            diffusion_model.bundle_identifier.endswith(
                "comfyui-diffusion-model"
            )
        )

    @patch("rocmplete.remote_import._request_json")
    def test_save_pack_is_atomic_idempotent_and_refuses_differences(
        self, request
    ):
        request.side_effect = (
            {
                "id": 456,
                "modelId": 123,
                "files": [
                    {
                        "id": 789,
                        "name": "model.safetensors",
                        "sizeKB": 1,
                        "primary": True,
                        "downloadUrl": (
                            "https://civitai.com/api/download/models/456"
                        ),
                        "hashes": {"SHA256": "d" * 64},
                    }
                ],
            },
            {"id": 123, "name": "Model", "type": "Checkpoint"},
        )
        discovery = discover_remote(
            "https://civitai.com/models/123?modelVersionId=456"
        )
        plan = build_import_plan(
            discovery,
            automatic_file(discovery),
            IMPORT_KINDS["comfyui:checkpoint"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imports" / "model.json"
            self.assertTrue(save_pack(path, plan))
            self.assertFalse(save_pack(path, plan))
            path.write_text("{}")
            with self.assertRaisesRegex(LauncherError, "different content"):
                save_pack(path, plan)


if __name__ == "__main__":
    unittest.main()
