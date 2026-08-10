import json
import tempfile
import unittest
from pathlib import Path

from rocmplete.catalog import (
    DEFAULT_CATALOG_PATH,
    load_catalog,
    load_content_packs,
)
from rocmplete.errors import LauncherError


class CatalogTests(unittest.TestCase):
    def _catalog_copy(self, directory, raw):
        root = Path(directory)
        path = root / "catalog.json"
        path.write_text(json.dumps(raw))
        return path

    def _content_pack(self, artifact_identifier="private-model"):
        contents = b"private model"
        return {
            "schema_version": 2,
            "artifacts": {
                artifact_identifier: {
                    "description": "Private model",
                    "source": {
                        "repository": "owner/private-repository",
                        "revision": "a" * 40,
                        "path": "models/private-model.bin",
                    },
                    "destination": "checkpoints/private-model.bin",
                    "size": len(contents),
                    "sha256": (
                        "cf50cbcd8b2e1d9c20c8e0e2c5f052fe14733b68"
                        "45f5709b3b72c4f6607f1177"
                    ),
                    "license": {
                        "spdx": "LicenseRef-Private",
                        "status": "verified",
                        "url": "https://example.invalid/private-license",
                    },
                }
            },
            "bundles": {
                "private-bootstrap": {
                    "description": "Private bootstrap content",
                    "application": "comfyui",
                    "artifacts": [artifact_identifier],
                    "groups": ["all", "comfyui"],
                }
            },
        }

    def _archive_collection(self):
        contents = b"workflow"
        return {
            "source": {
                "provider": "civitai",
                "model_id": 123,
                "model_version_id": 456,
                "filename": "workflow.zip",
            },
            "archive": {"max_size": 4096},
            "license": {
                "spdx": "MIT",
                "status": "verified",
                "url": "https://example.invalid/license",
            },
            "members": {
                "fixture-archive-workflow": {
                    "description": "Archive workflow fixture",
                    "member": "pack/workflow.json",
                    "destination": "fixtures/archive-workflow.json",
                    "target": "workflows",
                    "size": len(contents),
                    "sha256": (
                        "da7f739f627198465eeab537a6f7a435"
                        "dc4a0c332f9e4a8462293eb3f4ab7ee0"
                    ),
                }
            },
        }

    def test_default_catalog_contains_all_application_bundles(self):
        catalog = load_catalog()
        self.assertEqual(len(catalog.bundles), 50)
        self.assertEqual(len(catalog.artifacts), 65)
        self.assertEqual(len(catalog.benchmarks), 28)
        self.assertEqual(len(catalog.llama_presets), 15)
        self.assertFalse(
            [
                artifact.identifier
                for artifact in catalog.artifacts.values()
                if artifact.source.provider == "civitai"
            ],
            "the built-in catalog must not depend on mutable Civitai files",
        )
        krea = catalog.bundle("krea-2-turbo-fp8-base")
        self.assertIn("krea", krea.groups)
        self.assertNotIn("comfyui", krea.groups)
        self.assertEqual(
            [item.identifier for item in catalog.bundle_agreements(krea)],
            ["krea-2-community-license"],
        )
        llama = catalog.llama_preset("qwen3-0.6b-q8-0")
        self.assertEqual(llama.bundle, "llama-qwen3-0.6b-q8-0")
        self.assertEqual(llama.artifact, "qwen3-0.6b-q8-gguf")
        self.assertEqual(llama.default_context, 4096)
        self.assertEqual(
            catalog.artifact(llama.artifact).target, "llama-models"
        )
        dwarfstar = catalog.bundle(
            "dwarfstar-deepseek-v4-flash-0731-q2-imatrix"
        )
        self.assertEqual(dwarfstar.application, "dwarfstar")
        self.assertEqual(dwarfstar.groups, ("all", "dwarfstar"))
        dwarfstar_artifact = catalog.artifact(dwarfstar.artifacts[0])
        self.assertEqual(dwarfstar_artifact.target, "dwarfstar-models")
        self.assertEqual(dwarfstar_artifact.size, 86720111488)
        self.assertEqual(
            dwarfstar_artifact.sha256,
            "ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c"
            "2ac6e17fdb6261c0",
        )
        self.assertEqual(dwarfstar_artifact.license.spdx, "MIT")
        self.assertEqual(dwarfstar_artifact.license.status, "verified")
        qwen35 = catalog.llama_preset(
            "qwen3.6-35b-a3b-ud-q8-k-xl"
        )
        qwen35_artifact = catalog.artifact(qwen35.artifact)
        self.assertEqual(qwen35.default_context, 262144)
        self.assertEqual(qwen35.speculative_type, "")
        self.assertEqual(qwen35.draft_tokens, 0)
        self.assertTrue(qwen35.agent_tools)
        self.assertTrue(qwen35.reasoning_effort_budget)
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(qwen35.bundle)),
            38451182560,
        )
        self.assertEqual(
            qwen35_artifact.sha256,
            "b762215c5f507f4865df4ac3d1afa803828afa41"
            "e05ecac3fac431a67bbd88e8",
        )
        qwen35_mtp = catalog.llama_preset(
            "qwen3.6-35b-a3b-mtp-ud-q8-k-xl"
        )
        qwen35_mtp_artifact = catalog.artifact(qwen35_mtp.artifact)
        self.assertEqual(qwen35_mtp.default_context, 262144)
        self.assertEqual(qwen35_mtp.speculative_type, "draft-mtp")
        self.assertEqual(qwen35_mtp.draft_tokens, 3)
        self.assertTrue(qwen35_mtp.agent_tools)
        self.assertTrue(qwen35_mtp.reasoning_effort_budget)
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(qwen35_mtp.bundle)),
            39099447584,
        )
        self.assertEqual(
            qwen35_mtp_artifact.sha256,
            "6c6b816537abad90b250a0972b345466028d861dd"
            "fe316d5f0de31ca6440f781",
        )
        assistant = catalog.llama_preset("qwen3.6-27b-q8-0")
        assistant_artifact = catalog.artifact(assistant.artifact)
        self.assertEqual(
            assistant.bundle, "llama-qwen3.6-27b-q8-0"
        )
        self.assertEqual(assistant.default_context, 262144)
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(assistant.bundle)),
            28595763424,
        )
        self.assertEqual(
            assistant_artifact.sha256,
            "f93f517f38e696d35a1a7df2c0e3155a64f4c4dc"
            "d662107a146ae263f7fb14ce",
        )
        assistant_mtp = catalog.llama_preset("qwen3.6-27b-mtp-q8-0")
        assistant_mtp_artifact = catalog.artifact(assistant_mtp.artifact)
        self.assertEqual(
            assistant_mtp.bundle, "llama-qwen3.6-27b-mtp-q8-0"
        )
        self.assertEqual(assistant_mtp.default_context, 262144)
        self.assertEqual(assistant_mtp.speculative_type, "draft-mtp")
        self.assertEqual(assistant_mtp.draft_tokens, 2)
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(assistant_mtp.bundle)),
            29047084160,
        )
        self.assertEqual(
            assistant_mtp_artifact.sha256,
            "9408dcb356cc061a05c139e5647cbde0698ff980c"
            "6a69f7fc214e9989f86cfa8",
        )
        coding_candidates = {
            "ornith-1.0-35b-q8-0": (
                "deepreinforce-ai/Ornith-1.0-35B-GGUF",
                36903138880,
                "cbc992bca07901c1a51f33e65e6fc5d687de179c852a772dfd15e4c3261dbf5c",
                "MIT",
            ),
            "kat-coder-v2.5-dev-q8-0": (
                "bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF",
                36914690464,
                "5fa510f44779b0e3d38a6678985f417a1c65e3000405ca5d6dcf7fd065e47a15",
                "Apache-2.0",
            ),
        }
        for identifier, expected in coding_candidates.items():
            with self.subTest(preset=identifier):
                preset = catalog.llama_preset(identifier)
                artifact = catalog.artifact(preset.artifact)
                self.assertEqual(preset.default_context, 262144)
                self.assertTrue(preset.jinja)
                self.assertTrue(preset.agent_tools)
                self.assertTrue(preset.reasoning_effort_budget)
                self.assertEqual(artifact.source.repository, expected[0])
                self.assertEqual(artifact.size, expected[1])
                self.assertEqual(artifact.sha256, expected[2])
                self.assertEqual(artifact.license.spdx, expected[3])
                self.assertEqual(artifact.license.status, "verified")
        gemma = catalog.llama_preset("gemma4-31b-it-q8-0-mtp")
        self.assertEqual(gemma.default_context, 262144)
        self.assertEqual(gemma.speculative_type, "draft-mtp")
        self.assertEqual(gemma.draft_tokens, 4)
        self.assertTrue(gemma.jinja)
        self.assertTrue(gemma.agent_tools)
        self.assertTrue(gemma.reasoning_effort_budget)
        self.assertEqual(
            gemma.draft_artifact, "gemma4-31b-it-mtp-q8-gguf"
        )
        gemma_artifact = catalog.artifact(gemma.artifact)
        self.assertEqual(
            gemma_artifact.source.repository,
            "ggml-org/gemma-4-31B-it-GGUF",
        )
        self.assertEqual(
            gemma_artifact.sha256,
            "fcd52cebacb165a98df5abe6fb70dbf076835f4a"
            "06e064ffb33dd739b8835c9c",
        )
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(gemma.bundle)),
            33150364000,
        )
        muse = catalog.llama_preset(
            "muse-glimmer-30b-kquant-dynamic-dflash"
        )
        muse_base = catalog.llama_preset(
            "muse-glimmer-30b-kquant-dynamic"
        )
        muse_256k = catalog.llama_preset(
            "muse-glimmer-30b-kquant-dynamic-dflash-256k"
        )
        muse_artifact = catalog.artifact(muse.artifact)
        muse_draft = catalog.artifact(muse.draft_artifact)
        self.assertEqual(muse.default_context, 131072)
        self.assertEqual(muse_base.default_context, 131072)
        self.assertEqual(muse_base.speculative_type, "")
        self.assertEqual(muse_base.artifact, muse.artifact)
        self.assertEqual(muse_base.bundle, muse.bundle)
        self.assertEqual(muse.speculative_type, "draft-dflash")
        self.assertEqual(muse.draft_tokens, 15)
        self.assertEqual(muse.context_override_architectures, ())
        self.assertEqual(muse_256k.artifact, muse.artifact)
        self.assertEqual(muse_256k.draft_artifact, muse.draft_artifact)
        self.assertEqual(muse_256k.bundle, muse.bundle)
        self.assertEqual(muse_256k.default_context, 262144)
        self.assertEqual(
            muse_256k.context_override_architectures,
            ("muse-glimmer", "dflash"),
        )
        for preset in (muse_base, muse, muse_256k):
            with self.subTest(preset=preset.identifier):
                self.assertTrue(preset.jinja)
                self.assertTrue(preset.agent_tools)
                self.assertFalse(preset.reasoning_effort_budget)
                self.assertTrue(preset.reasoning_preserve)
        self.assertEqual(
            muse_artifact.source.repository,
            "meta-models/Muse-Glimmer-30B-GGUF",
        )
        self.assertEqual(
            muse_artifact.source.revision,
            "93769bc7ab5ad1e9cd22d857e3138cf5d977ae81",
        )
        self.assertEqual(
            muse_artifact.source.path,
            "muse-glimmer-30B-kquant-dynamic.gguf",
        )
        self.assertEqual(muse_artifact.size, 19653957984)
        self.assertEqual(
            muse_artifact.sha256,
            "513109c8319115f69eb09fb7b118c97c8167d15bc014fd7670d2e30489bf106c",
        )
        self.assertEqual(
            muse_draft.source.repository,
            "meta-models/Muse-Glimmer-30B-GGUF",
        )
        self.assertEqual(muse_draft.size, 1631205312)
        self.assertEqual(
            muse_draft.sha256,
            "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc",
        )
        self.assertEqual(
            catalog.bundle_size(catalog.bundle(muse.bundle)),
            21285163296,
        )
        qwen_presets = (
            llama,
            assistant,
            qwen35,
            qwen35_mtp,
        )
        self.assertTrue(all(preset.jinja for preset in qwen_presets))
        laguna = catalog.llama_preset("laguna-s-2.1-q4-k-m")
        laguna_bundle = catalog.bundle(laguna.bundle)
        laguna_artifact = catalog.artifact(laguna.artifact)
        self.assertEqual(laguna.default_context, 262144)
        self.assertTrue(laguna.jinja)
        self.assertEqual(
            laguna.flash_attention,
            {"strix-halo": "off", "strix-point": "off"},
        )
        self.assertEqual(
            catalog.bundle_size(laguna_bundle), 68248760064
        )
        self.assertEqual(laguna_artifact.license.spdx, "NOASSERTION")
        self.assertEqual(laguna_artifact.license.status, "unverified")
        self.assertEqual(
            [
                item.identifier
                for item in catalog.bundle_agreements(laguna_bundle)
            ],
            ["openmdw-1.1"],
        )
        hy = catalog.llama_preset("hy-mt1.5-7b-q8-0")
        hy_bundle = catalog.bundle(hy.bundle)
        hy_artifact = catalog.artifact(hy.artifact)
        self.assertEqual(hy.default_context, 32768)
        self.assertEqual(catalog.bundle_size(hy_bundle), 7981929344)
        self.assertEqual(
            hy_artifact.license.spdx,
            "LicenseRef-Tencent-HY-Community",
        )
        self.assertEqual(
            [
                item.identifier
                for item in catalog.bundle_agreements(hy_bundle)
            ],
            ["tencent-hy-community-license"],
        )
        translate_gemma = catalog.llama_preset("translategemma-27b-it-q8-0")
        self.assertEqual(translate_gemma.default_context, 4096)
        self.assertEqual(
            translate_gemma.chat_template, "translategemma-manual"
        )
        gemma_bundle = catalog.bundle(translate_gemma.bundle)
        self.assertEqual(
            catalog.bundle_size(gemma_bundle), 28707987936
        )
        self.assertEqual(
            [
                item.identifier
                for item in catalog.bundle_agreements(gemma_bundle)
            ],
            ["gemma-terms"],
        )
        shisa = catalog.llama_preset("shisa-v2.1-llama3.3-70b-q8-0")
        shisa_bundle = catalog.bundle(shisa.bundle)
        shisa_artifact = catalog.artifact(shisa.artifact)
        self.assertEqual(shisa.default_context, 16384)
        self.assertTrue(shisa.jinja)
        self.assertFalse(shisa.agent_tools)
        self.assertEqual(
            catalog.bundle_size(shisa_bundle), 74975055328
        )
        self.assertEqual(
            shisa_artifact.source.repository,
            "mradermacher/shisa-v2.1-llama3.3-70b-GGUF",
        )
        self.assertEqual(
            shisa_artifact.source.revision,
            "25cb86e709e7ecbe19a71452bbc374f1f5a6462b",
        )
        self.assertEqual(
            shisa_artifact.sha256,
            "1e02e2c7273bee1f84bf212a901ba6a4206859592f63afeb760127ecd0eb1ad5",
        )
        self.assertEqual(
            shisa_artifact.license.spdx,
            "LicenseRef-Llama-3.3-Community",
        )
        self.assertEqual(shisa_artifact.license.status, "verified")
        self.assertEqual(
            [
                item.identifier
                for item in catalog.bundle_agreements(shisa_bundle)
            ],
            ["llama-3.3-community-license"],
        )
        template_path = (
            DEFAULT_CATALOG_PATH.parent.parent
            / "applications"
            / "llama-cpp"
            / "chat-templates"
            / "translategemma-manual.jinja"
        )
        template = template_path.read_text()
        self.assertIn('message["content"] | trim', template)
        self.assertNotIn("English", template)
        self.assertNotIn("Japanese", template)

        base = catalog.bundle("qwen-image-2512-fp8-base")
        lightning = catalog.bundle("qwen-image-2512-fp8-lightning")
        self.assertEqual(base.artifacts, lightning.artifacts[:3])
        self.assertEqual(catalog.bundle_size(base), 30069156070)

        edit = catalog.bundle("qwen-image-edit-2511-fp8-base")
        self.assertIn("qwen-2.5-vl-7b-fp8-text-encoder", edit.artifacts)
        self.assertIn("qwen-image-vae", edit.artifacts)

        ltx = catalog.bundle("ltx-2-t2v-19b-fp8-full")
        self.assertEqual(
            [item.identifier for item in catalog.bundle_agreements(ltx)],
            ["ltx-2-community-license", "gemma-terms"],
        )
        camera = catalog.bundle("ltx-2-camera-dolly-left")
        self.assertEqual(camera.workflow, "")
        self.assertIn("ltx-camera", camera.groups)
        self.assertNotIn("ltx", camera.groups)
        hunyuan = catalog.bundle(
            "hunyuan-video-1.5-i2v-720p-fp16"
        )
        self.assertEqual(
            catalog.bundle_agreements(hunyuan)[0].identifier,
            "hunyuan-video-1.5-community-license",
        )
        wan_v2 = catalog.bundle("wan-2.2-t2v-14b-fp8-lightning-v2")
        self.assertIn("wan-2.2-t2v-lightning-v2-high", wan_v2.artifacts)
        self.assertEqual(
            catalog.artifact("wan-2.2-t2v-lightning-v2-high").sha256,
            "78cc2c9b44aca9ded8f69c6619639edca7459308051161253c3aa04ac6169a58",
        )

    def test_qwen_is_verified_and_wan_repack_is_unverified(self):
        catalog = load_catalog()
        qwen = catalog.artifact("qwen-image-2512-fp8-model")
        wan = catalog.artifact("wan-2.2-t2v-high-fp8")
        self.assertEqual(qwen.license.spdx, "Apache-2.0")
        self.assertEqual(qwen.license.status, "verified")
        self.assertEqual(wan.license.spdx, "NOASSERTION")
        self.assertEqual(wan.license.status, "unverified")
        self.assertEqual(wan.license.upstream_license, "Apache-2.0")
        self.assertTrue(wan.license.warning)

    def test_catalog_rejects_removed_model_tree_collection(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["model_trees"] = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(
                LauncherError, "unsupported collections: model_trees"
            ):
                load_catalog(path)

    def test_catalog_rejects_removed_bundle_trees_field(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["bundles"]["qwen-image-2512-fp8-base"]["trees"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(LauncherError, "removed trees field"):
                load_catalog(path)

    def test_duplicate_destination_is_rejected(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        duplicate = dict(raw["artifacts"]["qwen-image-2512-fp8-model"])
        duplicate["description"] = "Duplicate"
        raw["artifacts"]["duplicate-artifact"] = duplicate
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(LauncherError, "share destination"):
                load_catalog(path)

    def test_llama_mtp_draft_count_is_bounded(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["llama_presets"]["gemma4-31b-it-q8-0-mtp"][
            "draft_tokens"
        ] = 9
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "between 1 and 8"
            ):
                load_catalog(path)

    def test_llama_dflash_requires_a_separate_draft_artifact(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"][
            "muse-glimmer-30b-kquant-dynamic-dflash"
        ]
        del preset["draft_artifact"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "draft-dflash requires draft_artifact"
            ):
                load_catalog(path)

    def test_llama_draft_tokens_require_a_speculative_type(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"]["qwen3-0.6b-q8-0"]
        preset["draft_tokens"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "draft_tokens requires speculative_type"
            ):
                load_catalog(path)

    def test_llama_context_override_architectures_are_strict(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"][
            "muse-glimmer-30b-kquant-dynamic-dflash-256k"
        ]
        preset["context_override_architectures"] = [
            "muse-glimmer",
            "dflash.context_length=int:1",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "contain GGUF architecture names"
            ):
                load_catalog(path)

    def test_llama_preset_rejects_obsolete_context_field(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"]["qwen3-0.6b-q8-0"]
        preset["context"] = preset.pop("default_context")
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "unsupported fields: context"
            ):
                load_catalog(path)

    def test_llama_preset_policy_is_strictly_typed(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"]["laguna-s-2.1-q4-k-m"]
        preset["jinja"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(LauncherError, "jinja must be"):
                load_catalog(path)

        preset["jinja"] = True
        preset["agent_tools"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "agent_tools must be"
            ):
                load_catalog(path)

        preset["agent_tools"] = True
        preset["reasoning_effort_budget"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "reasoning_effort_budget must be"
            ):
                load_catalog(path)

        preset["reasoning_effort_budget"] = False
        preset["reasoning_preserve"] = "yes"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "reasoning_preserve must be"
            ):
                load_catalog(path)

        preset["reasoning_preserve"] = False
        preset["default_context"] = 4096
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "requires Jinja and at least 16384"
            ):
                load_catalog(path)

        preset["jinja"] = True
        preset["agent_tools"] = True
        preset["default_context"] = 32768
        preset["flash_attention"] = {"auto": "off"}
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError,
                "profile must be one of rdna4, strix-halo, strix-point",
            ):
                load_catalog(path)

        preset["flash_attention"] = {"strix-halo": "sometimes"}
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "value must be on, off, or auto"
            ):
                load_catalog(path)

        preset["flash_attention"] = {"strix-halo": "off"}
        preset["arbitrary_arguments"] = ["--unsafe"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "unsupported fields: arbitrary_arguments"
            ):
                load_catalog(path)

        del preset["arbitrary_arguments"]
        preset["flash_attention"] = {
            "rdna4": "auto",
            "strix-halo": "off",
            "strix-point": "off",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            loaded = load_catalog(path).llama_preset(
                "laguna-s-2.1-q4-k-m"
            )
        self.assertEqual(
            loaded.flash_attention,
            {
                "rdna4": "auto",
                "strix-halo": "off",
                "strix-point": "off",
            },
        )

        preset["chat_template"] = "some-file"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "chat_template must be"
            ):
                load_catalog(path)

        preset["chat_template"] = "translategemma-manual"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "already enables Jinja"
            ):
                load_catalog(path)

    def test_llama_reasoning_budget_requires_agent_tools(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"]["qwen3-0.6b-q8-0"]
        preset["reasoning_effort_budget"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError,
                "reasoning_effort_budget requires agent_tools",
            ):
                load_catalog(path)

    def test_llama_reasoning_preserve_requires_agent_tools(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        preset = raw["llama_presets"]["qwen3-0.6b-q8-0"]
        preset["reasoning_preserve"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError,
                "reasoning_preserve requires agent_tools",
            ):
                load_catalog(path)

    def test_llama_mtp_draft_must_belong_to_preset_bundle(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["llama_presets"]["gemma4-31b-it-q8-0-mtp"][
            "draft_artifact"
        ] = "qwen3-0.6b-q8-gguf"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "draft artifact is not in bundle"
            ):
                load_catalog(path)

    def test_archive_collection_rejects_unknown_member_fields(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["archive_collections"]["fixture"] = self._archive_collection()
        collection = raw["archive_collections"]["fixture"]
        member = next(iter(collection["members"].values()))
        member["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(
                LauncherError, "unsupported fields: unexpected"
            ):
                load_catalog(path)

    def test_archive_collection_rejects_obsolete_envelope_hash(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["archive_collections"]["fixture"] = self._archive_collection()
        collection = raw["archive_collections"]["fixture"]
        collection["archive"]["sha256"] = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "unsupported fields: sha256"
            ):
                load_catalog(path)

    def test_content_pack_can_reference_builtin_content(self):
        pack = {
            "schema_version": 2,
            "bundles": {
                "private-bootstrap": {
                    "description": "Private bootstrap content",
                    "application": "comfyui",
                    "artifacts": ["qwen-image-vae"],
                    "groups": ["all", "comfyui"],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps(pack))
            merged, selected = load_content_packs(
                load_catalog(), (path,)
            )
        self.assertEqual(selected, ("private-bootstrap",))
        self.assertIn("qwen-image-vae", merged.artifacts)
        self.assertEqual(
            merged.bundle("private-bootstrap").artifacts,
            ("qwen-image-vae",),
        )

    def test_schema_one_content_pack_is_rejected(self):
        pack = self._content_pack()
        pack["schema_version"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "unsupported or invalid content pack schema"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_content_packs_can_reference_each_other(self):
        first = self._content_pack()
        second = {
            "schema_version": 2,
            "bundles": {
                "private-second-machine": {
                    "description": "Second private selection",
                    "application": "comfyui",
                    "artifacts": ["private-model"],
                    "groups": ["all", "comfyui"],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "base.json"
            second_path = root / "second.json"
            first_path.write_text(json.dumps(first))
            second_path.write_text(json.dumps(second))
            merged, selected = load_content_packs(
                load_catalog(), (first_path, second_path)
            )
        self.assertEqual(
            selected,
            ("private-bootstrap", "private-second-machine"),
        )
        self.assertEqual(
            merged.bundle("private-second-machine").artifacts,
            ("private-model",),
        )

    def test_content_pack_definition_cannot_override_builtin_catalog(self):
        pack = self._content_pack("qwen-image-vae")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "conflicts with an existing definition"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_content_pack_destination_collision_is_rejected(self):
        pack = self._content_pack()
        pack["artifacts"]["private-model"]["destination"] = (
            "vae/qwen_image_vae.safetensors"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "share destination"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_content_pack_rejects_non_declarative_collections(self):
        pack = self._content_pack()
        pack["workflows"] = {}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "unsupported collections: workflows"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_content_pack_loads_pinned_civitai_workflow_artifact(self):
        pack = self._content_pack()
        artifact = pack["artifacts"]["private-model"]
        artifact["source"] = {
            "provider": "civitai",
            "host": "civitai.red",
            "model_id": 123,
            "model_version_id": 456,
            "filename": "workflow.json",
            "download_url": "https://civitai.red/api/download/models/456",
            "requires_auth": False,
            "archive": {
                "member": "pack/workflow.json",
                "max_size": 1234,
            },
        }
        artifact["destination"] = "krea/workflow.json"
        artifact["target"] = "workflows"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(pack))
            merged, _ = load_content_packs(load_catalog(), (path,))
        loaded = merged.artifact("private-model")
        self.assertEqual(loaded.source.provider, "civitai")
        self.assertEqual(loaded.source.provider_host, "civitai.red")
        self.assertEqual(
            loaded.source.download_url,
            "https://civitai.red/api/download/models/456",
        )
        self.assertEqual(loaded.source.model_version_id, 456)
        self.assertEqual(
            loaded.source.archive_member, "pack/workflow.json"
        )
        self.assertEqual(loaded.source.archive_max_size, 1234)
        self.assertEqual(loaded.target, "workflows")

    def test_content_pack_rejects_obsolete_archive_size(self):
        pack = self._content_pack()
        artifact = pack["artifacts"]["private-model"]
        artifact["source"] = {
            "provider": "civitai",
            "model_id": 123,
            "model_version_id": 456,
            "filename": "workflow.zip",
            "archive": {
                "member": "pack/workflow.json",
                "max_size": 1234,
                "size": 1000,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "unsupported fields: size"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_content_pack_rejects_cross_host_civitai_download_url(self):
        pack = self._content_pack()
        artifact = pack["artifacts"]["private-model"]
        artifact["source"] = {
            "provider": "civitai",
            "host": "civitai.red",
            "model_id": 123,
            "model_version_id": 456,
            "filename": "model.safetensors",
            "download_url": "https://example.invalid/api/download/models/456",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "download URL on civitai.red"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_workflow_artifact_destination_must_be_json(self):
        pack = self._content_pack()
        pack["artifacts"]["private-model"]["target"] = "workflows"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.json"
            path.write_text(json.dumps(pack))
            with self.assertRaisesRegex(
                LauncherError, "destination must end in .json"
            ):
                load_content_packs(load_catalog(), (path,))

    def test_unknown_selector_group_is_rejected(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        bundle = raw["bundles"]["qwen-image-2512-fp8-base"]
        bundle["groups"].append("surprise")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(LauncherError, "unknown selector"):
                load_catalog(path)

    def test_dwarfstar_bundle_requires_one_dedicated_gguf(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        bundle = raw["bundles"][
            "dwarfstar-deepseek-v4-flash-0731-q2-imatrix"
        ]
        artifact = raw["artifacts"][bundle["artifacts"][0]]
        artifact["target"] = "llama-models"
        with tempfile.TemporaryDirectory() as directory:
            path = self._catalog_copy(directory, raw)
            with self.assertRaisesRegex(
                LauncherError, "must reference one dwarfstar-models"
            ):
                load_catalog(path)

    def test_unknown_application_is_rejected(self):
        raw = json.loads(DEFAULT_CATALOG_PATH.read_text())
        raw["bundles"]["qwen-image-2512-fp8-base"][
            "application"
        ] = "unmanaged"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(
                LauncherError, "unknown application"
            ):
                load_catalog(path)


if __name__ == "__main__":
    unittest.main()
