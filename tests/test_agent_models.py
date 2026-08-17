import unittest

from rocmplete.agent_models import (
    RECOMMENDED_MODEL,
    agent_sampling_parameters,
    is_agent_capable,
    recommended_agent_model,
    reasoning_client_default,
    reasoning_client_levels,
    reasoning_native_value,
)
from rocmplete.catalog import load_catalog
from rocmplete.errors import LauncherError


class AgentModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_every_agent_preset_has_its_reviewed_sampling_policy(self):
        expected = {
            "kat-coder-v2.5-dev-q8-0": (
                1.0,
                0.95,
                20,
                0.0,
                1.5,
                1.0,
            ),
            "qwen3.6-27b-q8-0": (0.6, 0.95, 20, 0.0, 0.0, 1.0),
            "qwen3.6-27b-mtp-q8-0": (
                0.6,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.6-35b-a3b-ud-q8-k-xl": (
                0.6,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.6-35b-a3b-mtp-ud-q8-k-xl": (
                0.6,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.8-27b-ud-q8-k-xl": (
                1.0,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.8-27b-mtp-ud-q8-k-xl": (
                1.0,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.8-27b-ud-q4-k-xl": (
                1.0,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "qwen3.8-27b-mtp-ud-q4-k-xl": (
                1.0,
                0.95,
                20,
                0.0,
                0.0,
                1.0,
            ),
            "gemma4-31b-it-q8-0-mtp": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
            "muse-glimmer-30b-kquant-dynamic-q4-k-xl": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
            "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
            "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash-256k": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
        }
        actual = {}
        for identifier, preset in self.catalog.llama_presets.items():
            if not is_agent_capable(preset):
                continue
            params = agent_sampling_parameters(identifier)
            actual[identifier] = (
                params["temperature"],
                params["top_p"],
                params["top_k"],
                params["min_p"],
                params["presence_penalty"],
                params["repeat_penalty"],
            )
        self.assertEqual(actual, expected)

    def test_missing_agent_sampling_policy_fails_closed(self):
        with self.assertRaisesRegex(LauncherError, "no reviewed sampling policy"):
            agent_sampling_parameters("unreviewed-agent")

    def test_reasoning_controls_are_model_native(self):
        recommended = self.catalog.llama_preset(RECOMMENDED_MODEL)
        self.assertEqual(RECOMMENDED_MODEL, "qwen3.8-27b-mtp-ud-q8-k-xl")
        self.assertEqual(
            reasoning_client_levels(recommended),
            ("off", "low", "medium", "xhigh"),
        )
        self.assertEqual(reasoning_client_default(recommended), "medium")
        self.assertEqual(
            reasoning_native_value(recommended, "medium"), "medium"
        )
        with self.assertRaisesRegex(LauncherError, "supports --thinking"):
            reasoning_native_value(recommended, "high")

        qwen36 = self.catalog.llama_preset("qwen3.6-27b-mtp-q8-0")
        self.assertEqual(reasoning_client_levels(qwen36), ("off", "high"))
        self.assertEqual(reasoning_client_default(qwen36), "high")
        self.assertEqual(reasoning_native_value(qwen36, "high"), "on")
        qwen36_sparse = self.catalog.llama_preset(
            "qwen3.6-35b-a3b-mtp-ud-q8-k-xl"
        )
        self.assertEqual(
            reasoning_client_levels(qwen36_sparse), ("off", "high")
        )
        self.assertEqual(reasoning_client_default(qwen36_sparse), "high")
        self.assertEqual(
            reasoning_native_value(qwen36_sparse, "high"), "on"
        )

        muse = self.catalog.llama_preset(
            "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash-256k"
        )
        self.assertEqual(
            reasoning_client_levels(muse),
            ("low", "medium", "high", "xhigh"),
        )
        self.assertEqual(reasoning_client_default(muse), "high")
        self.assertEqual(reasoning_native_value(muse, "high"), "high")
        with self.assertRaisesRegex(LauncherError, "supports --thinking"):
            reasoning_native_value(muse, "off")

    def test_shared_default_uses_the_recommended_native_reasoning_level(self):
        self.assertEqual(
            recommended_agent_model(self.catalog),
            ("rocmplete", "qwen3.8-27b-mtp-ud-q8-k-xl", "medium"),
        )


if __name__ == "__main__":
    unittest.main()
