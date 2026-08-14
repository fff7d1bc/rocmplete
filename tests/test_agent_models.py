import unittest

from rocmplete.agent_models import (
    NORMALIZED_COMPARISON_CONTEXT,
    NORMALIZED_COMPARISON_MODELS,
    NORMALIZED_COMPARISON_THINKING,
    RECOMMENDED_MODEL,
    agent_sampling_parameters,
    is_agent_capable,
    reasoning_budget,
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

    def test_normalized_comparison_and_reasoning_budgets_are_explicit(self):
        self.assertEqual(
            NORMALIZED_COMPARISON_MODELS,
            (
                "qwen3.6-27b-mtp-q8-0",
                "qwen3.8-27b-mtp-ud-q8-k-xl",
                RECOMMENDED_MODEL,
            ),
        )
        self.assertEqual(NORMALIZED_COMPARISON_CONTEXT, 262144)
        self.assertEqual(NORMALIZED_COMPARISON_THINKING, "high")
        muse = self.catalog.llama_preset(RECOMMENDED_MODEL)
        self.assertEqual(reasoning_budget(muse, "low"), 1024)
        self.assertEqual(reasoning_budget(muse, "high"), 8192)
        self.assertEqual(reasoning_budget(muse, "xhigh"), 16384)
        with self.assertRaisesRegex(LauncherError, "supports --thinking"):
            reasoning_budget(muse, "off")


if __name__ == "__main__":
    unittest.main()
