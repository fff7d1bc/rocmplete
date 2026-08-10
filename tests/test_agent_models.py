import unittest

from rocmplete.agent_models import agent_sampling_parameters, is_agent_capable
from rocmplete.catalog import load_catalog
from rocmplete.errors import LauncherError


class AgentModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_every_agent_preset_has_its_reviewed_sampling_policy(self):
        expected = {
            "ornith-1.0-35b-q8-0": (1.0, 0.95, 20, 0.0, 0.0, 1.0),
            "kat-coder-v2.5-dev-q8-0": (
                1.0,
                0.95,
                20,
                0.0,
                1.5,
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
            "qwen3.6-27b-q8-0": (0.6, 0.95, 20, 0.0, 0.0, 1.0),
            "qwen3.6-27b-mtp-q8-0": (
                0.6,
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
            "laguna-s-2.1-q4-k-m": (1.0, 1.0, 20, 0.0, 0.0, 1.0),
            "muse-glimmer-30b-ud-q8-k-xl": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
            "muse-glimmer-30b-ud-q8-k-xl-dflash": (
                1.0,
                0.95,
                64,
                0.0,
                0.0,
                1.0,
            ),
            "muse-glimmer-30b-ud-q8-k-xl-dflash-256k": (
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


if __name__ == "__main__":
    unittest.main()
