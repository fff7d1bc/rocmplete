import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from rocmplete.errors import LauncherError
from rocmplete.llama_speculative_benchmark import (
    SpeculativeBenchmarkOptions,
    _digest,
    benchmark_messages,
    parse_trial_response,
    run_speculative_benchmark,
    summarize_trials,
)


def _options(root: Path, **changes) -> SpeculativeBenchmarkOptions:
    values = {
        "data_dir": root,
        "preset": "qwen",
        "image": "localhost/llama",
        "image_id": "sha256:" + "a" * 64,
        "source_identity": "b" * 40,
        "profile": "strix-halo",
        "backend": "rocm",
        "render_nodes": ("/dev/dri/renderD128",),
        "port": 8190,
        "context": 8192,
        "thinking": "medium",
        "native_reasoning": "medium",
        "speculative_type": "draft-mtp",
        "incumbent_depth": 1,
        "depths": (1, 2),
        "context_depths": (4096,),
        "repetitions": 1,
        "generation_tokens": 64,
        "seed": 42,
        "draft_probability_min": 0.0,
        "draft_backend_sampling": True,
        "graph_optimization": False,
        "poll": None,
        "no_host": False,
        "flash_attention": "preset",
        "cache_type_k": "preset",
        "cache_type_v": "preset",
        "batch_size": 2048,
        "ubatch_size": 512,
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "model": {"artifact": "qwen", "sha256": "c" * 64},
        "commands": {1: ("server", "1"), 2: ("server", "2")},
        "output": root / "result.json",
    }
    values.update(changes)
    return SpeculativeBenchmarkOptions(**values)


def _metrics(depth: int = 1):
    rate = 10.0 + depth
    return {
        "request_seconds": 6.0,
        "startup_seconds": 2.0,
        "prompt_sha256": "d" * 64,
        "prompt_tokens": 4096,
        "completion_tokens": 64,
        "prompt_ms": 4096.0,
        "predicted_ms": 64000.0 / rate,
        "prompt_tokens_per_second": 1000.0,
        "generation_tokens_per_second": rate,
        "drafted_tokens": 60,
        "accepted_draft_tokens": 48 + depth,
        "acceptance_percent": 100.0 * (48 + depth) / 60.0,
        "finish_reason": "length",
        "response_sha256": "e" * 64,
        "message": {"content": "", "reasoning_content": "test"},
        "system_fingerprint": "test",
        "server_log_tail": "healthy",
    }


class LlamaSpeculativeBenchmarkTests(unittest.TestCase):
    def test_prompt_generator_is_deterministic_and_seeded(self):
        first = benchmark_messages(3, 42)
        self.assertEqual(first, benchmark_messages(3, 42))
        self.assertNotEqual(first, benchmark_messages(3, 43))
        self.assertIn("transform000002", first[1]["content"])

    def test_response_parser_preserves_speculative_and_timing_metrics(self):
        parsed = parse_trial_response(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "analysis",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 64,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
                "timings": {
                    "cache_n": 0,
                    "prompt_n": 100,
                    "prompt_ms": 50.0,
                    "prompt_per_second": 2000.0,
                    "predicted_n": 64,
                    "predicted_ms": 4000.0,
                    "predicted_per_second": 16.0,
                    "draft_n": 60,
                    "draft_n_accepted": 48,
                },
                "system_fingerprint": "b1-test",
            },
            request_seconds=4.2,
            startup_seconds=1.5,
            prompt_sha256="f" * 64,
        )
        self.assertEqual(parsed["generation_tokens_per_second"], 16.0)
        self.assertEqual(parsed["accepted_draft_tokens"], 48)
        self.assertEqual(parsed["acceptance_percent"], 80.0)
        self.assertEqual(parsed["system_fingerprint"], "b1-test")

    def test_response_parser_rejects_cache_reuse(self):
        with self.assertRaisesRegex(LauncherError, "prompt cache"):
            parse_trial_response(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "ok"}}
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 1},
                    },
                    "timings": {
                        "cache_n": 1,
                        "prompt_ms": 1.0,
                        "prompt_per_second": 2.0,
                        "predicted_n": 1,
                        "predicted_ms": 1.0,
                        "predicted_per_second": 1.0,
                        "draft_n": 1,
                        "draft_n_accepted": 1,
                    },
                },
                request_seconds=1.0,
                startup_seconds=1.0,
                prompt_sha256="f" * 64,
            )

    def test_summary_uses_aggregate_tokens_over_time(self):
        trials = []
        for depth, predicted_ms in ((1, 8000.0), (2, 4000.0)):
            item = {
                "status": "complete",
                "depth": depth,
                "context_depth": 4096,
                "seed": 42,
                **_metrics(depth),
            }
            item["predicted_ms"] = predicted_ms
            trials.append(item)
        summary = summarize_trials(trials, incumbent_depth=1)
        self.assertEqual(summary["winner_depth"], 2)
        self.assertEqual(
            summary["depths"]["2"]["generation_tokens_per_second"], 12.0
        )
        self.assertEqual(summary["screening_decision"], "retest-candidate")

    def test_interrupt_checkpoints_and_resume_skips_completed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _options(root)
            workload = {
                "target_prompt_tokens": 4096,
                "line_count": 0,
                "calibrated_prompt_tokens": 4096,
                "prompt_sha256": _digest(benchmark_messages(0, 42)),
            }
            calls = []

            def interrupt_second(options, trial, selected_workload):
                calls.append(trial["identifier"])
                if len(calls) == 2:
                    raise KeyboardInterrupt
                return _metrics(int(trial["depth"]))

            with patch(
                "rocmplete.llama_speculative_benchmark._calibrate_workloads",
                return_value=[workload],
            ), patch(
                "rocmplete.llama_speculative_benchmark._run_trial",
                side_effect=interrupt_second,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_speculative_benchmark(options)
            checkpoint = json.loads((root / "result.json").read_text())
            self.assertEqual(checkpoint["status"], "interrupted")
            self.assertEqual(
                checkpoint["definition"]["runtime"],
                {
                    "cache_type_k": "preset",
                    "cache_type_v": "preset",
                    "batch_size": 2048,
                    "ubatch_size": 512,
                    "draft_backend_sampling": True,
                    "draft_probability_min": 0.0,
                    "flash_attention": "preset",
                    "graph_optimization": False,
                    "no_host": False,
                    "poll": None,
                },
            )
            self.assertEqual(
                [item["status"] for item in checkpoint["trials"]].count("complete"),
                1,
            )

            resumed = replace(
                options,
                output=None,
                resume=root / "result.json",
            )
            with patch(
                "rocmplete.llama_speculative_benchmark._run_trial",
                side_effect=lambda options, trial, workload: _metrics(
                    int(trial["depth"])
                ),
            ) as run_trial:
                path, result = run_speculative_benchmark(resumed)
            self.assertEqual(path, root / "result.json")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(run_trial.call_count, 1)
            self.assertEqual(result["summary"]["complete_trials"], 2)

    def test_resume_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            options = _options(root)
            workload = {
                "target_prompt_tokens": 4096,
                "line_count": 0,
                "calibrated_prompt_tokens": 4096,
                "prompt_sha256": _digest(benchmark_messages(0, 42)),
            }
            with patch(
                "rocmplete.llama_speculative_benchmark._calibrate_workloads",
                return_value=[workload],
            ), patch(
                "rocmplete.llama_speculative_benchmark._run_trial",
                side_effect=lambda options, trial, selected: _metrics(
                    int(trial["depth"])
                ),
            ):
                run_speculative_benchmark(options)
            changed = replace(
                options,
                output=None,
                resume=root / "result.json",
                draft_probability_min=0.5,
            )
            with self.assertRaisesRegex(LauncherError, "resume inputs"):
                run_speculative_benchmark(changed)


if __name__ == "__main__":
    unittest.main()
