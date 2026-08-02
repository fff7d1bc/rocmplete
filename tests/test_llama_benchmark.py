import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rocmplete.errors import LauncherError
from rocmplete.llama_benchmark import (
    benchmark_rates,
    run_llama_benchmark,
    write_backend_comparison,
)


class LlamaBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.remove_container_patcher = patch(
            "rocmplete.llama_benchmark.podman.remove_container"
        )
        self.remove_container = self.remove_container_patcher.start()
        self.addCleanup(self.remove_container_patcher.stop)

    @staticmethod
    def _write_result(
        path: Path,
        backend: str,
        prompt_rate: float,
        generation_rate: float,
    ) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "application": "llama-cpp",
                    "backend": backend,
                    "parameters": {
                        "repetitions": 5,
                        "prompt_tokens": 512,
                        "generation_tokens": 128,
                    },
                    "results": [
                        {
                            "n_prompt": 512,
                            "n_gen": 0,
                            "avg_ts": prompt_rate,
                        },
                        {
                            "n_prompt": 0,
                            "n_gen": 128,
                            "avg_ts": generation_rate,
                        },
                    ],
                }
            )
        )

    @patch(
        "rocmplete.llama_benchmark.podman.image_id",
        return_value="sha256:" + "a" * 64,
    )
    @patch(
        "rocmplete.llama_benchmark.podman.capture_stdout",
        return_value='[{"model_filename": "model.gguf", "avg_ts": 42.0}]',
    )
    @patch(
        "rocmplete.llama_benchmark.podman.container_exists",
        side_effect=(False, False),
    )
    def test_result_wraps_structured_output_and_image_identity(
        self, container_exists, capture_stdout, image_id
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            result = run_llama_benchmark(
                ["podman", "run", "--rm", "image"],
                data_dir=root,
                image="localhost/llama",
                profile="cpu",
                backend="rocm",
                render_nodes=(),
                model={"kind": "local", "path": "/models/model.gguf"},
                parameters={
                    "repetitions": 2,
                    "prompt_tokens": 32,
                    "generation_tokens": 16,
                },
                output=output,
            )
            value = json.loads(output.read_text())
        self.assertEqual(result, output)
        self.assertEqual(value["schema_version"], 3)
        self.assertEqual(value["backend"], "rocm")
        self.assertEqual(value["image"]["id"], "sha256:" + "a" * 64)
        self.assertEqual(value["render_nodes"], [])
        self.assertEqual(value["gpu_split"], "none")
        self.assertEqual(value["results"][0]["avg_ts"], 42.0)
        self.assertEqual(value["parameters"]["prompt_tokens"], 32)
        self.remove_container.assert_called_once_with(
            "rocmplete-llama-benchmark", stop_timeout=0
        )

    @patch(
        "rocmplete.llama_benchmark.podman.capture_stdout",
        side_effect=LauncherError("interrupted"),
    )
    @patch(
        "rocmplete.llama_benchmark.podman.container_exists",
        side_effect=(False, True),
    )
    def test_failure_removes_named_container(
        self, container_exists, capture_stdout
    ):
        with self.assertRaisesRegex(LauncherError, "interrupted"):
            run_llama_benchmark(
                ["podman", "run", "--rm", "image"],
                data_dir=Path("/data"),
                image="localhost/llama",
                profile="rdna4",
                backend="rocm",
                render_nodes=("/dev/dri/renderD128",),
                model={"kind": "local", "path": "/models/model.gguf"},
                parameters={
                    "repetitions": 1,
                    "prompt_tokens": 1,
                    "generation_tokens": 1,
                },
            )
        self.remove_container.assert_called_once_with(
            "rocmplete-llama-benchmark", stop_timeout=0
        )

    @patch(
        "rocmplete.llama_benchmark.podman.capture_stdout",
        side_effect=LauncherError("primary benchmark failure"),
    )
    @patch(
        "rocmplete.llama_benchmark.podman.container_exists",
        return_value=False,
    )
    def test_cleanup_failure_does_not_replace_benchmark_failure(
        self, container_exists, capture_stdout
    ):
        self.remove_container.side_effect = LauncherError("cleanup failure")
        with redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaisesRegex(
                LauncherError, "primary benchmark failure"
            ):
                run_llama_benchmark(
                    ["podman", "run", "image"],
                    data_dir=Path("/data"),
                    image="localhost/llama",
                    profile="rdna4",
                    backend="rocm",
                    render_nodes=("/dev/dri/renderD128",),
                    model={"kind": "managed"},
                    parameters={"repetitions": 1},
                )
        self.assertIn("cleanup failure", stderr.getvalue())

    @patch(
        "rocmplete.llama_benchmark.podman.image_id",
        return_value="sha256:" + "b" * 64,
    )
    @patch(
        "rocmplete.llama_benchmark.podman.capture_stdout",
        return_value='[{"avg_ts": 42.0}]',
    )
    @patch(
        "rocmplete.llama_benchmark.podman.container_exists",
        side_effect=(False, False),
    )
    def test_result_records_managed_layer_split(
        self, container_exists, capture_stdout, image_id
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            run_llama_benchmark(
                ["podman", "run", "--rm", "image"],
                data_dir=Path(directory),
                image="localhost/llama",
                profile="rdna4",
                backend="vulkan",
                render_nodes=(
                    "/dev/dri/renderD128",
                    "/dev/dri/renderD129",
                ),
                model={"kind": "managed", "path": "model.gguf"},
                parameters={
                    "repetitions": 1,
                    "prompt_tokens": 1,
                    "generation_tokens": 1,
                },
                output=output,
            )
            value = json.loads(output.read_text())
        self.assertEqual(
            value["render_nodes"],
            ["/dev/dri/renderD128", "/dev/dri/renderD129"],
        )
        self.assertEqual(value["gpu_split"], "layer")
        self.assertEqual(value["backend"], "vulkan")

    @patch(
        "rocmplete.llama_benchmark.podman.capture_stdout",
        return_value="not json",
    )
    @patch(
        "rocmplete.llama_benchmark.podman.container_exists",
        side_effect=(False, False),
    )
    def test_invalid_json_is_not_written(
        self, container_exists, capture_stdout
    ):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with self.assertRaisesRegex(LauncherError, "invalid JSON"):
                run_llama_benchmark(
                    ["podman", "run", "--rm", "image"],
                    data_dir=Path(directory),
                    image="localhost/llama",
                    profile="cpu",
                    backend="rocm",
                    render_nodes=(),
                    model={"kind": "local", "path": "/models/model.gguf"},
                    parameters={
                        "repetitions": 1,
                        "prompt_tokens": 1,
                        "generation_tokens": 1,
                    },
                    output=output,
                )
            self.assertFalse(output.exists())

    @patch(
        "rocmplete.llama_benchmark.podman.image_id",
        return_value="sha256:" + "c" * 64,
    )
    def test_backend_comparison_records_rates_and_winners(self, image_id):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rocm = root / "rocm.json"
            vulkan = root / "vulkan.json"
            output = root / "comparison.json"
            self._write_result(rocm, "rocm", 1079.4, 51.54)
            self._write_result(vulkan, "vulkan", 966.9, 57.86)
            destination, value = write_backend_comparison(
                data_dir=root,
                image="localhost/llama",
                profile="strix-halo",
                render_nodes=("/dev/dri/renderD128",),
                model={"kind": "catalog", "preset": "assistant"},
                parameters={
                    "repetitions": 5,
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                },
                results={"rocm": rocm, "vulkan": vulkan},
                errors={},
                output=output,
            )
            saved = json.loads(output.read_text())
        self.assertEqual(destination, output)
        self.assertEqual(value, saved)
        self.assertEqual(saved["schema_version"], 1)
        self.assertEqual(saved["kind"], "backend-comparison")
        self.assertEqual(
            saved["comparison"]["prompt_processing"]["winner"], "rocm"
        )
        self.assertEqual(
            saved["comparison"]["token_generation"]["winner"], "vulkan"
        )
        self.assertEqual(
            saved["comparison"]["estimated_inference_time"]["winner"],
            "vulkan",
        )
        self.assertAlmostEqual(
            saved["backends"]["rocm"]["estimated_inference_seconds"],
            512 / 1079.4 + 128 / 51.54,
        )

    @patch(
        "rocmplete.llama_benchmark.podman.image_id",
        return_value="sha256:" + "d" * 64,
    )
    def test_backend_comparison_preserves_partial_failure(self, image_id):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rocm = root / "rocm.json"
            self._write_result(rocm, "rocm", 1000.0, 50.0)
            output, value = write_backend_comparison(
                data_dir=root,
                image="localhost/llama",
                profile="strix-halo",
                render_nodes=("/dev/dri/renderD128",),
                model={"kind": "catalog", "preset": "assistant"},
                parameters={
                    "repetitions": 5,
                    "prompt_tokens": 512,
                    "generation_tokens": 128,
                },
                results={"rocm": rocm},
                errors={"vulkan": "Vulkan initialization failed"},
            )
            saved = json.loads(output.read_text())
        self.assertIsNone(value["comparison"])
        self.assertEqual(saved["backends"]["rocm"]["status"], "pass")
        self.assertEqual(saved["backends"]["vulkan"]["status"], "fail")
        self.assertEqual(
            saved["backends"]["vulkan"]["error"],
            "Vulkan initialization failed",
        )

    def test_benchmark_rates_require_exact_prompt_and_generation_rows(self):
        result = {
            "parameters": {
                "prompt_tokens": 512,
                "generation_tokens": 128,
            },
            "results": [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 1000.0},
            ],
        }
        with self.assertRaisesRegex(LauncherError, "pp0/tg128"):
            benchmark_rates(result)


if __name__ == "__main__":
    unittest.main()
