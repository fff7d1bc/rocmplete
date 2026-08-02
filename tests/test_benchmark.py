import hashlib
import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rocmplete.benchmark import (
    BenchmarkOptions,
    SYNTHETIC_INPUT_NAME,
    _cleanup_benchmark_resources,
    _write_result,
    load_benchmark_prompt,
    prepare_prompt,
    render_suite_html,
    render_suite_markdown,
    run_benchmark,
    run_benchmark_suite,
    synthetic_png,
    write_suite_reports,
)
from rocmplete.catalog import load_catalog
from rocmplete.errors import LauncherError


class BenchmarkTests(unittest.TestCase):
    @patch(
        "rocmplete.benchmark._remove_container",
        side_effect=LauncherError("cleanup failure"),
    )
    def test_cleanup_failure_is_secondary_to_primary_failure(self, remove):
        primary = LauncherError("primary benchmark failure")
        with tempfile.TemporaryDirectory() as directory:
            isolated_cache = Path(directory) / "isolated-cache"
            isolated_cache.mkdir()
            with redirect_stderr(io.StringIO()) as stderr:
                _cleanup_benchmark_resources(isolated_cache, primary)
            self.assertFalse(isolated_cache.exists())
        self.assertIn("cleanup failure", stderr.getvalue())

    @patch(
        "rocmplete.benchmark._remove_container",
        side_effect=LauncherError("cleanup failure"),
    )
    def test_cleanup_failure_remains_fatal_after_success(self, remove):
        with self.assertRaisesRegex(LauncherError, "cleanup failure"):
            _cleanup_benchmark_resources(None, None)

    def test_run_preserves_workload_failure_when_cleanup_also_fails(self):
        catalog = load_catalog()
        bundle = catalog.bundle("qwen-image-2512-fp8-base")
        statuses = tuple(
            SimpleNamespace(
                state="installed",
                integrity="verified",
                artifact=artifact,
            )
            for artifact in catalog.bundle_artifacts(bundle)
        )
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            options = BenchmarkOptions(
                image="localhost/test",
                profile="rdna4",
                port=8190,
                data_dir=data_dir,
                render_node="/dev/dri/renderD128",
                runs=1,
            )
            with patch(
                "rocmplete.benchmark.inspect_bundle", return_value=statuses
            ), patch(
                "rocmplete.benchmark.web_command",
                return_value=["podman", "run"],
            ), patch(
                "rocmplete.benchmark.podman.require_rootless"
            ), patch(
                "rocmplete.benchmark.podman.image_exists", return_value=True
            ), patch(
                "rocmplete.benchmark.podman.container_exists",
                return_value=False,
            ), patch(
                "rocmplete.benchmark._assert_port_available"
            ), patch(
                "rocmplete.benchmark._image_metadata",
                return_value={"reference": "localhost/test", "id": "image-id"},
            ), patch(
                "rocmplete.benchmark.podman.run", return_value=1
            ), patch(
                "rocmplete.benchmark._container_logs", return_value="log"
            ), patch(
                "rocmplete.benchmark._remove_container",
                side_effect=LauncherError("cleanup failure"),
            ), redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaisesRegex(
                    LauncherError, "cannot start benchmark container"
                ):
                    run_benchmark(
                        catalog,
                        bundle,
                        options,
                        ":rw",
                        run_id="double-failure",
                    )

            result = json.loads(
                (
                    data_dir
                    / "apps"
                    / "comfyui"
                    / "benchmarks"
                    / "double-failure.json"
                ).read_text()
            )
        self.assertEqual(result["error"], "cannot start benchmark container")
        self.assertIn("cleanup failure", stderr.getvalue())

    def test_checkpoint_write_failure_preserves_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text('{"status":"previous"}\n')
            with patch(
                "rocmplete.benchmark.os.replace",
                side_effect=OSError("simulated replacement failure"),
            ):
                with self.assertRaisesRegex(
                    LauncherError, "cannot write benchmark result"
                ):
                    _write_result(path, {"status": "new"})
            self.assertEqual(path.read_text(), '{"status":"previous"}\n')
            self.assertEqual(tuple(path.parent.glob(".*.tmp")), ())

    def test_every_bundle_has_a_pinned_loadable_prompt(self):
        catalog = load_catalog()
        for bundle in catalog.bundles.values():
            if not bundle.workflow:
                continue
            with self.subTest(bundle=bundle.identifier):
                spec = catalog.benchmark(bundle.identifier)
                prompt = load_benchmark_prompt(spec)
                self.assertTrue(prompt)
                serialized = json.dumps(prompt)
                for artifact in catalog.bundle_artifacts(bundle):
                    self.assertIn(
                        artifact.destination.rsplit("/", 1)[-1],
                        serialized,
                    )

    def test_prompt_preparation_sets_seeds_media_and_outputs(self):
        prompt = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": 99, "model": ["2", 0]},
            },
            "2": {
                "class_type": "LoadImage",
                "inputs": {"image": "upstream.png"},
            },
            "3": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "ComfyUI"},
            },
        }
        prepared, has_input = prepare_prompt(
            prompt, 10, "rocmplete-benchmarks/test"
        )
        self.assertTrue(has_input)
        self.assertEqual(prepared["1"]["inputs"]["seed"], 10)
        self.assertEqual(
            prepared["2"]["inputs"]["image"], SYNTHETIC_INPUT_NAME
        )
        self.assertEqual(
            prepared["3"]["inputs"]["filename_prefix"],
            "rocmplete-benchmarks/test",
        )
        self.assertEqual(prompt["1"]["inputs"]["seed"], 99)

    def test_synthetic_png_is_deterministic_rgb_768(self):
        first = synthetic_png()
        second = synthetic_png()
        self.assertEqual(first, second)
        self.assertEqual(first[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, color = struct.unpack(
            ">IIBB", first[16:26]
        )
        self.assertEqual((width, height, depth, color), (768, 768, 8, 2))
        self.assertEqual(
            hashlib.sha256(first).hexdigest(),
            hashlib.sha256(second).hexdigest(),
        )

    def test_isolated_cache_mode_is_explicit_in_dry_run(self):
        catalog = load_catalog()
        bundle = catalog.bundle("qwen-image-2512-fp8-base")
        with tempfile.TemporaryDirectory() as directory:
            options = BenchmarkOptions(
                image="localhost/test",
                profile="rdna4",
                port=8190,
                data_dir=Path(directory),
                render_node="/dev/dri/renderD128",
                dry_run=True,
                cache_mode="isolated",
            )
            with redirect_stdout(io.StringIO()) as output:
                run_benchmark(
                    catalog,
                    bundle,
                    options,
                    ":rw",
                    run_id="isolated-test",
                )
            text = output.getvalue()
            self.assertIn("Cache mode: isolated", text)
            self.assertIn(
                "HOME=/data/benchmarks/.cache/isolated-test/home", text
            )
            self.assertIn("core=0:0", text)

    def test_isolated_cache_rejects_parent_directory_run_id(self):
        catalog = load_catalog()
        bundle = catalog.bundle("qwen-image-2512-fp8-base")
        with tempfile.TemporaryDirectory() as directory:
            options = BenchmarkOptions(
                image="localhost/test",
                profile="rdna4",
                port=8190,
                data_dir=Path(directory),
                render_node="/dev/dri/renderD128",
                dry_run=True,
                cache_mode="isolated",
            )
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(LauncherError, "path-safe"):
                    run_benchmark(
                        catalog,
                        bundle,
                        options,
                        ":rw",
                        run_id="..",
                    )

    def test_hunyuan_accelerated_prompts_match_task_and_resolution(self):
        catalog = load_catalog()
        cases = (
            (
                "hunyuan-video-1.5-i2v-480p-step-distilled",
                "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors",
                12,
                7,
            ),
            (
                "hunyuan-video-1.5-t2v-480p-cfg-distilled",
                "hunyuanvideo1.5_480p_t2v_cfg_distilled_fp16.safetensors",
                50,
                5,
            ),
        )
        for identifier, model, steps, shift in cases:
            with self.subTest(bundle=identifier):
                prompt = load_benchmark_prompt(catalog.benchmark(identifier))
                nodes = tuple(prompt.values())
                loader = next(
                    node for node in nodes if node["class_type"] == "UNETLoader"
                )
                scheduler = next(
                    node
                    for node in nodes
                    if node["class_type"] == "BasicScheduler"
                )
                sampling = next(
                    node
                    for node in nodes
                    if node["class_type"] == "ModelSamplingSD3"
                )
                dimensions = next(
                    node
                    for node in nodes
                    if node["class_type"]
                    in (
                        "HunyuanVideo15ImageToVideo",
                        "EmptyHunyuanVideo15Latent",
                    )
                )
                self.assertEqual(loader["inputs"]["unet_name"], model)
                self.assertEqual(scheduler["inputs"]["steps"], steps)
                self.assertEqual(sampling["inputs"]["shift"], shift)
                self.assertEqual(
                    (
                        dimensions["inputs"]["width"],
                        dimensions["inputs"]["height"],
                    ),
                    (832, 480),
                )
                self.assertFalse(
                    any(
                        node["class_type"]
                        == "HunyuanVideo15SuperResolution"
                        for node in nodes
                    )
                )

    def test_suite_reports_summarize_cold_and_warm_results(self):
        suite = {
            "suite_id": "test-suite",
            "status": "completed",
            "configuration": {
                "profile": "rdna4",
                "image": "localhost/rocmplete:latest",
                "runs": 3,
                "memory_policy": "balanced",
                "kernel_policy": "default",
            },
            "entries": [
                {
                    "bundle": "example",
                    "status": "completed",
                    "result": "/tmp/example.json",
                    "summary": {
                        "cold_seconds": 12.5,
                        "warm_mean_seconds": 10.25,
                    },
                }
            ],
        }
        markdown = render_suite_markdown(suite)
        html = render_suite_html(suite)
        self.assertIn("| `example` | completed | 12.50 | 10.25 |", markdown)
        self.assertIn("<td>example</td>", html)
        self.assertIn("<td>10.25</td>", html)
        with tempfile.TemporaryDirectory() as directory:
            suite_path = Path(directory) / "suite.json"
            outputs = write_suite_reports(suite_path, suite, "both")
            self.assertEqual(
                outputs,
                (
                    Path(directory) / "suite.md",
                    Path(directory) / "suite.html",
                ),
            )
            self.assertTrue(all(path.exists() for path in outputs))

    def test_suite_resume_skips_intact_completed_result(self):
        catalog = load_catalog()
        bundle = catalog.bundle("qwen-image-2512-fp8-base")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            options = BenchmarkOptions(
                image="localhost/test",
                profile="rdna4",
                port=8190,
                data_dir=data_dir,
                render_node="/dev/dri/renderD128",
            )

            def fake_benchmark(
                catalog, bundle, options, volume_suffix, run_id=None
            ):
                path = (
                    data_dir
                    / "apps"
                    / "comfyui"
                    / "benchmarks"
                    / (run_id + ".json")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 4,
                            "run_id": run_id,
                            "bundle": bundle.identifier,
                            "status": "completed",
                            "profile": options.profile,
                            "render_node": options.render_node,
                            "memory_policy": options.memory_policy,
                            "kernel_policy": options.kernel_policy,
                            "cache_mode": options.cache_mode,
                            "unconfined": options.unconfined,
                            "image": {"id": "sha256:test-image"},
                            "runs": [
                                {
                                    "index": 0,
                                    "kind": "cold",
                                    "seed": options.seed,
                                    "wall_seconds": 2.0,
                                },
                                {
                                    "index": 1,
                                    "kind": "warm",
                                    "seed": options.seed + 1,
                                    "wall_seconds": 1.0,
                                },
                            ],
                        }
                    )
                )
                return path

            installed = [
                SimpleNamespace(state="installed", integrity="verified")
            ]
            with patch(
                "rocmplete.benchmark.inspect_bundle",
                return_value=installed,
            ), patch(
                "rocmplete.benchmark._image_metadata",
                return_value={
                    "reference": "localhost/test",
                    "id": "sha256:test-image",
                },
            ), patch(
                "rocmplete.benchmark.run_benchmark",
                side_effect=fake_benchmark,
            ) as runner:
                with redirect_stdout(io.StringIO()):
                    suite_path = run_benchmark_suite(
                        catalog,
                        (bundle,),
                        options,
                        ":rw",
                        report_format="none",
                    )
                self.assertEqual(runner.call_count, 1)

            with patch(
                "rocmplete.benchmark.inspect_bundle",
                return_value=installed,
            ), patch(
                "rocmplete.benchmark._image_metadata",
                return_value={
                    "reference": "localhost/test",
                    "id": "sha256:test-image",
                },
            ), patch(
                "rocmplete.benchmark.run_benchmark"
            ) as runner:
                with redirect_stdout(io.StringIO()):
                    run_benchmark_suite(
                        catalog,
                        (bundle,),
                        options,
                        ":rw",
                        resume_path=suite_path,
                        report_format="none",
                    )
                runner.assert_not_called()

    def test_suite_resume_rejects_a_rebuilt_image(self):
        catalog = load_catalog()
        bundle = catalog.bundle("qwen-image-2512-fp8-base")
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            options = BenchmarkOptions(
                image="localhost/test",
                profile="rdna4",
                port=8190,
                data_dir=data_dir,
                render_node="/dev/dri/renderD128",
                dry_run=False,
            )
            installed = [
                SimpleNamespace(state="installed", integrity="verified")
            ]

            def fake_benchmark(
                catalog, bundle, options, volume_suffix, run_id=None
            ):
                path = (
                    data_dir
                    / "apps"
                    / "comfyui"
                    / "benchmarks"
                    / (run_id + ".json")
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 4,
                            "run_id": run_id,
                            "bundle": bundle.identifier,
                            "status": "completed",
                            "profile": options.profile,
                            "render_node": options.render_node,
                            "memory_policy": options.memory_policy,
                            "kernel_policy": options.kernel_policy,
                            "cache_mode": options.cache_mode,
                            "unconfined": options.unconfined,
                            "image": {"id": "sha256:first"},
                            "runs": [
                                {
                                    "index": index,
                                    "kind": (
                                        "cold" if index == 0 else "warm"
                                    ),
                                    "seed": options.seed + index,
                                    "wall_seconds": 1.0,
                                }
                                for index in range(options.runs)
                            ],
                        }
                    )
                )
                return path

            with patch(
                "rocmplete.benchmark.inspect_bundle",
                return_value=installed,
            ), patch(
                "rocmplete.benchmark._image_metadata",
                return_value={"id": "sha256:first"},
            ), patch(
                "rocmplete.benchmark.run_benchmark",
                side_effect=fake_benchmark,
            ):
                with redirect_stdout(io.StringIO()):
                    suite_path = run_benchmark_suite(
                        catalog,
                        (bundle,),
                        options,
                        ":rw",
                        report_format="none",
                    )

            with patch(
                "rocmplete.benchmark.inspect_bundle",
                return_value=installed,
            ), patch(
                "rocmplete.benchmark._image_metadata",
                return_value={"id": "sha256:rebuilt"},
            ):
                with self.assertRaisesRegex(
                    LauncherError, "configuration or catalog inputs changed"
                ):
                    run_benchmark_suite(
                        catalog,
                        (bundle,),
                        options,
                        ":rw",
                        resume_path=suite_path,
                        report_format="none",
                    )


if __name__ == "__main__":
    unittest.main()
