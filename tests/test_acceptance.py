import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocmplete.acceptance import (
    _acceptance_output_path,
    acceptance_definition,
    acceptance_fingerprint,
    begin_case,
    checkpoint,
    complete_case,
    create_result,
    finish,
    load_result,
    pending_case_identifiers,
    render_markdown,
    required_bundles,
    required_images,
    run_comfyui_case,
    run_dwarfstar_case,
    selected_cases,
    smoke_comfy_prompt,
    smoke_comfy_video_prompt,
    source_identity,
)
from rocmplete.benchmark import load_benchmark_prompt
from rocmplete.catalog import load_catalog
from rocmplete.config import APPLICATIONS, ROCM_BASE_IMAGE
from rocmplete.errors import LauncherError


class AcceptanceTests(unittest.TestCase):
    def test_source_identity_distinguishes_tracked_dirty_state(self):
        revision = b"a" * 40

        def result(stdout):
            return subprocess.CompletedProcess(
                args=("git",),
                returncode=0,
                stdout=stdout,
                stderr=b"",
            )

        with patch(
            "rocmplete.acceptance.subprocess.run",
            side_effect=(
                result(revision + b"\n"),
                result(b""),
                result(b""),
            ),
        ):
            self.assertEqual(source_identity(), revision.decode("ascii"))

        with patch(
            "rocmplete.acceptance.subprocess.run",
            side_effect=(
                result(revision + b"\n"),
                result(b"diff --git a/src/a.py b/src/a.py\n+change\n"),
                result(b""),
            ),
        ):
            dirty = source_identity()
        self.assertRegex(
            dirty,
            r"^{}-dirty-[0-9a-f]{{16}}$".format(
                revision.decode("ascii")
            ),
        )

    def test_comfy_acceptance_passes_the_selected_render_node(self):
        catalog = load_catalog()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)

            def fake_benchmark(
                catalog,
                bundle,
                options,
                volume_suffix,
                run_id=None,
                prompt_transform=None,
            ):
                self.assertEqual(
                    options.render_node, "/dev/dri/renderD129"
                )
                output = (
                    data_dir
                    / "apps"
                    / "comfyui"
                    / "output"
                    / "rocmplete-benchmarks"
                    / run_id
                    / "result.png"
                )
                output.parent.mkdir(parents=True)
                output.write_bytes(b"test png")
                return data_dir / "benchmark.json"

            with patch(
                "rocmplete.acceptance.run_benchmark",
                side_effect=fake_benchmark,
            ), patch(
                "rocmplete.acceptance._validate_png",
                return_value={"width": 1, "height": 1},
            ), patch(
                "rocmplete.acceptance.podman.selinux_volume_suffix",
                return_value=":rw",
            ):
                result = run_comfyui_case(
                    catalog,
                    identifier="comfyui-image",
                    data_dir=data_dir,
                    profile="rdna4",
                    render_node="/dev/dri/renderD129",
                    port=8190,
                    suite_id="test",
                    attempt=1,
                    memory_policy="balanced",
                    kernel_policy="default",
                )
        self.assertEqual(result["media"], {"width": 1, "height": 1})

    def test_llama_only_definition_ignores_memory_policy(self):
        catalog = load_catalog()
        llama_cases = selected_cases("strix-point", ("llama-cpp",))
        common = {
            "profile": "strix-point",
            "architecture": "gfx1150",
            "render_node": "/dev/dri/renderD128",
            "image_ids": {},
            "source_identity": "test",
            "kernel_policy": "default",
        }
        balanced = acceptance_definition(
            catalog,
            llama_cases,
            memory_policy="balanced",
            **common,
        )
        conservative = acceptance_definition(
            catalog,
            llama_cases,
            memory_policy="conservative",
            **common,
        )
        self.assertNotIn("memory_policy", balanced)
        self.assertEqual(
            acceptance_fingerprint(balanced),
            acceptance_fingerprint(conservative),
        )

    def test_auto_plans_every_case_until_hardware_is_detected(self):
        cases = selected_cases("auto")

        self.assertEqual(len(cases), 5)
        self.assertTrue(all(not reason for _, reason in cases))

    def test_default_rdna4_suite_keeps_dwarfstar_opt_in(self):
        cases = selected_cases("rdna4")
        states = {
            case.identifier: reason
            for case, reason in cases
        }

        self.assertFalse(states["comfyui-image"])
        self.assertFalse(states["comfyui-video"])
        self.assertFalse(states["llama-cpp"])
        self.assertIn("opt-in outside Strix Halo", states["dwarfstar"])

    def test_explicit_dwarfstar_is_applicable_to_every_gpu_profile(self):
        catalog = load_catalog()
        for profile in ("rdna4", "strix-halo", "strix-point"):
            with self.subTest(profile=profile):
                cases = selected_cases(profile, ("dwarfstar",))
                self.assertEqual(
                    [(case.identifier, reason) for case, reason in cases],
                    [("host-gpu", ""), ("dwarfstar", "")],
                )
                self.assertEqual(
                    required_images(cases),
                    (
                        ("base", ROCM_BASE_IMAGE),
                        ("dwarfstar", APPLICATIONS["dwarfstar"].image),
                    ),
                )
                self.assertEqual(
                    [
                        bundle.identifier
                        for bundle in required_bundles(catalog, cases)
                    ],
                    ["dwarfstar-deepseek-v4-flash-0731-q2-imatrix"],
                )

    def test_dwarfstar_acceptance_runs_a_bounded_direct_answer(self):
        catalog = load_catalog()
        artifact = catalog.artifact(
            "deepseek-v4-flash-0731-iq2xxs-gguf"
        )
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            installed = (
                data_dir
                / "content"
                / "dwarfstar"
                / "models"
                / artifact.destination
            )
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"fixture")
            with patch(
                "rocmplete.acceptance.podman.selinux_volume_suffix",
                return_value=":rw",
            ), patch(
                "rocmplete.acceptance.podman.run_managed_foreground",
                return_value=0,
            ) as run:
                result = run_dwarfstar_case(
                    catalog,
                    data_dir=data_dir,
                    profile="strix-halo",
                    render_node="/dev/dri/renderD128",
                )

        command = run.call_args.args[0]
        self.assertIn("ROCMLETE_DWARFSTAR_CONTEXT=4096", command)
        self.assertIn("ROCMLETE_DWARFSTAR_OUTPUT_TOKENS=64", command)
        self.assertIn("ROCMLETE_DWARFSTAR_NO_THINKING=1", command)
        self.assertIn("/dev/dri/renderD128", command)
        self.assertEqual(result["model"], str(installed))

    def test_visual_review_criteria_distinguish_function_from_aesthetics(self):
        cases = {
            case.identifier: case
            for case, _ in selected_cases("strix-point")
        }

        self.assertIn(
            "aesthetic quality are not graded",
            cases["comfyui-image"].review_criteria[-1],
        )
        self.assertIn(
            "some blur",
            cases["comfyui-video"].review_criteria[1],
        )
        self.assertIn(
            "corruption",
            cases["comfyui-video"].review_criteria[-1],
        )

    def test_application_selection_keeps_host_probe_and_exact_prerequisites(self):
        catalog = load_catalog()
        cases = selected_cases("strix-point", ("llama-cpp",))

        self.assertEqual(
            [case.identifier for case, _ in cases],
            ["host-gpu", "llama-cpp"],
        )
        self.assertEqual(
            required_images(cases),
            (
                ("base", ROCM_BASE_IMAGE),
                ("llama-cpp", APPLICATIONS["llama-cpp"].image),
            ),
        )
        self.assertEqual(
            [bundle.identifier for bundle in required_bundles(catalog, cases)],
            ["llama-qwen3-0.6b-q8-0"],
        )

    def test_comfy_smoke_transform_is_small_four_step_and_non_mutating(self):
        catalog = load_catalog()
        spec = catalog.benchmark("qwen-image-2512-fp8-lightning")
        source = load_benchmark_prompt(spec)
        transformed = smoke_comfy_prompt(source)

        source_serialized = json.dumps(source)
        transformed_serialized = json.dumps(transformed)
        self.assertIn('"value": false', source_serialized.lower())
        self.assertIn('"value": true', transformed_serialized.lower())
        self.assertIn('"width": 768', transformed_serialized)
        self.assertIn('"height": 768', transformed_serialized)
        self.assertNotEqual(source, transformed)

    def test_comfy_smoke_transform_fails_closed_on_upstream_shape(self):
        with self.assertRaisesRegex(LauncherError, "expected switch"):
            smoke_comfy_prompt({"1": {"class_type": "KSampler", "inputs": {}}})

    def test_comfy_video_smoke_transform_is_five_frame_lightning(self):
        catalog = load_catalog()
        spec = catalog.benchmark("wan-2.2-t2v-14b-fp8-lightning")
        source = load_benchmark_prompt(spec)
        transformed = smoke_comfy_video_prompt(source)

        serialized = json.dumps(transformed)
        self.assertIn('"width": 832', serialized)
        self.assertIn('"height": 480', serialized)
        self.assertIn('"value": 0.25', serialized)
        self.assertIn('"value": true', serialized.lower())
        self.assertIn("red cube slowly rotating", serialized)
        self.assertNotEqual(source, transformed)

    def test_comfy_video_smoke_transform_fails_closed_on_upstream_shape(self):
        with self.assertRaisesRegex(LauncherError, "expected switch"):
            smoke_comfy_video_prompt(
                {"1": {"class_type": "KSampler", "inputs": {}}}
            )

    def test_checkpoint_resume_and_visual_review_state_are_explicit(self):
        cases = selected_cases("strix-point", ("comfyui",))
        definition = acceptance_definition(
            load_catalog(),
            cases,
            profile="strix-point",
            architecture="gfx1150",
            render_node="/dev/dri/renderD128",
            image_ids={},
            source_identity="test",
            memory_policy="balanced",
            kernel_policy="default",
        )
        fingerprint = acceptance_fingerprint(definition)
        result = create_result(
            definition,
            cases,
            hardware={
                "Device": "test",
                "Architecture": "gfx1150",
                "Profile": "strix-point",
            },
        )
        self.assertEqual(result["fingerprint"], fingerprint)
        self.assertNotIn("signature", result)
        host = result["cases"][0]
        visuals = result["cases"][1:]
        begin_case(host)
        complete_case(host, {}, started=0.0)
        for index, visual in enumerate(visuals):
            begin_case(visual)
            complete_case(
                visual,
                {"artifacts": ["/tmp/output-{}".format(index)]},
                started=0.0,
            )

        self.assertEqual(host["status"], "pass")
        self.assertTrue(all(item["status"] == "blocked" for item in visuals))
        self.assertEqual(pending_case_identifiers(result), ())
        self.assertIn(
            "Composition, sharpness, and aesthetic quality are not graded.",
            visuals[0]["review_criteria"],
        )
        self.assertIn(
            "## Visual review criteria",
            render_markdown(result),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            checkpoint(path, result)
            resumed = load_result(path, fingerprint)
            self.assertEqual(resumed["status"], "blocked")
            with self.assertRaisesRegex(LauncherError, "changed"):
                load_result(path, "0" * 64)
            tampered = json.loads(path.read_text())
            tampered["definition"]["profile"] = "rdna4"
            path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(LauncherError, "definition"):
                load_result(path, fingerprint)

    def test_resume_rejects_unsafe_suite_id_and_changed_case_metadata(self):
        catalog = load_catalog()
        cases = selected_cases("strix-point", ("comfyui",))
        definition = acceptance_definition(
            catalog,
            cases,
            profile="strix-point",
            architecture="gfx1150",
            render_node="/dev/dri/renderD128",
            image_ids={},
            source_identity="test",
            memory_policy="balanced",
            kernel_policy="default",
        )
        result = create_result(
            definition,
            cases,
            hardware={"Architecture": "gfx1150"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            checkpoint(path, result)
            unsafe = json.loads(path.read_text())
            unsafe["suite_id"] = "../escape"
            path.write_text(json.dumps(unsafe))
            with self.assertRaisesRegex(LauncherError, "suite ID"):
                load_result(path)

            changed = dict(result)
            changed["cases"] = [dict(entry) for entry in result["cases"]]
            changed["cases"][0]["description"] = "changed"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(LauncherError, "metadata"):
                load_result(path)

            invalid_state = dict(result)
            invalid_state["cases"] = [
                dict(entry) for entry in result["cases"]
            ]
            invalid_state["cases"][0]["attempts"] = -1
            path.write_text(json.dumps(invalid_state))
            with self.assertRaisesRegex(LauncherError, "case state"):
                load_result(path)

            wrong_status = dict(result)
            wrong_status["status"] = "pass"
            path.write_text(json.dumps(wrong_status))
            with self.assertRaisesRegex(LauncherError, "status.*cases"):
                load_result(path)

    def test_schema_two_result_is_preserved_but_not_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "acceptance.json"
            original = b'{"schema_version": 2, "suite_id": "old"}\n'
            path.write_bytes(original)
            with self.assertRaisesRegex(LauncherError, "unsupported schema"):
                load_result(path)
            self.assertEqual(path.read_bytes(), original)

    def test_acceptance_output_rejects_symlinked_suite_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            root.mkdir()
            outside = Path(directory) / "outside"
            outside.mkdir()
            (root / "safe").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(LauncherError, "escapes"):
                _acceptance_output_path(
                    root, "safe", 1, "safe/result.png"
                )

    def test_failed_and_interrupted_cases_are_retryable(self):
        result = {
            "cases": [
                {"identifier": "failed", "status": "fail"},
                {"identifier": "interrupted", "status": "interrupted"},
                {"identifier": "passed", "status": "pass"},
            ]
        }

        self.assertEqual(
            pending_case_identifiers(result),
            ("failed", "interrupted"),
        )

    def test_finish_writes_neighboring_markdown_report(self):
        result = {
            "suite_id": "test",
            "status": "running",
            "cases": [
                {
                    "identifier": "host-gpu",
                    "application": None,
                    "status": "pass",
                    "artifacts": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            report = finish(path, result)

            self.assertEqual(report, Path(directory) / "result.md")
            self.assertIn("# ROCmplete smoke acceptance", report.read_text())
            self.assertEqual(json.loads(path.read_text())["status"], "pass")

    def test_initial_checkpoint_refuses_to_replace_racing_output(self):
        result = {
            "suite_id": "test",
            "status": "running",
            "cases": [
                {"identifier": "host-gpu", "status": "pending"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            checkpoint(path, result, create=True)
            original = path.read_bytes()
            with self.assertRaisesRegex(LauncherError, "refusing to replace"):
                checkpoint(path, result, create=True)
            self.assertEqual(path.read_bytes(), original)

    def test_new_report_refuses_to_replace_racing_neighbor(self):
        result = {
            "suite_id": "test",
            "status": "running",
            "cases": [
                {"identifier": "host-gpu", "status": "pass"}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            checkpoint(path, result, create=True)
            report = path.with_suffix(".md")
            report.write_text("preserve")
            with self.assertRaisesRegex(LauncherError, "refusing to replace"):
                finish(path, result, create_report=True)
            self.assertEqual(report.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
