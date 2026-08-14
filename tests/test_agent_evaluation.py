import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rocmplete.agent_evaluation import (
    RESULT_SCHEMA,
    AgentEvaluationOptions,
    PreparedAttempt,
    _build_command,
    _dependency_changed,
    _extract_git_archive,
    _evaluation_sandbox_environment,
    _generated_artifacts,
    _git_fixture,
    _model_identity,
    _server_command,
    _server_readiness_url,
    _snapshot_protected,
    _test_command,
    _validate_agent_tree,
    grade_review,
    load_coding_suite,
    parse_server_metrics,
    render_agent_evaluation_markdown,
    run_agent_evaluation,
    select_coding_tasks,
    transcript_network_attempts,
    transcript_usage,
)
from rocmplete.catalog import load_catalog
from rocmplete.errors import LauncherError


class AgentEvaluationTests(unittest.TestCase):
    def test_result_schema_records_native_reasoning_contract(self):
        self.assertEqual(RESULT_SCHEMA, "rocmplete.coding-agent-evaluation.v2")

    def test_frozen_definition_has_nine_implementation_and_two_review_tasks(self):
        suite = load_coding_suite()
        self.assertEqual(suite.identifier, "rocmplete-coding-v5")
        self.assertEqual(len(suite.tasks), 11)
        self.assertEqual(
            sum(task.kind == "implementation" for task in suite.tasks), 9
        )
        self.assertEqual(sum(task.kind == "review" for task in suite.tasks), 2)
        self.assertEqual(
            sum(task.toolchain == "python-stdlib" for task in suite.tasks), 1
        )
        self.assertEqual(len(suite.fingerprint), 64)
        for task in suite.tasks:
            self.assertEqual(len(task.base_commit), 40)
            self.assertEqual(len(task.base_tree), 40)
            self.assertTrue(task.remote.endswith("/{}.git".format(task.repository)))
            if task.kind == "implementation":
                self.assertIsNotNone(task.hidden)
                self.assertTrue(task.hidden.resource.is_file())
            else:
                self.assertEqual(task.answer, "ROCMLETE_EVAL_ANSWER.md")
                self.assertIn("between 200 and 2,000 words", task.prompt)

    def test_model_identity_records_profile_runtime_policy(self):
        identity = _model_identity(
            load_catalog(),
            AgentEvaluationOptions(
                data_dir=Path("/unused"),
                preset="qwen3.6-27b-mtp-q8-0",
            ),
        )
        self.assertEqual(identity["draft_tokens"], 3)
        self.assertEqual(
            identity["flash_attention"], {"strix-halo": "on"}
        )
        self.assertEqual(identity["kv_cache"], {"strix-halo": "q8_0"})
        self.assertEqual(
            identity["reasoning"],
            {
                "control": "toggle",
                "client_level": "high",
                "native_value": "on",
            },
        )

        muse_identity = _model_identity(
            load_catalog(),
            AgentEvaluationOptions(
                data_dir=Path("/unused"),
                preset=(
                    "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash"
                ),
                backend="vulkan",
            ),
        )
        self.assertEqual(muse_identity["draft_tokens"], 4)
        self.assertEqual(
            muse_identity["draft_tokens_by_backend"], {"vulkan": 4}
        )
        self.assertEqual(muse_identity["reasoning"]["control"], "strength")

    def test_toolchains_select_fixed_controller_commands(self):
        tasks = {task.identifier: task for task in load_coding_suite().tasks}
        go_task = tasks["proxy-late-probe"]
        python_task = tasks["rc-selinux-verify"]

        self.assertEqual(
            _test_command(go_task), ("go", "test", "-count=1", "./...")
        )
        self.assertEqual(_build_command(go_task), ("go", "build", "./..."))
        self.assertEqual(
            _test_command(python_task),
            ("python3", "-m", "unittest", "discover", "-s", "tests"),
        )
        self.assertIn("compileall", _build_command(python_task))
        self.assertTrue(
            _dependency_changed(
                python_task,
                ((" M", "containers/x/requirements.txt"),),
            )
        )
        self.assertFalse(
            _dependency_changed(
                python_task,
                ((" M", "src/rocmplete/podman.py"),),
            )
        )

    def test_definition_rejects_repository_toolchain_mismatch(self):
        source = Path(__file__).parents[1] / "evaluations" / "coding"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "coding"
            import shutil

            shutil.copytree(source, destination)
            definition = destination / "tasks.json"
            raw = json.loads(definition.read_text())
            task = next(
                task
                for task in raw["tasks"]
                if task["identifier"] == "rc-selinux-verify"
            )
            task["toolchain"] = "go"
            definition.write_text(json.dumps(raw))
            with self.assertRaisesRegex(LauncherError, "unreviewed repository"):
                load_coding_suite(definition)

    def test_definition_rejects_unsafe_python_hidden_destination(self):
        source = Path(__file__).parents[1] / "evaluations" / "coding"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "coding"
            import shutil

            shutil.copytree(source, destination)
            definition = destination / "tasks.json"
            raw = json.loads(definition.read_text())
            task = next(
                task
                for task in raw["tasks"]
                if task["identifier"] == "rc-selinux-verify"
            )
            task["hidden"]["destination"] = "../test_escape.py"
            definition.write_text(json.dumps(raw))
            with self.assertRaisesRegex(LauncherError, "unsafe hidden-test"):
                load_coding_suite(definition)

    def test_definition_fails_closed_when_hidden_test_changes(self):
        source = Path(__file__).parents[1] / "evaluations" / "coding"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "coding"
            import shutil

            shutil.copytree(source, destination)
            hidden = destination / "hidden" / "re-align" / "rocmplete_eval_test.go"
            hidden.write_text(hidden.read_text() + "\n")
            with self.assertRaisesRegex(LauncherError, "hash mismatch"):
                load_coding_suite(destination / "tasks.json")

    def test_task_selection_preserves_requested_order_and_deduplicates(self):
        suite = load_coding_suite()
        selected = select_coding_tasks(
            suite, ("fz-eintr", "re-align", "fz-eintr")
        )
        self.assertEqual(
            [task.identifier for task in selected], ["fz-eintr", "re-align"]
        )
        with self.assertRaisesRegex(LauncherError, "unknown.*missing"):
            select_coding_tasks(suite, ("missing",))

    def test_archive_extraction_accepts_regular_files_and_rejects_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as handle:
                member = tarfile.TarInfo("src/main.go")
                contents = b"package main\n"
                member.size = len(contents)
                member.mode = 0o644
                handle.addfile(member, io.BytesIO(contents))
            _extract_git_archive(archive.getvalue(), root / "good")
            self.assertEqual(
                (root / "good" / "src" / "main.go").read_bytes(), contents
            )

            unsafe = io.BytesIO()
            with tarfile.open(fileobj=unsafe, mode="w") as handle:
                member = tarfile.TarInfo("../escape")
                member.size = 1
                handle.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(LauncherError, "unsafe path"):
                _extract_git_archive(unsafe.getvalue(), root / "bad")

    def test_fixture_has_one_commit_no_remote_and_controlled_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture"
            fixture.mkdir()
            (fixture / "main.go").write_text("package main\n")
            _git_fixture(fixture, "# controlled\n")
            count = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=fixture,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            remotes = subprocess.run(
                ["git", "remote"],
                cwd=fixture,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            self.assertEqual(count, "1")
            self.assertEqual(remotes, "")
            self.assertEqual((fixture / "AGENTS.md").read_text(), "# controlled\n")

    def test_fixture_replaces_existing_instructions_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rejected = root / "rejected"
            rejected.mkdir()
            (rejected / "AGENTS.md").write_text("# upstream\n")
            with self.assertRaisesRegex(LauncherError, "unexpectedly contains"):
                _git_fixture(rejected, "# controlled\n")

            allowed = root / "allowed"
            allowed.mkdir()
            (allowed / "AGENTS.md").write_text("# upstream\n")
            _git_fixture(
                allowed,
                "# controlled\n",
                replace_existing_notes=True,
            )
            self.assertEqual((allowed / "AGENTS.md").read_text(), "# controlled\n")

    def test_agent_tree_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "regular").write_text("ok")
            (root / "link").symlink_to(root / "regular")
            valid, reason = _validate_agent_tree(root)
            self.assertFalse(valid)
            self.assertIn("unsupported file type", reason)

    def test_transcript_extracts_usage_and_audits_network_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            transcript = Path(directory) / "pi.jsonl"
            events = (
                {
                    "type": "message_end",
                    "message": {
                        "usage": {
                            "input": 100,
                            "output": 20,
                            "cacheRead": 5,
                            "cacheWrite": 3,
                        }
                    },
                },
                {"type": "tool", "input": {"command": "git status --short"}},
                {"type": "tool", "input": {"command": "curl https://example.invalid"}},
                {
                    "type": "tool",
                    "input": {"command": "python3 -m pip install example"},
                },
            )
            transcript.write_text("".join(json.dumps(item) + "\n" for item in events))
            self.assertEqual(
                transcript_usage(transcript),
                {
                    "input": 100,
                    "output": 20,
                    "reasoning": 0,
                    "cache_read": 5,
                    "cache_write": 3,
                },
            )
            self.assertEqual(
                transcript_network_attempts(transcript),
                (
                    "curl https://example.invalid",
                    "python3 -m pip install example",
                ),
            )

    def test_evaluation_sandbox_uses_neutral_identity_and_terminal(self):
        environment = _evaluation_sandbox_environment(
            {
                "PATH": "/bin",
                "TERM": "xterm-private",
                "GIT_AUTHOR_NAME": "Host User",
                "GIT_AUTHOR_EMAIL": "host@example.invalid",
            }
        )
        self.assertEqual(environment["PATH"], "/bin")
        self.assertEqual(environment["TERM"], "dumb")
        self.assertEqual(environment["GIT_AUTHOR_NAME"], "ROCmplete Evaluation")
        self.assertEqual(
            environment["GIT_AUTHOR_EMAIL"], "evaluation@invalid.local"
        )

    def test_server_metrics_use_aggregate_tokens_and_last_reported_rates(self):
        metrics = parse_server_metrics(
            b"prompt eval time = 10 ms / 100 tokens (1000.0 tokens per second)\n"
            b"eval time = 20 ms / 20 runs (100.0 tokens per second)\n"
            b"prompt eval time = 5 ms / 50 tokens (900.0 tokens per second)\n"
            b"eval time = 10 ms / 10 runs (80.0 tokens per second)\n"
        )
        self.assertEqual(metrics["prompt_tokens"], 150)
        self.assertEqual(metrics["generation_tokens"], 30)
        self.assertEqual(metrics["prompt_tokens_per_second"], 900.0)
        self.assertEqual(metrics["generation_tokens_per_second"], 80.0)

    def test_repository_named_build_output_is_a_generated_artifact(self):
        tasks = {task.identifier: task for task in load_coding_suite().tasks}
        task = tasks["fz-eintr"]
        self.assertEqual(
            _generated_artifacts(
                task,
                ((" M", "scanner.go"), ("??", "fzr"), ("??", "notes.txt")),
            ),
            ("fzr",),
        )
        self.assertEqual(
            _generated_artifacts(
                tasks["rc-selinux-verify"], (("??", "rocmplete"),)
            ),
            (),
        )

    def test_review_grade_preserves_answer_and_rejects_other_changes(self):
        suite = load_coding_suite()
        task = next(
            task
            for task in suite.tasks
            if task.identifier == "review-fzr-concurrency"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            (fixture / "main.go").write_text("package main\n")
            _git_fixture(fixture, suite.fixture_instructions)
            protected = root / "protected"
            _snapshot_protected(fixture, protected)
            answer = "picker.go scanner.go " + "evidence " * 220
            (fixture / task.answer).write_text(answer)
            attempt = PreparedAttempt(
                task=task,
                repetition=1,
                root=root,
                fixture=fixture,
                protected=protected,
            )
            grade = grade_review(attempt, pi_exit=0, network_attempts=())
            self.assertEqual(grade["outcome"], "review-pending")
            self.assertTrue((root / "answer.md").is_file())

            (fixture / task.answer).write_text(
                "picker.go scanner.go " + "evidence " * 2001
            )
            grade = grade_review(attempt, pi_exit=0, network_attempts=())
            self.assertEqual(grade["outcome"], "invalid")
            self.assertGreater(grade["answer_words"], 2000)

    def test_server_command_is_explicit_for_llama_and_dwarfstar(self):
        llama = _server_command(
            AgentEvaluationOptions(
                data_dir=Path("/data"), preset="qwen", port=9000
            )
        )
        self.assertIn("llama-cpp", llama)
        self.assertIn("--preset", llama)
        self.assertIn("qwen", llama)
        self.assertIn("--detach", llama)
        dwarfstar = _server_command(
            AgentEvaluationOptions(
                data_dir=Path("/data"), dwarfstar=True, port=9001
            )
        )
        self.assertIn("dwarfstar", dwarfstar)
        self.assertNotIn("--backend", dwarfstar)

    def test_server_readiness_endpoint_is_engine_specific(self):
        llama = AgentEvaluationOptions(
            data_dir=Path("/data"), preset="qwen", port=9000
        )
        dwarfstar = AgentEvaluationOptions(
            data_dir=Path("/data"), dwarfstar=True, port=9001
        )
        self.assertEqual(
            _server_readiness_url(llama), "http://127.0.0.1:9000/health"
        )
        self.assertEqual(
            _server_readiness_url(dwarfstar),
            "http://127.0.0.1:9001/v1/models",
        )

    def test_dwarfstar_rejects_unmapped_thinking_level(self):
        with self.assertRaisesRegex(LauncherError, "thinking off or high"):
            run_agent_evaluation(
                load_catalog(),
                AgentEvaluationOptions(
                    data_dir=Path("/unused"),
                    dwarfstar=True,
                    thinking="medium",
                    dry_run=True,
                ),
            )

    def test_dry_run_does_not_create_data_or_start_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "absent"
            with redirect_stdout(io.StringIO()) as output:
                path, result = run_agent_evaluation(
                    load_catalog(),
                    AgentEvaluationOptions(
                        data_dir=data_dir,
                        preset="qwen3.6-27b-q8-0",
                        tasks=("re-align",),
                        dry_run=True,
                    ),
                )
            self.assertEqual(path, Path())
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(data_dir.exists())
            self.assertIn("Coding-agent evaluation", output.getvalue())

    def test_model_native_reasoning_controls_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "absent"
            options = AgentEvaluationOptions(
                data_dir=data_dir,
                preset="qwen3.8-27b-mtp-ud-q8-k-xl",
                tasks=("re-align",),
                context=262144,
                thinking="medium",
                dry_run=True,
            )
            with redirect_stdout(io.StringIO()) as output:
                run_agent_evaluation(load_catalog(), options)
            self.assertIn("Model       qwen3.8", output.getvalue())
            self.assertIn(
                "Thinking    medium (effort=medium)", output.getvalue()
            )
            self.assertFalse(data_dir.exists())

            with self.assertRaisesRegex(
                LauncherError, "supports --thinking off or low or medium or xhigh"
            ):
                run_agent_evaluation(
                    load_catalog(),
                    AgentEvaluationOptions(
                        data_dir=data_dir,
                        preset="qwen3.8-27b-mtp-ud-q8-k-xl",
                        tasks=("re-align",),
                        thinking="high",
                        dry_run=True,
                    ),
                )
            with self.assertRaisesRegex(
                LauncherError, "supports --thinking low or medium or high or xhigh"
            ):
                run_agent_evaluation(
                    load_catalog(),
                    AgentEvaluationOptions(
                        data_dir=data_dir,
                        preset=(
                            "muse-glimmer-30b-kquant-dynamic-q4-k-xl-dflash-256k"
                        ),
                        tasks=("re-align",),
                        thinking="off",
                        dry_run=True,
                    ),
                )

    def test_markdown_keeps_solve_rate_separate_from_review(self):
        report = render_agent_evaluation_markdown(
            {
                "suite": "v1",
                "status": "complete",
                "model": {"identifier": "model"},
                "harness": {"version": "1"},
                "tasks": [
                    {
                        "identifier": "implementation",
                        "kind": "implementation",
                        "difficulty": "easy",
                        "attempts": [
                            {
                                "grade": {"outcome": "solved"},
                                "harness": {
                                    "wall_seconds": 2,
                                    "usage": {"output": 10},
                                },
                            }
                        ],
                    },
                    {
                        "identifier": "review",
                        "kind": "review",
                        "difficulty": "review",
                        "attempts": [
                            {
                                "grade": {"outcome": "review-pending"},
                                "harness": {
                                    "wall_seconds": 1,
                                    "usage": {"output": 5},
                                },
                            }
                        ],
                    },
                ],
            }
        )
        self.assertIn("Implementation solve rate: **1/1**", report)
        self.assertIn("review-pending", report)


if __name__ == "__main__":
    unittest.main()
