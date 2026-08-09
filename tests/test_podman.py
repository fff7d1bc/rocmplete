import io
import signal
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, call, mock_open, patch

from rocmplete import podman
from rocmplete.errors import LauncherError


_query_managed_download_containers = podman._managed_download_containers


class PodmanAdapterTests(unittest.TestCase):
    def setUp(self):
        self.download_container_patcher = patch(
            "rocmplete.podman._managed_download_containers",
            return_value=(),
        )
        self.download_containers = self.download_container_patcher.start()
        self.addCleanup(self.download_container_patcher.stop)
        self.stderr = io.StringIO()
        self.stderr_patcher = patch(
            "rocmplete.podman.sys.stderr",
            self.stderr,
        )
        self.stderr_patcher.start()
        self.addCleanup(self.stderr_patcher.stop)

    @patch("rocmplete.podman.shutil.which")
    @patch("rocmplete.podman.subprocess.run")
    def test_selinux_container_device_policy_reads_enforcing_boolean(
        self, run, which
    ):
        which.side_effect = lambda command: "/usr/bin/{}".format(command)
        enforcing = Mock(returncode=0, stdout="Enforcing\n")
        enabled = Mock(
            returncode=0,
            stdout="container_use_devices --> on\n",
        )
        run.side_effect = [enforcing, enabled]

        self.assertIs(podman.selinux_container_device_access(), True)
        self.assertEqual(
            run.call_args_list,
            [
                call(
                    ["/usr/bin/getenforce"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ),
                call(
                    ["/usr/bin/getsebool", "container_use_devices"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ),
            ],
        )

    @patch(
        "rocmplete.podman.selinux_container_device_access",
        return_value=False,
    )
    def test_disabled_selinux_container_devices_fail_with_host_action(
        self, device_access
    ):
        with self.assertRaisesRegex(
            LauncherError,
            r"sudo setsebool -P container_use_devices 1",
        ):
            podman.require_container_device_access()

    @patch(
        "rocmplete.podman.selinux_volume_suffix",
        return_value=":rw,Z",
    )
    def test_shared_selinux_suffix_uses_shared_mcs_label(self, suffix):
        self.assertEqual(
            podman.shared_selinux_volume_suffix(),
            ":rw,z",
        )

    @patch("rocmplete.podman.shutil.which")
    @patch("rocmplete.podman.subprocess.run")
    def test_shared_content_label_matches_runtime_mount(self, run, which):
        which.side_effect = lambda command: "/usr/bin/{}".format(command)
        run.side_effect = [
            Mock(returncode=0, stdout="Enforcing\n"),
            Mock(returncode=0, stderr=""),
        ]

        podman.prepare_shared_content_label(Path("/data/model.gguf"))

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    ["/usr/bin/getenforce"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                ),
                call(
                    [
                        "/usr/bin/chcon",
                        "--no-dereference",
                        "system_u:object_r:container_file_t:s0",
                        "--",
                        "/data/model.gguf",
                    ],
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=False,
                ),
            ],
        )

    @patch("rocmplete.podman.shutil.which", return_value=None)
    @patch("rocmplete.podman.subprocess.run")
    def test_shared_content_label_is_skipped_without_selinux(self, run, which):
        podman.prepare_shared_content_label(Path("/data/model.gguf"))

        run.assert_not_called()

    @patch("rocmplete.podman.subprocess.run")
    def test_managed_download_query_filters_exact_owned_prefix(self, run):
        run.return_value = Mock(
            returncode=0,
            stdout=(
                "rocmplete-comfyui\n"
                "rocmplete-download-bbbbbbbbbbbb\n"
                "rocmplete-download-aaaaaaaaaaaa\n"
            ),
            stderr="",
        )

        self.assertEqual(
            _query_managed_download_containers(),
            [
                "rocmplete-download-aaaaaaaaaaaa",
                "rocmplete-download-bbbbbbbbbbbb",
            ],
        )
        run.assert_called_once_with(
            ["podman", "ps", "--all", "--format", "{{.Names}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )

    @patch("rocmplete.podman.subprocess.run")
    def test_managed_container_query_uses_ownership_labels(self, run):
        run.return_value = Mock(
            returncode=0,
            stdout="rocmplete-llama-benchmark\n",
            stderr="",
        )

        self.assertEqual(
            podman.managed_container_names("llama-cpp"),
            ["rocmplete-llama-benchmark"],
        )
        run.assert_called_once_with(
            [
                "podman",
                "ps",
                "--all",
                "--filter",
                "label=io.github.fff7d1bc.rocmplete.managed=true",
                "--filter",
                "label=io.github.fff7d1bc.rocmplete.application=llama-cpp",
                "--format",
                "{{.Names}}",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
        )

    def test_managed_container_arguments_record_owner_and_role(self):
        self.assertEqual(
            podman.managed_container_arguments(
                application="llama-cpp", role="benchmark"
            ),
            [
                "--label",
                "io.github.fff7d1bc.rocmplete.managed=true",
                "--label",
                "io.github.fff7d1bc.rocmplete.role=benchmark",
                "--label",
                "io.github.fff7d1bc.rocmplete.application=llama-cpp",
            ],
        )

    @patch("rocmplete.podman.command_exists", return_value=False)
    def test_image_and_container_checks_report_missing_podman(self, command_exists):
        for operation in (
            lambda: podman.image_exists("localhost/test:image"),
            lambda: podman.container_exists("test-container"),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    LauncherError, "podman is not installed"
                ):
                    operation()

    @patch("rocmplete.podman.subprocess.run")
    def test_quiet_run_suppresses_only_stdout(self, run):
        run.return_value.returncode = 0

        self.assertEqual(podman.run_quiet_stdout(["podman", "rm", "name"]), 0)

        run.assert_called_once_with(
            ["podman", "rm", "name"],
            stdout=subprocess.DEVNULL,
            check=False,
        )

    @patch(
        "rocmplete.podman.capture",
        return_value="a" * 64,
    )
    def test_image_id_uses_structured_inspection(self, capture):
        self.assertEqual(
            podman.image_id("localhost/test:image"),
            "sha256:" + "a" * 64,
        )
        self.assertEqual(
            capture.call_args.args[0],
            [
                "podman",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                "localhost/test:image",
            ],
        )

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="Name:\tpython3\nUmask:\t0077\n",
    )
    def test_current_umask_uses_process_status(self, status):
        self.assertEqual(podman.current_umask(), "0077")
        status.assert_called_once_with("/proc/self/status", encoding="utf-8")

    @patch("rocmplete.podman._force_remove_container", return_value="")
    @patch("rocmplete.podman._request_container_removal", return_value="")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_interrupted_foreground_container_is_removed_and_reaped(
        self, popen, request_removal, force_remove
    ):
        process = Mock()
        process.wait.side_effect = [KeyboardInterrupt, 0]
        process.poll.return_value = None
        popen.return_value = process
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "rocmplete-comfyui",
            "image",
        ]

        with self.assertRaises(KeyboardInterrupt):
            podman.run_managed_foreground(
                command, "rocmplete-comfyui", "ComfyUI failed"
            )

        popen.assert_called_once_with(command)
        process.terminate.assert_called_once_with()
        request_removal.assert_called_once_with(
            "rocmplete-comfyui", stop_timeout=1
        )
        force_remove.assert_called_once_with(
            "rocmplete-comfyui", stop_timeout=1
        )

    @patch("rocmplete.podman._force_remove_container")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_successful_foreground_container_needs_no_forced_cleanup(
        self, popen, force_remove
    ):
        process = Mock()
        process.wait.return_value = 0
        popen.return_value = process
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "rocmplete-comfyui",
            "image",
        ]

        self.assertEqual(
            podman.run_managed_foreground(
                command, "rocmplete-comfyui", "ComfyUI failed"
            ),
            0,
        )

        popen.assert_called_once_with(command)
        force_remove.assert_not_called()

    @patch("rocmplete.podman._force_remove_container")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_failed_foreground_container_reports_child_status(
        self, popen, force_remove
    ):
        process = Mock()
        process.wait.return_value = 139
        popen.return_value = process
        command = [
            "podman",
            "run",
            "--rm",
            "--name",
            "rocmplete-llama-cpp",
            "image",
        ]

        with self.assertRaisesRegex(
            LauncherError,
            r"llama\.cpp CLI failed \(exit status 139\)",
        ):
            podman.run_managed_foreground(
                command,
                "rocmplete-llama-cpp",
                "llama.cpp CLI failed",
            )

        force_remove.assert_not_called()

    def test_foreground_container_rejects_detached_or_mismatched_commands(self):
        with self.assertRaisesRegex(LauncherError, "does not accept --detach"):
            podman.run_managed_foreground(
                [
                    "podman",
                    "run",
                    "--name",
                    "rocmplete-comfyui",
                    "--detach",
                    "image",
                ],
                "rocmplete-comfyui",
                "ComfyUI failed",
            )
        with self.assertRaisesRegex(LauncherError, "does not match"):
            podman.run_managed_foreground(
                [
                    "podman",
                    "run",
                    "--name",
                    "different-name",
                    "image",
                ],
                "rocmplete-comfyui",
                "ComfyUI failed",
            )

    @patch("rocmplete.podman.subprocess.run")
    def test_forced_cleanup_targets_one_exact_managed_container(self, run):
        removed = Mock(returncode=0, stderr="")
        absent = Mock(returncode=1)
        run.side_effect = [removed, absent]

        self.assertEqual(
            podman._force_remove_container(
                "rocmplete-download-0123456789ab",
                stop_timeout=0,
            ),
            "",
        )

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        "podman",
                        "rm",
                        "--force",
                        "--time",
                        "0",
                        "--ignore",
                        "rocmplete-download-0123456789ab",
                    ],
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=10.0,
                ),
                call(
                    [
                        "podman",
                        "container",
                        "exists",
                        "rocmplete-download-0123456789ab",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5.0,
                ),
            ],
        )

    @patch("rocmplete.podman.time.monotonic", side_effect=[0.0, 15.0])
    @patch("rocmplete.podman.subprocess.run")
    def test_forced_cleanup_reports_container_that_still_exists(
        self, run, monotonic
    ):
        run.side_effect = [
            Mock(returncode=0, stderr=""),
            Mock(returncode=0),
        ]

        self.assertEqual(
            podman._force_remove_container(
                "rocmplete-download-0123456789ab",
                stop_timeout=0,
            ),
            "container still exists 15 seconds after forced removal",
        )

    @patch("rocmplete.podman.time.sleep")
    @patch("rocmplete.podman.time.monotonic", side_effect=[0.0, 0.0])
    @patch("rocmplete.podman.subprocess.run")
    def test_forced_cleanup_waits_through_already_stopping_race(
        self, run, monotonic, sleep
    ):
        run.side_effect = [
            Mock(returncode=1, stderr="container is already stopping"),
            Mock(returncode=0),
            Mock(returncode=1),
        ]

        self.assertEqual(
            podman._force_remove_container(
                "rocmplete-download-0123456789ab",
                stop_timeout=0,
            ),
            "",
        )

        sleep.assert_called_once_with(0.25)

    @patch("rocmplete.podman._force_remove_container", return_value="")
    def test_public_removal_uses_requested_forced_timeout(self, force_remove):
        podman.remove_container("rocmplete-llama-benchmark", stop_timeout=0)

        force_remove.assert_called_once_with(
            "rocmplete-llama-benchmark", stop_timeout=0
        )

    @patch(
        "rocmplete.podman._force_remove_container",
        return_value="container still exists",
    )
    def test_public_removal_reports_forced_cleanup_failure(
        self, force_remove
    ):
        with self.assertRaisesRegex(
            LauncherError,
            "rocmplete-llama-benchmark.*container still exists",
        ):
            podman.remove_container(
                "rocmplete-llama-benchmark", stop_timeout=0
            )

    @patch("rocmplete.podman.signal.signal")
    @patch("rocmplete.podman.signal.getsignal")
    def test_cleanup_interrupt_mask_is_scoped(self, getsignal, set_signal):
        previous = Mock()
        getsignal.return_value = previous

        with podman._ignore_terminal_interrupts():
            pass

        self.assertEqual(
            set_signal.call_args_list,
            [
                call(signal.SIGINT, signal.SIG_IGN),
                call(signal.SIGINT, previous),
            ],
        )

    @patch("rocmplete.podman._force_remove_container", return_value="")
    @patch("rocmplete.podman._request_container_removal", return_value="")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_monitored_process_is_reaped_when_progress_fails(
        self, popen, request_removal, force_remove
    ):
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        popen.return_value = process

        def fail_progress():
            raise RuntimeError("cannot report progress")

        with self.assertRaisesRegex(RuntimeError, "cannot report"):
            podman.run_with_progress(["podman", "run"], fail_progress)

        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5.0)
        process.kill.assert_not_called()
        command = popen.call_args.args[0]
        container_name = command[command.index("--name") + 1]
        self.assertTrue(container_name.startswith("rocmplete-download-"))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        request_removal.assert_called_once_with(
            container_name, stop_timeout=0
        )
        force_remove.assert_called_once_with(
            container_name, stop_timeout=0
        )

    @patch("rocmplete.podman._force_remove_container", return_value="")
    @patch("rocmplete.podman._request_container_removal", return_value="")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_monitored_process_is_killed_if_termination_times_out(
        self, popen, request_removal, force_remove
    ):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [
            subprocess.TimeoutExpired("podman", 5.0),
            0,
        ]
        popen.return_value = process

        def interrupt_progress():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            podman.run_with_progress(
                ["podman", "run"],
                interrupt_progress,
            )

        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.wait.call_args_list[-1].args, ())
        self.assertIn("partial data will be kept", self.stderr.getvalue())
        container_name = popen.call_args.args[0][3]
        request_removal.assert_called_once_with(
            container_name, stop_timeout=0
        )
        force_remove.assert_called_once_with(
            container_name, stop_timeout=0
        )

    @patch("rocmplete.podman._force_remove_container", return_value="")
    @patch("rocmplete.podman._request_container_removal", return_value="")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_container_is_removed_even_if_podman_client_already_exited(
        self, popen, request_removal, force_remove
    ):
        process = Mock()
        process.poll.return_value = 130
        popen.return_value = process

        def interrupt_progress():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            podman.run_with_progress(
                ["podman", "run", "--rm", "image"],
                interrupt_progress,
            )

        process.terminate.assert_not_called()
        container_name = popen.call_args.args[0][3]
        request_removal.assert_called_once_with(
            container_name, stop_timeout=0
        )
        force_remove.assert_called_once_with(
            container_name, stop_timeout=0
        )

    @patch(
        "rocmplete.podman._force_remove_container",
        return_value="container is still running",
    )
    @patch("rocmplete.podman._request_container_removal", return_value="")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_cleanup_failure_is_reported_instead_of_leaking_silently(
        self, popen, request_removal, force_remove
    ):
        process = Mock()
        process.poll.return_value = 130
        popen.return_value = process

        with self.assertRaisesRegex(
            LauncherError, "cannot remove interrupted downloader"
        ):
            podman.run_with_progress(
                ["podman", "run", "--rm", "image"],
                Mock(side_effect=KeyboardInterrupt),
            )
        container_name = popen.call_args.args[0][3]
        request_removal.assert_called_once_with(
            container_name, stop_timeout=0
        )
        force_remove.assert_called_once_with(
            container_name, stop_timeout=0
        )

    @patch("rocmplete.podman._force_remove_container")
    @patch("rocmplete.podman.subprocess.Popen")
    def test_successful_monitored_process_needs_no_forced_cleanup(
        self, popen, force_remove
    ):
        process = Mock()
        process.wait.return_value = 0
        popen.return_value = process
        update = Mock()

        self.assertEqual(
            podman.run_with_progress(
                ["podman", "run", "--rm", "image"],
                update,
            ),
            0,
        )

        update.assert_called_once_with()
        force_remove.assert_not_called()
        self.assertIs(
            popen.call_args.kwargs["stdout"],
            subprocess.DEVNULL,
        )

    @patch("rocmplete.podman.subprocess.Popen")
    def test_monitored_download_refuses_existing_managed_container(
        self, popen
    ):
        self.download_containers.return_value = (
            "rocmplete-download-leftover",
        )

        with self.assertRaisesRegex(
            LauncherError,
            r"podman rm --force rocmplete-download-leftover",
        ):
            podman.run_with_progress(
                ["podman", "run", "--rm", "image"],
                Mock(),
            )

        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
