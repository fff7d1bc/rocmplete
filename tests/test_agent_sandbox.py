import tempfile
import unittest
from pathlib import Path

from rocmplete.agent_sandbox import _runtime_resolver_target
from rocmplete.errors import LauncherError


class AgentSandboxResolverTests(unittest.TestCase):
    def test_regular_resolver_needs_no_runtime_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolver = root / "etc" / "resolv.conf"
            resolver.parent.mkdir()
            resolver.write_text("nameserver 192.0.2.1\n")

            self.assertIsNone(
                _runtime_resolver_target(resolver, root / "run")
            )

    def test_runtime_resolver_symlink_returns_exact_regular_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "run"
            target = runtime / "systemd" / "resolve" / "stub-resolv.conf"
            target.parent.mkdir(parents=True)
            target.write_text("nameserver 127.0.0.53\n")
            resolver = root / "etc" / "resolv.conf"
            resolver.parent.mkdir()
            resolver.symlink_to("../run/systemd/resolve/stub-resolv.conf")

            self.assertEqual(
                _runtime_resolver_target(resolver, runtime), target
            )

    def test_symlink_outside_runtime_needs_no_runtime_bind(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "etc" / "static-resolv.conf"
            target.parent.mkdir()
            target.write_text("nameserver 192.0.2.1\n")
            resolver = root / "etc" / "resolv.conf"
            resolver.symlink_to("static-resolv.conf")

            self.assertIsNone(
                _runtime_resolver_target(resolver, root / "run")
            )

    def test_broken_runtime_resolver_symlink_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            resolver = root / "etc" / "resolv.conf"
            resolver.parent.mkdir()
            resolver.symlink_to("../run/NetworkManager/resolv.conf")

            with self.assertRaisesRegex(
                LauncherError, "cannot resolve host resolver symlink"
            ):
                _runtime_resolver_target(resolver, root / "run")

    def test_runtime_resolver_target_must_be_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "run"
            target = runtime / "resolver"
            target.mkdir(parents=True)
            resolver = root / "etc" / "resolv.conf"
            resolver.parent.mkdir()
            resolver.symlink_to("../run/resolver")

            with self.assertRaisesRegex(
                LauncherError, "not a regular file"
            ):
                _runtime_resolver_target(resolver, runtime)


if __name__ == "__main__":
    unittest.main()
