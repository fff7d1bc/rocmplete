import tempfile
import unittest
from pathlib import Path

from rocmplete.config import (
    APPLICATIONS,
    APPLICATION_NAMES,
    BUILD_APPLICATIONS,
    LOG_APPLICATIONS,
    SHELL_APPLICATIONS,
    WEB_APPLICATIONS,
    default_data_dir,
    environment_value,
    is_loopback_address,
    reject_managed_comfy_args,
    selected_data_dir,
    validate_port,
    validate_listen_address,
    validate_profile,
    version_at_least,
)
from rocmplete.errors import LauncherError


class ConfigTests(unittest.TestCase):
    def test_application_registry_is_the_capability_source_of_truth(self):
        self.assertEqual(tuple(APPLICATIONS), APPLICATION_NAMES)
        self.assertEqual(BUILD_APPLICATIONS, APPLICATION_NAMES)
        self.assertEqual(SHELL_APPLICATIONS, APPLICATION_NAMES)
        self.assertEqual(
            LOG_APPLICATIONS, ("comfyui", "llama-cpp", "dwarfstar")
        )
        self.assertEqual(
            WEB_APPLICATIONS, ("comfyui", "llama-cpp", "dwarfstar")
        )
        for identifier, application in APPLICATIONS.items():
            self.assertEqual(application.identifier, identifier)
            self.assertTrue(application.image.startswith("localhost/rocmplete:"))
            self.assertTrue(application.container_name.startswith("rocmplete-"))
            self.assertTrue(application.after_build)
            self.assertTrue(application.after_content)
        self.assertFalse(APPLICATIONS["llama-cpp"].shared_pytorch_base)
        self.assertEqual(
            APPLICATIONS["llama-cpp"].after_build,
            "./rocmplete content install llama-cpp qwen3.8",
        )
        self.assertEqual(
            APPLICATIONS["llama-cpp"].after_content,
            "./rocmplete run llama-cpp server "
            "--preset qwen3.8-27b-mtp-ud-q8-k-xl",
        )
        self.assertFalse(APPLICATIONS["dwarfstar"].shared_pytorch_base)
        self.assertFalse(APPLICATIONS["dwarfstar"].multi_gpu)

    def test_default_data_dir_uses_xdg_data_home(self):
        self.assertEqual(
            default_data_dir({"HOME": "/home/test", "XDG_DATA_HOME": "/data"}),
            Path("/data/rocmplete"),
        )

    def test_default_data_dir_falls_back_to_home(self):
        self.assertEqual(
            default_data_dir({"HOME": "/home/test"}),
            Path("/home/test/.local/share/rocmplete"),
        )

    def test_toml_config_selects_absolute_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / ".config" / "rocmplete" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text(
                '[storage]\ndata_dir = "/mnt/ai/rocmplete"\n'
            )
            self.assertEqual(
                default_data_dir(
                    {
                        "HOME": str(home),
                        "XDG_DATA_HOME": "/default-data",
                    }
                ),
                Path("/mnt/ai/rocmplete"),
            )

    def test_xdg_config_home_selects_configuration_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_home = root / "configuration"
            config = config_home / "rocmplete" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text(
                '[storage]\ndata_dir = "/storage/rocmplete"\n'
            )
            self.assertEqual(
                default_data_dir(
                    {
                        "HOME": "/home/test",
                        "XDG_CONFIG_HOME": str(config_home),
                    }
                ),
                Path("/storage/rocmplete"),
            )

    def test_config_is_strict_and_requires_absolute_data_directory(self):
        cases = (
            ("not = [valid", "cannot read ROCmplete configuration"),
            ('[storage]\ndata_dir = "relative"\n', "must be an absolute"),
            ('[storage]\ndata_root = "/data"\n', "unknown.*setting"),
            ('[runtime]\nprofile = "auto"\n', "unknown.*section"),
        )
        for contents, message in cases:
            with self.subTest(contents=contents):
                with tempfile.TemporaryDirectory() as directory:
                    home = Path(directory)
                    config = (
                        home / ".config" / "rocmplete" / "config.toml"
                    )
                    config.parent.mkdir(parents=True)
                    config.write_text(contents)
                    with self.assertRaisesRegex(LauncherError, message):
                        default_data_dir({"HOME": str(home)})

    def test_data_directory_precedence_is_cli_environment_config_default(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            config = home / ".config" / "rocmplete" / "config.toml"
            config.parent.mkdir(parents=True)
            config.write_text(
                '[storage]\ndata_dir = "/configured"\n'
            )
            env = {
                "HOME": str(home),
                "ROCMLETE_DATA_DIR": "/environment",
            }
            self.assertEqual(
                selected_data_dir("/command-line", env),
                Path("/command-line"),
            )
            self.assertEqual(
                selected_data_dir(None, env), Path("/environment")
            )
            self.assertEqual(
                selected_data_dir(None, {"HOME": str(home)}),
                Path("/configured"),
            )

            config.write_text("invalid = [toml")
            self.assertEqual(
                selected_data_dir("/command-line", env),
                Path("/command-line"),
            )
            self.assertEqual(
                selected_data_dir(None, env), Path("/environment")
            )

    def test_missing_config_does_not_create_configuration_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            self.assertEqual(
                default_data_dir({"HOME": str(home)}),
                home / ".local" / "share" / "rocmplete",
            )
            self.assertFalse((home / ".config").exists())

    def test_environment_value_uses_rocmplete_namespace(self):
        self.assertEqual(
            environment_value(
                {"ROCMLETE_PROFILE": "rdna4"},
                "PROFILE",
            ),
            "rdna4",
        )

    def test_relative_xdg_data_home_is_rejected(self):
        with self.assertRaisesRegex(LauncherError, "must be an absolute"):
            default_data_dir({"HOME": "/home/test", "XDG_DATA_HOME": "relative"})

    def test_relative_xdg_config_home_is_rejected(self):
        with self.assertRaisesRegex(LauncherError, "must be an absolute"):
            default_data_dir(
                {"HOME": "/home/test", "XDG_CONFIG_HOME": "relative"}
            )

    def test_port_validation(self):
        self.assertEqual(validate_port("8188"), 8188)
        for value in ("0", "65536", "eight", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(LauncherError):
                    validate_port(value)

    def test_listen_address_validation(self):
        for value, expected in (
            ("127.0.0.1", "127.0.0.1"),
            ("0.0.0.0", "0.0.0.0"),
            ("100.64.12.34", "100.64.12.34"),
            ("::1", "::1"),
        ):
            with self.subTest(value=value):
                self.assertEqual(validate_listen_address(value), expected)
        for value in ("localhost", "", "100.64.1.999"):
            with self.subTest(value=value):
                with self.assertRaises(LauncherError):
                    validate_listen_address(value)
        self.assertTrue(is_loopback_address("127.0.0.2"))
        self.assertTrue(is_loopback_address("::1"))
        self.assertFalse(is_loopback_address("100.64.12.34"))

    def test_profile_validation(self):
        for profile in (
            "auto",
            "rdna4",
            "strix-halo",
            "strix-point",
            "cpu",
        ):
            self.assertEqual(validate_profile(profile), profile)
        for architecture in ("gfx1200", "gfx1201"):
            with self.subTest(architecture=architecture):
                with self.assertRaises(LauncherError):
                    validate_profile(architecture)

    def test_launcher_owned_comfy_options_are_rejected(self):
        for argument in ("--listen", "--port=9000", "--cpu"):
            with self.subTest(argument=argument):
                with self.assertRaises(LauncherError):
                    reject_managed_comfy_args([argument])
        reject_managed_comfy_args(["--lowvram"])

    def test_version_comparison(self):
        self.assertTrue(version_at_least("6.18.4", "6.18.4"))
        self.assertTrue(version_at_least("7.0.0", "6.18.4"))
        self.assertFalse(version_at_least("6.18.3", "6.18.4"))


if __name__ == "__main__":
    unittest.main()
