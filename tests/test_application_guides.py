import io
import os
import shlex
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from rocmplete.application_guides import (
    APPLICATION_GUIDES,
    guide_commands,
    print_application_guide,
)
from rocmplete.cli_parser import parse_arguments
from rocmplete.config import APPLICATIONS, APPLICATION_NAMES
from rocmplete.recipes import application_recipes


class _TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


class ApplicationGuideTests(unittest.TestCase):
    def test_every_application_has_one_guide(self):
        self.assertEqual(tuple(APPLICATION_GUIDES), APPLICATION_NAMES)
        for identifier, guide in APPLICATION_GUIDES.items():
            with self.subTest(application=identifier):
                self.assertEqual(guide.application, identifier)
                self.assertTrue(guide.title)
                self.assertTrue(guide.summary)
                self.assertTrue(guide.reference.startswith("guide/"))

    def test_guides_include_every_application_recipe(self):
        for application, guide in APPLICATION_GUIDES.items():
            commands = set(guide_commands(guide))
            for recipe in application_recipes(application):
                with self.subTest(
                    application=application,
                    recipe=recipe.identifier,
                ):
                    self.assertIn(
                        "./rocmplete content install {} {}".format(
                            application, recipe.identifier
                        ),
                        commands,
                    )

    def test_every_rocmplete_guide_command_still_parses(self):
        for application, guide in APPLICATION_GUIDES.items():
            for command in guide_commands(guide):
                with self.subTest(
                    application=application,
                    command=command,
                ), redirect_stdout(io.StringIO()), redirect_stderr(
                    io.StringIO()
                ):
                    try:
                        _, arguments = parse_arguments(
                            shlex.split(command)[1:]
                        )
                    except SystemExit as result:
                        self.assertEqual(result.code, 0)
                    else:
                        self.assertIsNotNone(arguments.command)

    def test_web_guide_urls_use_registered_ports(self):
        for application in ("comfyui",):
            guide = APPLICATION_GUIDES[application]
            text = " ".join(
                section_text
                for section in guide.sections
                for section_text in section.paragraphs
            )
            self.assertIn(
                "127.0.0.1:{}".format(APPLICATIONS[application].port),
                text,
            )

    def test_dwarfstar_guide_explains_bind_and_model_boundaries(self):
        guide = APPLICATION_GUIDES["dwarfstar"]
        text = " ".join(
            section_text
            for section in guide.sections
            for section_text in section.paragraphs
        )
        commands = guide_commands(guide)
        self.assertIn("private container namespace", text)
        self.assertIn("host publication remains 127.0.0.1", text)
        self.assertIn("Without --model", text)
        self.assertIn("3.9 generated tokens per second", text)
        self.assertIn("feasibility observation", text)
        self.assertIn(
            "./rocmplete run dwarfstar server --model "
            "/path/to/deepseek-v4.gguf",
            commands,
        )

    def test_terminal_guide_uses_semantic_color_roles(self):
        output = _TerminalBuffer()
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True
        ), redirect_stdout(output):
            self.assertEqual(print_application_guide("llama-cpp"), 0)

        text = output.getvalue()
        normalized = " ".join(text.split())
        self.assertIn("\033[1;36mllama.cpp\033[0m", text)
        self.assertIn("\033[1;32mStart here\033[0m", text)
        self.assertIn(
            "\033[1;34mServe one model\033[0m",
            text,
        )
        self.assertIn("Use an OpenAI-compatible client for automation", text)
        self.assertNotIn("./rocmplete client", text)
        self.assertIn("qwen3.6 recipe installs dense 27B MTP Q8_0", text)
        self.assertIn(
            "separate ornith and kat-coder recipes install the official Ornith",
            text,
        )
        self.assertIn(
            "laguna-xs-2.1 recipe installs Poolside's official", text
        )
        self.assertIn(
            "Qwen3.6 35B-A3B MTP Q8_K_XL preset is the recommended "
            "OpenCode starting point",
            normalized,
        )
        self.assertIn(
            "Matching dense and sparse non-MTP controls remain", normalized
        )
        self.assertIn("non-MTP control", normalized)
        self.assertIn("MTP proposes and verifies extra tokens", text)
        self.assertIn("not a reasoning mode", text)
        self.assertIn(
            "Qwen3.6, Ornith, KAT-Coder, Gemma 4, and both Laguna "
            "families start at their native 256K",
            normalized,
        )
        self.assertIn("--context 131072", text)
        self.assertIn("omit it for a mixed-model router", normalized)
        self.assertIn("Tool-using clients", text)
        self.assertIn("embedded Jinja", text)
        self.assertIn("./rocmplete agent opencode", text)
        self.assertIn("bin/opencode", text)
        self.assertIn("./rocmplete agent pi", text)
        self.assertIn("bin/pi", text)
        self.assertIn("./rocmplete agent omp", text)
        self.assertIn("bin/omp", text)
        self.assertIn("./rocmplete agent maki", text)
        self.assertIn("bin/maki", text)
        self.assertIn("Pi package commands such as install", normalized)
        self.assertIn("OMP is a separate Pi fork", normalized)
        self.assertIn("trusted executable inputs", normalized)
        self.assertIn("content install llama-cpp qwen3.6", text)
        self.assertIn("content install llama-cpp shisa-v2.1", text)
        self.assertIn("Shisa V2.1", text)
        self.assertIn("starts at 16K", text)
        self.assertIn("content install llama-cpp all --dry-run", text)
        self.assertIn("Build and Plan ask before edits", text)
        self.assertIn("auto-approve", text)
        self.assertIn("Investigate agent", text)
        self.assertIn("hard read-only", text)
        self.assertIn("hidden read-only", text)
        self.assertIn("separate child sessions", text)
        self.assertIn("disabled choice turns thinking off", text)
        self.assertIn("Pi uses Shift+Tab or /settings", normalized)
        self.assertIn("OMP accepts --thinking", normalized)
        self.assertIn("Maki uses /thinking", normalized)
        self.assertIn("ctrl+t", text)
        self.assertIn("/variants", text)
        self.assertNotIn("OPENCODE_CONFIG", text)
        self.assertNotIn("OPENCODE_TUI_CONFIG", text)
        self.assertIn("\033[1;33mNetwork access\033[0m", text)
        self.assertIn(
            "\033[1;36m./rocmplete build llama-cpp\033[0m",
            text,
        )
        self.assertIn("\033[2mGGUF terminal chat", text)

    def test_redirected_guide_output_stays_plain(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(print_application_guide("llama-cpp"), 0)

        self.assertNotIn("\033[", output.getvalue())


if __name__ == "__main__":
    unittest.main()
