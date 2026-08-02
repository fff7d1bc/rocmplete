import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from rocmplete.ui import (
    ColumnSpec,
    column_lines,
    display_width,
    finish_rewrite,
    next_actions,
    next_step,
    next_steps,
    print_numbered_choices,
    prompt,
    rewrite_line,
    state,
    style,
)


class _TerminalBuffer(io.StringIO):
    def isatty(self):
        return True


class UiTests(unittest.TestCase):
    def test_terminal_output_uses_semantic_ansi_styles(self):
        output = _TerminalBuffer()
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True
        ):
            self.assertIn("\033[", style("Ready", "success", output))
            self.assertIn("\033[", state("missing", 12, stream=output))
            self.assertIn(
                "\033[1;33m",
                state("TERMS+UNVERIFIED", stream=output),
            )
            next_step("./rocmplete run comfyui", stream=output)
        self.assertIn("\033[", output.getvalue())
        self.assertIn("./rocmplete run comfyui", output.getvalue())

    def test_redirected_output_remains_plain(self):
        output = io.StringIO()
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True
        ):
            self.assertEqual(style("Ready", "success", output), "Ready")
            self.assertEqual(state("missing", 12, stream=output), "missing     ")
            next_step("./rocmplete status", stream=output)
        self.assertEqual(
            output.getvalue(), "\nNext:\n    ./rocmplete status\n"
        )

    def test_next_steps_groups_copyable_commands(self):
        output = io.StringIO()
        next_steps(
            ("./rocmplete run comfyui", "./rocmplete status"),
            stream=output,
        )
        self.assertEqual(
            output.getvalue(),
            "\nNext:\n"
            "    ./rocmplete run comfyui\n"
            "    ./rocmplete status\n",
        )

    def test_next_actions_explains_each_command(self):
        output = io.StringIO()
        next_actions(
            (
                ("./rocmplete content install", "Guided installation."),
                ("./rocmplete content list", "Inspect only."),
            ),
            stream=output,
        )
        self.assertEqual(
            output.getvalue(),
            "\nNext:\n"
            "    ./rocmplete content install\n"
            "        Guided installation.\n"
            "    ./rocmplete content list\n"
            "        Inspect only.\n",
        )

    def test_no_color_and_dumb_terminal_disable_styles(self):
        output = _TerminalBuffer()
        with patch.dict(
            os.environ,
            {"TERM": "xterm-256color", "NO_COLOR": ""},
            clear=True,
        ):
            self.assertEqual(style("Warning", "warning", output), "Warning")
        with patch.dict(os.environ, {"TERM": "dumb"}, clear=True):
            self.assertEqual(style("Warning", "warning", output), "Warning")

    def test_prompt_uses_current_stdout_terminal(self):
        output = _TerminalBuffer()
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True
        ):
            with redirect_stdout(output):
                rendered = prompt("Continue? [y/N] ")
        self.assertTrue(rendered.startswith("\n\033["))
        self.assertEqual(rendered.count("\n"), 1)
        self.assertIn("\033[", rendered)
        self.assertIn("Continue? [y/N]", rendered)

    def test_plain_prompt_keeps_the_separating_blank_line(self):
        output = io.StringIO()
        with redirect_stdout(output):
            rendered = prompt("Continue? [y/N] ")
        self.assertEqual(rendered, "\nContinue? [y/N] ")

    def test_prompt_can_omit_the_separating_blank_line(self):
        output = io.StringIO()
        with redirect_stdout(output):
            rendered = prompt("URL: ", leading_blank=False)
        self.assertEqual(rendered, "URL: ")

    def test_columns_measure_all_rows_before_rendering(self):
        lines = column_lines(
            (
                ("short", "first description"),
                ("a-much-longer-value", "second description"),
            ),
            columns=(ColumnSpec(role="command"), ColumnSpec()),
        )
        self.assertEqual(
            lines,
            (
                "short                first description",
                "a-much-longer-value  second description",
            ),
        )

    def test_columns_measure_terminal_width_not_python_length(self):
        lines = column_lines(
            (("界", "wide"), ("aa", "ascii")),
            columns=(ColumnSpec(), ColumnSpec()),
        )
        prefixes = tuple(line.rsplit("  ", 1)[0] for line in lines)
        self.assertEqual(
            tuple(display_width(prefix) for prefix in prefixes),
            (2, 2),
        )

    def test_columns_apply_style_after_measuring_cells(self):
        output = _TerminalBuffer()
        with patch.dict(
            os.environ, {"TERM": "xterm-256color"}, clear=True
        ):
            lines = column_lines(
                (
                    ("short", "description one"),
                    ("a-longer-command", "description two"),
                ),
                columns=(ColumnSpec(role="command"), ColumnSpec()),
                stream=output,
            )
        self.assertTrue(all("\033[" in line for line in lines))
        prefixes = tuple(
            line.split("description", 1)[0] for line in lines
        )
        self.assertEqual(
            tuple(display_width(prefix) for prefix in prefixes),
            (18, 18),
        )

    def test_numbered_choices_align_dynamic_values_and_double_digits(self):
        output = io.StringIO()
        print_numbered_choices(
            tuple(
                (
                    "choice-{}".format(index),
                    "description {}".format(index),
                )
                for index in range(1, 11)
            ),
            stream=output,
        )
        lines = output.getvalue().splitlines()
        description_columns = tuple(
            line.index("description") for line in lines
        )
        self.assertEqual(len(set(description_columns)), 1)
        self.assertTrue(lines[0].startswith("   1)"))
        self.assertTrue(lines[9].startswith("  10)"))

    def test_rewrite_line_updates_tty_and_finishes_with_one_newline(self):
        output = _TerminalBuffer()
        width = rewrite_line("Progress: 50%", stream=output)
        rewrite_line(
            "Done",
            previous_width=width,
            complete=True,
            stream=output,
        )
        rendered = output.getvalue()
        self.assertEqual(rendered.count("\r"), 2)
        self.assertEqual(rendered.count("\n"), 1)
        self.assertIn("\rDone         \n", rendered)

    def test_rewrite_line_keeps_redirected_updates_line_delimited(self):
        output = io.StringIO()
        width = rewrite_line("Progress: 50%", stream=output)
        rewrite_line(
            "Progress: 100%",
            previous_width=width,
            complete=True,
            stream=output,
        )
        self.assertEqual(
            output.getvalue(),
            "Progress: 50%\nProgress: 100%\n",
        )

    def test_finish_rewrite_only_ends_a_terminal_line(self):
        terminal = _TerminalBuffer()
        redirected = io.StringIO()
        rewrite_line("Progress", stream=terminal)
        finish_rewrite(stream=terminal)
        finish_rewrite(stream=redirected)
        self.assertEqual(terminal.getvalue(), "\rProgress\n")
        self.assertEqual(redirected.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
