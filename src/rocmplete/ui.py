"""Small, dependency-free helpers for human-oriented terminal output."""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, TextIO, Tuple, Union


_RESET = "\033[0m"
_ANSI_STYLE = re.compile(r"\033\[[0-9;]*m")
_STYLES = {
    "heading": "\033[1;36m",
    "command": "\033[1;36m",
    "success": "\033[1;32m",
    "warning": "\033[1;33m",
    "error": "\033[1;31m",
    "info": "\033[1;34m",
    "muted": "\033[2m",
    "label": "\033[1m",
    "prompt": "\033[1;36m",
}

_SUCCESS_STATES = {
    "available",
    "built",
    "complete",
    "identical",
    "installed",
    "read/write",
    "ready",
    "running",
    "verified",
    "writable",
    "yes",
}
_WARNING_STATES = {
    "download",
    "link-missing",
    "link-mismatch",
    "missing",
    "partial",
    "repair link",
    "terms",
    "unverified",
}
_ERROR_STATES = {
    "broken",
    "conflict",
    "error",
    "failed",
    "hash-mismatch",
    "insufficient access",
    "link mismatch",
    "no access",
    "not a directory",
    "size-mismatch",
    "user-file",
}
_MUTED_STATES = {"absent", "not present"}


@dataclass(frozen=True)
class ColumnSpec:
    """Rendering policy for one measured output column."""

    align: str = "<"
    role: Optional[str] = None
    min_width: int = 0

    def __post_init__(self) -> None:
        if self.align not in ("<", ">"):
            raise ValueError("column alignment must be '<' or '>'")
        if self.min_width < 0:
            raise ValueError("column minimum width must not be negative")


def _output_stream(stream: Optional[TextIO]) -> TextIO:
    return sys.stdout if stream is None else stream


def terminal_output(stream: Optional[TextIO] = None) -> bool:
    """Return whether output can safely use terminal line rewriting."""
    output = _output_stream(stream)
    try:
        return bool(output.isatty())
    except (AttributeError, OSError):
        return False


def color_enabled(stream: Optional[TextIO] = None) -> bool:
    """Return whether ANSI styling is appropriate for this output stream."""
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb":
        return False
    return terminal_output(stream)


def style(text: object, role: str, stream: Optional[TextIO] = None) -> str:
    """Style text for a semantic role, or return plain text off-terminal."""
    rendered = str(text)
    if role not in _STYLES:
        raise ValueError("unknown output role: {}".format(role))
    if not color_enabled(stream):
        return rendered
    return "{}{}{}".format(_STYLES[role], rendered, _RESET)


def state_role(value: str) -> str:
    """Map a human-facing state to its semantic output role."""
    normalized = value.strip().lower()
    if normalized in _SUCCESS_STATES:
        return "success"
    if normalized in _ERROR_STATES:
        return "error"
    if normalized in _MUTED_STATES:
        return "muted"
    if (
        normalized in _WARNING_STATES
        or "terms" in normalized
        or "unverified" in normalized
    ):
        return "warning"
    return "label"


def state(
    value: str,
    width: int = 0,
    align: str = "<",
    stream: Optional[TextIO] = None,
) -> str:
    """Format and style a state without letting ANSI bytes affect alignment."""
    if align not in ("<", ">"):
        raise ValueError("state alignment must be '<' or '>'")
    rendered = (
        "{:{align}{width}}".format(value, align=align, width=width)
        if width
        else value
    )
    return style(rendered, state_role(value), stream)


def display_width(value: object) -> int:
    """Return the terminal-cell width of plain or ANSI-styled text."""
    rendered = _ANSI_STYLE.sub("", str(value))
    width = 0
    for character in rendered:
        if unicodedata.combining(character):
            continue
        if unicodedata.category(character) in ("Cf", "Me", "Mn"):
            continue
        width += (
            2
            if unicodedata.east_asian_width(character) in ("F", "W")
            else 1
        )
    return width


def _pad_column(value: str, width: int, align: str) -> str:
    padding = " " * max(0, width - display_width(value))
    return "{}{}".format(padding, value) if align == ">" else "{}{}".format(
        value, padding
    )


def column_lines(
    rows: Iterable[Sequence[object]],
    columns: Optional[Sequence[ColumnSpec]] = None,
    indent: str = "",
    separators: Union[str, Sequence[str]] = "  ",
    stream: Optional[TextIO] = None,
) -> Tuple[str, ...]:
    """Measure complete rows, then render aligned terminal columns."""
    output = _output_stream(stream)
    materialized = tuple(tuple(str(cell) for cell in row) for row in rows)
    if not materialized:
        return ()
    column_count = len(materialized[0])
    if column_count == 0:
        raise ValueError("column rows must not be empty")
    if any(len(row) != column_count for row in materialized):
        raise ValueError("column rows must have the same number of cells")
    specs = (
        tuple(columns)
        if columns is not None
        else tuple(ColumnSpec() for _ in range(column_count))
    )
    if len(specs) != column_count:
        raise ValueError("column specs must match the row width")
    if isinstance(separators, str):
        gaps = (separators,) * (column_count - 1)
    else:
        gaps = tuple(separators)
    if len(gaps) != column_count - 1:
        raise ValueError("column separators must fit between every column")
    widths = tuple(
        max(
            spec.min_width,
            max(display_width(row[index]) for row in materialized),
        )
        for index, spec in enumerate(specs)
    )
    lines = []
    for row in materialized:
        rendered = []
        for index, (cell, spec, width) in enumerate(
            zip(row, specs, widths)
        ):
            # A left-aligned final column needs no trailing padding.
            measured = (
                cell
                if index == column_count - 1 and spec.align == "<"
                else _pad_column(cell, width, spec.align)
            )
            rendered.append(
                style(measured, spec.role, output)
                if spec.role is not None
                else measured
            )
        line = rendered[0]
        for gap, cell in zip(gaps, rendered[1:]):
            line += gap + cell
        lines.append(indent + line)
    return tuple(lines)


def print_columns(
    rows: Iterable[Sequence[object]],
    columns: Optional[Sequence[ColumnSpec]] = None,
    indent: str = "",
    separators: Union[str, Sequence[str]] = "  ",
    stream: Optional[TextIO] = None,
) -> None:
    """Print a complete measured column block."""
    output = _output_stream(stream)
    for line in column_lines(
        rows,
        columns=columns,
        indent=indent,
        separators=separators,
        stream=output,
    ):
        print(line, file=output)


def print_numbered_choices(
    rows: Iterable[Sequence[object]],
    columns: Optional[Sequence[ColumnSpec]] = None,
    stream: Optional[TextIO] = None,
) -> None:
    """Print consistently numbered, width-aware interactive choices."""
    materialized = tuple(tuple(row) for row in rows)
    if not materialized:
        return
    data_width = len(materialized[0])
    if data_width == 0 or any(len(row) != data_width for row in materialized):
        raise ValueError("choice rows must have one consistent row width")
    specs = (
        tuple(columns)
        if columns is not None
        else (ColumnSpec(role="command"),)
        + tuple(ColumnSpec() for _ in range(data_width - 1))
    )
    if len(specs) != data_width:
        raise ValueError("choice column specs must match the row width")
    numbered = tuple(
        ("{})".format(index),) + row
        for index, row in enumerate(materialized, 1)
    )
    print_columns(
        numbered,
        columns=(ColumnSpec(align=">", min_width=3),) + specs,
        indent="  ",
        separators=(" ",) + ("  ",) * (data_width - 1),
        stream=stream,
    )


def prompt(message: str, *, leading_blank: bool = True) -> str:
    """Format an interactive prompt, optionally separated from prior output."""
    prefix = "\n" if leading_blank else ""
    return "{}{}".format(prefix, style(message, "prompt", sys.stdout))


def next_step(command: str, stream: Optional[TextIO] = None) -> None:
    """Print one copyable next action with a stable plain-text fallback."""
    next_steps((command,), stream=stream)


def next_steps(
    commands: Iterable[str], stream: Optional[TextIO] = None
) -> None:
    """Print copyable next actions as a visually separate output block."""
    next_actions(
        ((command, "") for command in commands),
        stream=stream,
    )


def next_actions(
    actions: Iterable[Tuple[str, str]],
    stream: Optional[TextIO] = None,
) -> None:
    """Print copyable next actions with optional explanatory text."""
    output = _output_stream(stream)
    actions = tuple(actions)
    if not actions:
        return
    print(file=output)
    print(style("Next:", "heading", output), file=output)
    for command, description in actions:
        print(
            "    {}".format(style(command, "command", output)),
            file=output,
        )
        if description:
            print(
                "        {}".format(
                    style(description, "muted", output)
                ),
                file=output,
            )


def rewrite_line(
    message: str,
    previous_width: int = 0,
    complete: bool = False,
    stream: Optional[TextIO] = None,
) -> int:
    """Rewrite one TTY line, while retaining normal lines in redirected output."""
    output = _output_stream(stream)
    visible_width = len(_ANSI_STYLE.sub("", message))
    if not terminal_output(output):
        print(message, file=output, flush=True)
        return visible_width
    padding = " " * max(0, previous_width - visible_width)
    print(
        "\r{}{}".format(message, padding),
        end="\n" if complete else "",
        file=output,
        flush=True,
    )
    return visible_width


def finish_rewrite(stream: Optional[TextIO] = None) -> None:
    """End an active rewritten TTY line without adding log output."""
    output = _output_stream(stream)
    if terminal_output(output):
        print(file=output, flush=True)
