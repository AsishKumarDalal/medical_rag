"""Pure text layout for the TUI — no curses in here, so it stays testable."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

PREFIXES = {
    "you": "you> ",
    "assistant": "harness> ",
    "tool": "tool> ",
    "notice": "  ",
    "error": "error> ",
}


@dataclass(frozen=True, slots=True)
class Entry:
    """One transcript entry. ``role`` selects its prefix and colour."""

    role: str
    text: str


def wrap_entry(entry: Entry, width: int) -> list[tuple[str, str]]:
    """Wrap one entry to ``width`` columns, returning ``(role, line)`` pairs.

    Explicit newlines are kept; continuation lines are indented under the prefix.
    """
    width = max(1, width)
    prefix = PREFIXES.get(entry.role, "")
    indent = " " * len(prefix)
    lines: list[tuple[str, str]] = []

    for paragraph in str(entry.text).splitlines() or [""]:
        lead = prefix if not lines else indent
        available = max(1, width - len(lead))
        chunks = textwrap.wrap(
            paragraph,
            width=available,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for chunk in chunks:
            lines.append((entry.role, lead + chunk))
            lead = indent

    return lines


def scroll_window(total: int, height: int, scroll: int) -> tuple[int, int, int]:
    """Return ``(start, end, scroll)`` for a bottom-anchored viewport.

    ``scroll`` is measured in lines above the bottom; 0 means pinned to the end.
    The returned scroll is clamped so callers can store it back.
    """
    height = max(0, height)
    limit = max(0, total - height)
    offset = min(max(0, scroll), limit)
    end = total - offset
    start = max(0, end - height)
    return start, end, offset
