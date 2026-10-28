"""A minimal curses front end for the medical harness.

    python -m tui                   # new session
    python -m tui --session <id>    # resume one
    python -m tui --db <path>       # another session database

Only the standard library is used. POSIX terminal only.
"""

from __future__ import annotations

import argparse
import curses
import queue
import sys
import time
from dataclasses import replace
from typing import Any, Sequence

from harness.config import Config, ConfigError

from .render import Entry, scroll_window, wrap_entry
from .worker import HarnessWorker

SPINNER = "|/-\\"
MAX_INPUT = 4_000
HISTORY = 100


class App:
    """Screen state and the key loop. All harness work happens in the worker."""

    def __init__(
        self,
        screen: "curses.window",
        *,
        config: Config,
        session_id: str | None = None,
    ) -> None:
        self.screen = screen
        self.config = config
        self.session_id = session_id or ""
        self.model = config.model
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.worker = HarnessWorker(config, self.events.put, session_id=session_id)

        self.entries: list[Entry] = []
        self.input = ""
        self.cursor = 0
        self.history: list[str] = []
        self.history_index: int | None = None
        self.draft = ""

        self.scroll = 0
        self.busy = True
        self.busy_since = time.monotonic()
        self.dead = False
        self.running = True
        self.status = "starting harness…"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.attrs: dict[str, int] = {}
        self.header_attr = curses.A_REVERSE

    # -- lifecycle --------------------------------------------------------

    def run(self) -> None:
        self._init_screen()
        self.worker.start()
        try:
            while self.running:
                self._drain()
                self._draw()
                try:
                    key = self.screen.get_wch()
                except curses.error:
                    continue  # timeout: redraw the spinner
                self._handle_key(key)
        finally:
            self.worker.stop()

    def _init_screen(self) -> None:
        self.screen.keypad(True)
        self.screen.timeout(100)
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        self._init_colors()

    def _init_colors(self) -> None:
        self.attrs = {
            "you": curses.A_BOLD,
            "assistant": 0,
            "tool": 0,
            "notice": curses.A_DIM,
            "error": curses.A_BOLD,
            "hint": curses.A_DIM,
            "input": curses.A_BOLD,
        }
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            try:
                curses.use_default_colors()
                background = -1
            except curses.error:
                background = 0
            pairs = (
                ("you", curses.COLOR_CYAN),
                ("assistant", curses.COLOR_WHITE),
                ("tool", curses.COLOR_YELLOW),
                ("notice", curses.COLOR_WHITE),
                ("error", curses.COLOR_RED),
                ("hint", curses.COLOR_WHITE),
                ("input", curses.COLOR_WHITE),
            )
            for number, (role, foreground) in enumerate(pairs, start=1):
                curses.init_pair(number, foreground, background)
                self.attrs[role] |= curses.color_pair(number)
            curses.init_pair(20, curses.COLOR_BLACK, curses.COLOR_CYAN)
            self.header_attr = curses.color_pair(20) | curses.A_BOLD
        except curses.error:
            self.attrs = {
                "you": curses.A_BOLD,
                "assistant": 0,
                "tool": 0,
                "notice": curses.A_DIM,
                "error": curses.A_BOLD,
                "hint": curses.A_DIM,
                "input": curses.A_BOLD,
            }
            self.header_attr = curses.A_REVERSE

    # -- events from the worker -------------------------------------------

    def _drain(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            self._event(event)

    def _event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")

        if kind == "ready":
            self.session_id = str(event.get("session") or "")
            self.model = str(event.get("model") or self.model)
            self.usage.update(event.get("usage") or {})
            self._replay(event.get("history") or [])
            self.busy = False
            self.status = "ready"
        elif kind == "session":
            self.session_id = str(event.get("id") or "")
            self.usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            self.usage.update(event.get("usage") or {})
            self._replay(event.get("history") or [])
        elif kind == "notice":
            self._append("notice", str(event.get("text") or ""))
        elif kind == "tool_call":
            self._append(
                "tool",
                f"calling {event.get('name', 'tool')}({event.get('arguments', '')})",
            )
            if self.busy:
                self.status = f"tool: {event.get('name', 'tool')}"
        elif kind == "tool_result":
            self._append("tool", f"{event.get('name', 'tool')}: {event.get('text', '')}")
            if self.busy:
                self.status = "thinking…"
        elif kind == "context":
            self._append("notice", f"[context] {event.get('text', '')}")
        elif kind == "answer":
            self._append("assistant", str(event.get("text") or ""))
        elif kind == "error":
            self._append("error", str(event.get("text") or "unknown error"))
            self.status = "error"
        elif kind == "usage":
            self.usage = {key: int(event.get(key, 0)) for key in self.usage}
        elif kind == "busy":
            self.busy = bool(event.get("value"))
            if self.busy:
                self.busy_since = time.monotonic()
                self.status = str(event.get("status") or "thinking…")
            elif self.status != "error":
                self.status = "ready"
        elif kind == "fatal":
            self.dead = True
            self.busy = False
            self.status = "startup failed"
            self._append("error", str(event.get("text") or "harness failed to start"))
        elif kind == "quit":
            self.running = False

    def _append(self, role: str, text: str) -> None:
        if text.strip():
            self.entries.append(Entry(role, text))

    def _replay(self, history: Sequence[dict[str, str]]) -> None:
        self.entries = []
        self.scroll = 0
        if history:
            self.entries.append(Entry("notice", f"session {self.session_id} (resumed)"))
            for item in history:
                role = item.get("role")
                text = str(item.get("text") or "")
                self.entries.append(Entry("you" if role == "user" else "assistant", text))
        else:
            self.entries.append(
                Entry(
                    "notice",
                    f"session {self.session_id or '(starting)'} — ask a medical question, "
                    "or type /help",
                )
            )

    # -- drawing ----------------------------------------------------------

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < 6 or width < 24:
            self._put(0, 0, "terminal too small", curses.A_BOLD)
            self.screen.refresh()
            return

        self._draw_header(width)
        self._draw_transcript(height, width)
        self._put(height - 3, 0, "-" * width, self.attrs.get("hint", 0))
        self._draw_input(height, width)
        self._draw_hint(height, width)
        self.screen.refresh()

    def _draw_header(self, width: int) -> None:
        left = " medical harness "
        if self.busy:
            spinner = SPINNER[int(time.monotonic() * 4) % len(SPINNER)]
            elapsed = time.monotonic() - self.busy_since
            left += f" {spinner} {self.status} {elapsed:.0f}s "
        elif self.status not in ("ready", ""):
            left += f" {self.status} "

        tokens = self.usage.get("total_tokens", 0)
        right = f"{self.model} | {tokens} tok | {self.session_id[:12]} "
        if len(left) + len(right) <= width:
            line = left + " " * (width - len(left) - len(right)) + right
        else:
            line = (left + "  " + right)[:width]
        self._put(0, 0, line, self.header_attr)

    def _draw_transcript(self, height: int, width: int) -> None:
        text_width = max(1, width - 1)
        lines = [
            (role, line)
            for entry in self.entries
            for role, line in wrap_entry(entry, text_width)
        ]
        start, end, self.scroll = scroll_window(
            len(lines), height - 4, self.scroll
        )
        for offset, (role, line) in enumerate(lines[start:end]):
            self._put(1 + offset, 0, line, self.attrs.get(role, 0))

    def _draw_input(self, height: int, width: int) -> None:
        prompt = "off> " if self.dead else "you> "
        available = max(1, width - len(prompt) - 1)
        start = max(0, self.cursor - available + 1)
        visible = self.input[start : start + available]
        self._put(height - 2, 0, prompt + visible, self.attrs.get("input", 0))

        if not self.dead:
            column = min(width - 2, len(prompt) + self.cursor - start)
            try:
                self.screen.move(height - 2, max(0, column))
            except curses.error:
                pass

    def _draw_hint(self, height: int, width: int) -> None:
        if self.dead:
            hint = "harness is not running - Ctrl-C to quit"
        elif self.busy:
            hint = "answering... | PgUp/PgDn scroll | Ctrl-C quit"
        else:
            hint = (
                "Enter send | PgUp/PgDn scroll | Up/Down history | /help | Ctrl-C quit"
            )
        self._put(height - 1, 0, hint, self.attrs.get("hint", 0))

    def _put(self, row: int, column: int, text: str, attr: int = 0) -> None:
        height, width = self.screen.getmaxyx()
        if row < 0 or row >= height or column >= width or not text:
            return
        try:
            self.screen.addnstr(row, column, text, max(0, width - column), attr)
        except curses.error:
            # The bottom-right cell cannot be written without scrolling.
            pass

    # -- input ------------------------------------------------------------

    def _handle_key(self, key: object) -> None:
        if key == curses.KEY_RESIZE:
            return
        if key == "\x03":  # Ctrl-C
            self.running = False
            return
        if key == curses.KEY_PPAGE:
            self._scroll_by(self._page())
            return
        if key == curses.KEY_NPAGE:
            self._scroll_by(-self._page())
            return
        if key == "\x07":  # Ctrl-G: jump to the newest line
            self.scroll = 0
            return
        if self.dead:
            return

        if key in (curses.KEY_BACKSPACE, "\x7f", "\b"):
            if self.cursor:
                self.input = self.input[: self.cursor - 1] + self.input[self.cursor :]
                self.cursor -= 1
                self.history_index = None
        elif key == curses.KEY_DC:
            if self.cursor < len(self.input):
                self.input = self.input[: self.cursor] + self.input[self.cursor + 1 :]
                self.history_index = None
        elif key == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == curses.KEY_RIGHT:
            self.cursor = min(len(self.input), self.cursor + 1)
        elif key in (curses.KEY_HOME, "\x01"):
            self.cursor = 0
        elif key in (curses.KEY_END, "\x05"):
            self.cursor = len(self.input)
        elif key == curses.KEY_UP:
            self._history(-1)
        elif key == curses.KEY_DOWN:
            self._history(1)
        elif key in ("\n", "\r", curses.KEY_ENTER):
            self._submit()
        elif key == "\x04":  # Ctrl-D on an empty line quits
            if not self.input:
                self.running = False
        elif key == "\x15":  # Ctrl-U clears the line
            self.input = self.input[self.cursor :]
            self.cursor = 0
            self.history_index = None
        elif key == "\x0c":  # Ctrl-L
            self.screen.clear()
        elif isinstance(key, str) and key.isprintable():
            self._insert(key)
        else:
            curses.beep()

    def _insert(self, character: str) -> None:
        if len(self.input) >= MAX_INPUT:
            curses.beep()
            return
        self.input = self.input[: self.cursor] + character + self.input[self.cursor :]
        self.cursor += 1
        self.history_index = None

    def _history(self, step: int) -> None:
        if not self.history:
            return
        if self.history_index is None:
            if step < 0:
                self.draft = self.input
                self.history_index = len(self.history) - 1
        else:
            self.history_index += step
            if self.history_index >= len(self.history):
                self.history_index = None
                self.input = self.draft
                self.cursor = len(self.input)
                return
            self.history_index = max(0, self.history_index)

        if self.history_index is not None:
            self.input = self.history[self.history_index]
            self.cursor = len(self.input)

    def _submit(self) -> None:
        if self.dead or self.busy:
            curses.beep()
            return
        text = self.input.strip()
        if not text:
            return

        self.entries.append(Entry("you", text))
        if not self.history or self.history[-1] != text:
            self.history.append(text)
            del self.history[:-HISTORY]
        self.input = ""
        self.cursor = 0
        self.history_index = None
        self.draft = ""
        self.scroll = 0
        self.busy = True
        self.busy_since = time.monotonic()
        self.status = "thinking…"

        if text.startswith("/"):
            self.worker.command(text)
        else:
            self.worker.ask(text)

    def _page(self) -> int:
        height, _ = self.screen.getmaxyx()
        return max(1, (height - 4) // 2)

    def _scroll_by(self, delta: int) -> None:
        height, width = self.screen.getmaxyx()
        text_width = max(1, width - 1)
        total = sum(len(wrap_entry(entry, text_width)) for entry in self.entries)
        _, _, self.scroll = scroll_window(total, max(0, height - 4), self.scroll + delta)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tui",
        description="A small full-screen terminal for the medical harness.",
    )
    parser.add_argument("--session", metavar="ID", help="resume this session")
    parser.add_argument("--db", metavar="PATH", help="session database (default: sessions.db)")
    args = parser.parse_args(argv)

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    if args.db:
        config = replace(config, db_path=args.db)

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "[tui] needs an interactive terminal; use `python -m harness` for pipes",
            file=sys.stderr,
        )
        return 2

    try:
        curses.wrapper(
            lambda screen: App(
                screen, config=config, session_id=args.session
            ).run()
        )
    except KeyboardInterrupt:
        return 130
    except curses.error as exc:
        print(f"[tui] cannot start: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
