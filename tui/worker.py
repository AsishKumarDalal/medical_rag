"""Run the harness on a worker thread so the TUI never freezes.

Sessions, the model client and the GraphRAG backend are all created *inside* the
worker thread. SQLite connections are thread-bound, and wiring the backend can be
slow, so nothing in the UI thread ever touches them.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

from harness.cli import build_tools
from harness.config import Config
from harness.context import ContextManager, LLMSummarizer
from harness.llm import LLMClient
from harness.security import redact
from harness.session import SessionStore

from .chat import TuiChat

Event = dict[str, Any]
Emit = Callable[[Event], None]

HISTORY_LIMIT = 60


def _history(store: SessionStore, session_id: str) -> list[dict[str, str]]:
    """The visible part of a resumed session, oldest first."""
    messages = [
        {"role": message.role, "text": message.content}
        for message in store.load_messages(session_id)
        if message.role in ("user", "assistant") and message.content.strip()
    ]
    return messages[-HISTORY_LIMIT:]


class HarnessWorker:
    """Owns the ``Chat`` and processes ask/command jobs one at a time."""

    def __init__(
        self,
        config: Config,
        events: Emit,
        *,
        session_id: str | None = None,
    ) -> None:
        self.config = config
        self.events = events
        self.session_id = session_id
        self._jobs: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="harness-tui", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def ask(self, question: str) -> None:
        self._jobs.put(("ask", question))

    def command(self, line: str) -> None:
        self._jobs.put(("command", line))

    def stop(self) -> None:
        self._jobs.put(("stop", ""))
        self._thread.join(timeout=0.5)

    # -- worker -----------------------------------------------------------

    def _run(self) -> None:
        store: SessionStore | None = None
        try:
            store = SessionStore(self.config.db_path, sanitize=redact)
            client = LLMClient(self.config)
            context = ContextManager.from_config(
                self.config, summarizer=LLMSummarizer(client)
            )
            chat = TuiChat(
                config=self.config,
                client=client,
                store=store,
                tools=build_tools(self.config),
                context=context,
                session_id=self.session_id,
                events=self.events,
            )
        except Exception as exc:
            self.events({"kind": "fatal", "text": f"{type(exc).__name__}: {exc}"})
            if store is not None:
                store.close()
            return

        current = chat.session_id
        self.events(
            {
                "kind": "ready",
                "session": current,
                "model": self.config.model,
                "history": _history(store, current),
                "usage": store.usage_totals(current),
            }
        )

        try:
            while True:
                action, payload = self._jobs.get()
                if action == "stop":
                    break

                self.events({"kind": "busy", "value": True, "status": "thinking…"})
                try:
                    if action == "ask":
                        chat.ask(payload)
                    elif not chat.handle(payload):
                        self.events({"kind": "quit"})
                        break

                    if chat.session_id != current:
                        current = chat.session_id
                        self.events(
                            {
                                "kind": "session",
                                "id": current,
                                "history": _history(store, current),
                                "usage": store.usage_totals(current),
                            }
                        )
                except Exception as exc:
                    self.events({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
                finally:
                    self.events({"kind": "busy", "value": False})
        finally:
            store.close()
