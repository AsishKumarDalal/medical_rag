"""The harness ``Chat`` with structured events instead of printed lines.

The TUI paints from dict events so it can choose colours and layout. Everything
else — sessions, context, tools, the loop — still comes from ``harness``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from harness.cli import Chat
from harness.loop import TurnResult

Event = dict[str, Any]
Emit = Callable[[Event], None]


def compact(value: object, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TuiChat(Chat):
    """A ``Chat`` whose output is published as events for the TUI."""

    def __init__(self, *, events: Emit, **kwargs: Any) -> None:
        self.events = events
        # Command output (``/help``, ``/sessions``, …) uses Chat.out.
        kwargs["out"] = self._notice
        super().__init__(**kwargs)

    def _notice(self, line: str) -> None:
        self.events({"kind": "notice", "text": line})

    def _report(self, result: TurnResult, before: Sequence[object]) -> None:
        for message in result.messages[len(before) :]:
            if message.tool_calls:
                for call in message.tool_calls:
                    arguments = call.raw_arguments or json.dumps(call.arguments)
                    self.events(
                        {
                            "kind": "tool_call",
                            "name": call.name,
                            "arguments": compact(arguments),
                        }
                    )
            elif message.role == "tool":
                self.events(
                    {
                        "kind": "tool_result",
                        "name": message.name or "tool",
                        "text": compact(message.content),
                    }
                )

        for event in result.context_events:
            self.events(
                {
                    "kind": "context",
                    "text": (
                        f"{event.kind}: {event.detail} "
                        f"({event.before_tokens} -> {event.after_tokens} tokens)"
                    ),
                }
            )

        if result.stop_reason == "answer":
            self.events({"kind": "answer", "text": result.final.strip() or "(empty answer)"})
        elif result.stop_reason == "max_iterations":
            self.events({"kind": "error", "text": result.error or "stopped without an answer"})
        else:
            self.events({"kind": "error", "text": result.error or "model call failed"})

        usage = self.store.usage_totals(self.session_id)
        if usage.get("calls"):
            self.events({"kind": "usage", **usage})
