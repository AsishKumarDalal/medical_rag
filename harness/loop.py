"""The agent loop: model → tool calls → results → model, until it answers.

One :func:`run_turn` call is one user turn. It never raises for model or tool
failures: those come back as a stop reason, so the caller (CLI, tests, later the
session recorder) always gets a transcript it can persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from .config import Config
from .context import ContextEvent, ContextManager
from .llm import LLMError, SupportsComplete
from .messages import Message
from .tools.base import ToolResult
from .tools.registry import ToolRegistry

StopReason = Literal["answer", "max_iterations", "llm_error"]


class TurnRecorder(Protocol):
    """Durable write path used during a turn (implemented by SessionRecorder)."""

    def message(self, message: Message) -> None: ...

    def context_event(self, event: ContextEvent) -> None: ...

    def usage(self, usage: dict[str, int]) -> None: ...


@dataclass(slots=True)
class TurnResult:
    """Everything one turn produced."""

    messages: list[Message] = field(default_factory=list)
    final: str = ""
    stop_reason: StopReason = "answer"
    iterations: int = 0
    error: str | None = None
    context_events: list[ContextEvent] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.stop_reason == "answer"


def run_turn(
    conversation: Sequence[Message],
    *,
    client: SupportsComplete,
    tools: ToolRegistry | None = None,
    max_iterations: int | None = None,
    config: Config | None = None,
    context: ContextManager | None = None,
    recorder: TurnRecorder | None = None,
) -> TurnResult:
    """Run one turn to a final answer, an iteration cap, or a model failure.

    ``conversation`` is copied; the returned ``messages`` is the full transcript,
    input messages first, ready to be stored as the next conversation state. If a
    context manager is given, the transcript may be compressed and the request is
    the possibly-stubbed view of it.
    """
    limit = max_iterations or (config.max_iterations if config else 12)
    messages = list(conversation)
    specs = tools.specs() if tools else None
    events: list[ContextEvent] = []
    steps = 0

    while steps < limit:
        request = messages
        if context is not None:
            budgeted = context.apply(messages)
            messages = budgeted.conversation
            request = budgeted.request
            events.extend(budgeted.events)
            if recorder is not None:
                for event in budgeted.events:
                    recorder.context_event(event)

        try:
            response = client.complete(request, tools=specs)
        except LLMError as exc:
            return TurnResult(
                messages=messages,
                stop_reason="llm_error",
                iterations=steps,
                error=str(exc),
                context_events=events,
            )

        steps += 1
        if recorder is not None and response.usage:
            recorder.usage(response.usage)
        messages.append(
            Message(role="assistant", content=response.content, tool_calls=list(response.tool_calls))
        )
        if recorder is not None:
            recorder.message(messages[-1])

        if not response.tool_calls:
            return TurnResult(
                messages=messages,
                final=response.content,
                stop_reason="answer",
                iterations=steps,
                context_events=events,
            )

        for call in response.tool_calls:
            if tools is None:
                result = ToolResult.error(f"no tools are registered; cannot call {call.name!r}")
            else:
                result = tools.execute(call)
            messages.append(
                Message(
                    role="tool",
                    content=result.content,
                    tool_call_id=call.id,
                    name=call.name,
                )
            )
            if recorder is not None:
                recorder.message(messages[-1])

    return TurnResult(
        messages=messages,
        stop_reason="max_iterations",
        iterations=steps,
        error=f"stopped after {limit} model calls without a final answer",
        context_events=events,
    )
