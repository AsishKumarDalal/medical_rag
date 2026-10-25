"""The agent loop: model → tool calls → results → model, until it answers.

One :func:`run_turn` call is one user turn. It never raises for model or tool
failures: those come back as a stop reason, so the caller (CLI, tests, later the
session recorder) always gets a transcript it can persist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .config import Config
from .llm import LLMError, SupportsComplete
from .messages import Message
from .tools.base import ToolResult
from .tools.registry import ToolRegistry

StopReason = Literal["answer", "max_iterations", "llm_error"]


@dataclass(slots=True)
class TurnResult:
    """Everything one turn produced."""

    messages: list[Message] = field(default_factory=list)
    final: str = ""
    stop_reason: StopReason = "answer"
    iterations: int = 0
    error: str | None = None

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
) -> TurnResult:
    """Run one turn to a final answer, an iteration cap, or a model failure.

    ``conversation`` is copied; the returned ``messages`` is the full transcript,
    input messages first, ready to be stored as the next conversation state.
    """
    limit = max_iterations or (config.max_iterations if config else 12)
    messages = list(conversation)
    specs = tools.specs() if tools else None
    steps = 0

    while steps < limit:
        try:
            response = client.complete(messages, tools=specs)
        except LLMError as exc:
            return TurnResult(
                messages=messages,
                stop_reason="llm_error",
                iterations=steps,
                error=str(exc),
            )

        steps += 1
        messages.append(Message(role="assistant", content=response.content, tool_calls=list(response.tool_calls)))

        if not response.tool_calls:
            return TurnResult(
                messages=messages,
                final=response.content,
                stop_reason="answer",
                iterations=steps,
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

    return TurnResult(
        messages=messages,
        stop_reason="max_iterations",
        iterations=steps,
        error=f"stopped after {limit} model calls without a final answer",
    )
