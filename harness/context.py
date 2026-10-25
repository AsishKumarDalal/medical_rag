"""Keep the conversation inside the model's context window.

Two passes, in order:

**75% of the prompt budget — compress.**
The model summarizes the middle of the conversation into a single message. The
first three and the last six messages are never touched and never compressed.
The compressed conversation replaces the stored one, because otherwise it would be
recompressed on every request.

**45% of the prompt budget — stub.**
Long tool outputs from earlier steps are replaced by a short stub *for this request
only*. The canonical conversation keeps the full text, so nothing is lost; the model
just stops paying for output it already read. The most recent tool outputs stay
intact because the model is usually still working with them.

Both passes only ever shorten; they are deterministic, idempotent, and every change
is reported as a :class:`ContextEvent` so a session can show what happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import ceil
from typing import Callable, Literal, Sequence

from .config import Config
from .llm import SupportsComplete
from .messages import Message, system, user

# One token is roughly four characters of English. Deliberately crude: the budget
# stays conservative because the estimate is applied to everything.
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_CALL_OVERHEAD_TOKENS = 8

STUB_MARKER = "[tool output omitted to save context"
SUMMARY_MARKER = "Summary of the earlier conversation:"

STUB_TEMPLATE = (
    "[tool output omitted to save context: {name} returned {chars} characters "
    "(~{tokens} tokens). The full text is in the session transcript; call the tool "
    "again if you need it.]"
)

EventKind = Literal["stub", "compress", "compress_fallback"]


class ContextError(RuntimeError):
    """The context manager was given an unusable configuration or input."""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / CHARS_PER_TOKEN))


def message_tokens(message: Message) -> int:
    total = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message.content)
    for call in message.tool_calls:
        arguments = call.raw_arguments or json.dumps(call.arguments)
        total += (
            TOOL_CALL_OVERHEAD_TOKENS
            + estimate_tokens(call.name)
            + estimate_tokens(arguments)
        )
    return total


def conversation_tokens(messages: Sequence[Message]) -> int:
    return sum(message_tokens(message) for message in messages)


@dataclass(slots=True)
class ContextEvent:
    kind: EventKind
    before_tokens: int
    after_tokens: int
    messages_before: int
    messages_after: int
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "before_tokens": self.before_tokens,
            "after_tokens": self.after_tokens,
            "messages_before": self.messages_before,
            "messages_after": self.messages_after,
            "detail": self.detail,
        }


@dataclass(slots=True)
class Budgeted:
    """What one model call should send, and what changed getting there."""

    conversation: list[Message]
    request: list[Message]
    events: list[ContextEvent] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return conversation_tokens(self.request)


Summarizer = Callable[[Sequence[Message]], str]


class ContextManager:
    """Applies the 75% and 45% passes to a conversation."""

    def __init__(
        self,
        *,
        budget: int,
        stub_at: float = 0.45,
        compress_at: float = 0.75,
        keep_first: int = 3,
        keep_last: int = 6,
        stub_min_tokens: int = 150,
        keep_recent_tool_outputs: int = 2,
        max_stub_chars: int = 600,
        max_summary_chars: int = 4_000,
        summarizer: Summarizer | None = None,
    ) -> None:
        if budget <= 0:
            raise ContextError("budget must be positive")
        if not 0 < stub_at < compress_at <= 1:
            raise ContextError("expected 0 < stub_at < compress_at <= 1")
        if keep_first < 0 or keep_last < 0:
            raise ContextError("keep_first and keep_last must not be negative")
        if stub_min_tokens <= 0:
            raise ContextError("stub_min_tokens must be positive")
        self.budget = budget
        self.stub_at = stub_at
        self.compress_at = compress_at
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.stub_min_tokens = stub_min_tokens
        self.keep_recent_tool_outputs = keep_recent_tool_outputs
        self.max_stub_chars = max_stub_chars
        self.max_summary_chars = max_summary_chars
        self.summarizer = summarizer

    @classmethod
    def from_config(cls, config: Config, *, summarizer: Summarizer | None = None) -> "ContextManager":
        """Budget the prompt, not the whole window: completion tokens are reserved."""
        return cls(
            budget=config.context_window - config.max_tokens,
            summarizer=summarizer,
        )

    def ratio(self, messages: Sequence[Message]) -> float:
        return conversation_tokens(messages) / self.budget

    def apply(self, messages: Sequence[Message]) -> Budgeted:
        conversation = list(messages)
        events: list[ContextEvent] = []

        if conversation_tokens(conversation) >= self.compress_at * self.budget:
            compressed, event = self._compress(conversation)
            if event is not None:
                conversation = compressed
                events.append(event)

        request, stub_event = self._stub(conversation)
        if stub_event is not None:
            events.append(stub_event)

        return Budgeted(conversation=conversation, request=request, events=events)

    # -- 75% pass ---------------------------------------------------------

    def _compress(
        self, messages: list[Message]
    ) -> tuple[list[Message], ContextEvent | None]:
        before_tokens = conversation_tokens(messages)
        start = self.keep_first
        # Never leave a tool result at the front of the summarized block: its
        # call would sit in the untouched prefix and become an orphan.
        while start < len(messages) and messages[start].role == "tool":
            start += 1
        end = max(start, len(messages) - self.keep_last)
        # Never split a call from its results: move the boundary back so the
        # assistant call and its results stay together in the untouched tail.
        while end > start and messages[end - 1].role == "assistant" and messages[end - 1].tool_calls:
            end -= 1
        if end - start < 2:
            return messages, None

        middle = messages[start:end]
        text, kind = self._summarize(middle)
        summary = Message(role="system", content=f"{SUMMARY_MARKER}\n{text}")
        compressed = messages[:start] + [summary] + messages[end:]
        event = ContextEvent(
            kind=kind,
            before_tokens=before_tokens,
            after_tokens=conversation_tokens(compressed),
            messages_before=len(messages),
            messages_after=len(compressed),
            detail=(
                f"summarized {len(middle)} messages into {len(text)} characters"
                if kind == "compress"
                else f"summarizer unavailable; replaced {len(middle)} messages with a note"
            ),
        )
        return compressed, event

    def _summarize(self, messages: list[Message]) -> tuple[str, EventKind]:
        if self.summarizer is not None:
            try:
                text = (self.summarizer(messages) or "").strip()
            except Exception:
                text = ""
            if text:
                return text[: self.max_summary_chars], "compress"
        return self._fallback_summary(messages), "compress_fallback"

    @staticmethod
    def _fallback_summary(messages: Sequence[Message]) -> str:
        counts: dict[str, int] = {}
        tools: list[str] = []
        for message in messages:
            counts[message.role] = counts.get(message.role, 0) + 1
            for call in message.tool_calls:
                if call.name not in tools:
                    tools.append(call.name)
        roles = ", ".join(f"{role} {count}" for role, count in sorted(counts.items()))
        called = ", ".join(tools) if tools else "none"
        return (
            f"{len(messages)} earlier messages were removed to stay inside the context "
            f"window ({roles}). Tools called: {called}. Details are in the session "
            "transcript."
        )

    # -- 45% pass ---------------------------------------------------------

    def _stub(self, messages: list[Message]) -> tuple[list[Message], ContextEvent | None]:
        before_tokens = conversation_tokens(messages)
        if before_tokens < self.stub_at * self.budget:
            return messages, None

        total = len(messages)
        protected = set(range(min(self.keep_first, total))) | set(
            range(max(0, total - self.keep_last), total)
        )
        tool_indices = [index for index, message in enumerate(messages) if message.role == "tool"]
        recent = set(tool_indices[-self.keep_recent_tool_outputs :]) if self.keep_recent_tool_outputs else set()

        request = list(messages)
        stubbed = 0
        omitted_tokens = 0
        for index in tool_indices:
            if index in protected or index in recent:
                continue
            message = messages[index]
            if message.content.startswith(STUB_MARKER):
                continue
            tokens = estimate_tokens(message.content)
            if tokens < self.stub_min_tokens:
                continue
            stub = STUB_TEMPLATE.format(
                name=message.name or "tool",
                chars=len(message.content),
                tokens=tokens,
            )[: self.max_stub_chars]
            request[index] = Message(
                role="tool",
                content=stub,
                tool_call_id=message.tool_call_id,
                name=message.name,
            )
            stubbed += 1
            omitted_tokens += tokens - estimate_tokens(stub)

        if stubbed == 0:
            return messages, None
        after_tokens = conversation_tokens(request)
        return request, ContextEvent(
            kind="stub",
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            messages_before=total,
            messages_after=total,
            detail=f"stubbed {stubbed} earlier tool outputs (~{omitted_tokens} tokens)",
        )


SUMMARY_SYSTEM_PROMPT = (
    "You compress a conversation transcript so it fits in a smaller context window. "
    "Keep: the user's goal, decisions made, facts and tool results that still matter "
    "(with their identifiers), open questions, and unresolved errors. Drop greetings, "
    "repetition and already-finished detail. Write plain text, at most 300 words, no "
    "preamble, no headings."
)


class LLMSummarizer:
    """Summarizes messages with the model, through the single client doorway."""

    def __init__(self, client: SupportsComplete, *, max_transcript_chars: int = 24_000) -> None:
        self.client = client
        self.max_transcript_chars = max_transcript_chars

    def __call__(self, messages: Sequence[Message]) -> str:
        transcript = render_transcript(messages, self.max_transcript_chars)
        response = self.client.complete(
            [system(SUMMARY_SYSTEM_PROMPT), user(transcript)],
            tools=None,
        )
        text = (response.content or "").strip()
        if not text:
            raise ContextError("summarizer returned an empty summary")
        return text


def render_transcript(messages: Sequence[Message], max_chars: int) -> str:
    """Flatten messages for summarization, keeping the head and tail if huge."""
    lines: list[str] = []
    for message in messages:
        prefix = message.role if not message.name else f"{message.role} ({message.name})"
        if message.tool_calls:
            calls = ", ".join(
                f"{call.name}({call.raw_arguments or json.dumps(call.arguments)})"
                for call in message.tool_calls
            )
            lines.append(f"{prefix}: [tool calls] {calls}")
        if message.content:
            lines.append(f"{prefix}: {message.content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head]}\n[... middle omitted ...]\n{text[-tail:]}"
