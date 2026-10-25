"""H3 demo: drive both context passes with a scripted conversation.

Offline and deterministic — the summarizer is stubbed, the model is not called.

    .venv/bin/python test/context_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.context import SUMMARY_MARKER, ContextManager, conversation_tokens
from harness.messages import Message, ToolCall, assistant, system, tool_result, user

OUTPUT = "clinical note: " + ("patient history, labs, imaging. " * 80)


def group(index: int) -> list[Message]:
    call = ToolCall(id=f"call_{index}", name="echo", arguments={"text": f"batch {index}"})
    return [assistant("", [call]), tool_result(call, f"{OUTPUT} [{index}]")]


def build(groups: int) -> list[Message]:
    messages: list[Message] = [
        system("You are a medical literature assistant. Cite sources."),
        user("Summarize the management of asthma exacerbations."),
        assistant("I will search the corpus."),
    ]
    for index in range(groups):
        messages.extend(group(index))
    messages.append(user("Answer in two sentences, with citations."))
    return messages


def main() -> int:
    messages = build(8)
    tokens = conversation_tokens(messages)
    print(f"conversation: {len(messages)} messages, ~{tokens} tokens")

    print("\n--- stub pass: budget set so the conversation sits at ~50% ---")
    stub_manager = ContextManager(budget=tokens * 2)
    stubbed = stub_manager.apply(messages)
    for event in stubbed.events:
        print(
            f"{event.kind}: {event.messages_before} -> {event.messages_after} messages, "
            f"{event.before_tokens} -> {event.after_tokens} tokens | {event.detail}"
        )
    print(f"request tokens after stubbing: ~{stubbed.tokens}")
    print(f"first 3 untouched: {stubbed.request[:3] == messages[:3]}")
    print(f"last 6 untouched: {stubbed.request[-6:] == messages[-6:]}")
    print(f"canonical keeps full text: {len(stubbed.conversation[4].content)} chars")

    print("\n--- compress pass: same conversation at ~80% of budget ---")
    compress_manager = ContextManager(
        budget=int(tokens / 0.8),
        summarizer=lambda middle: f"[stub summary of {len(middle)} messages]",
    )
    compressed = compress_manager.apply(messages)
    for event in compressed.events:
        print(
            f"{event.kind}: {event.messages_before} -> {event.messages_after} messages, "
            f"{event.before_tokens} -> {event.after_tokens} tokens | {event.detail}"
        )
    print(f"summary inserted at index 3: {compressed.conversation[3].content.startswith(SUMMARY_MARKER)}")
    print(f"first 3 untouched: {compressed.conversation[:3] == messages[:3]}")
    print(f"last 6 untouched: {compressed.conversation[-6:] == messages[-6:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
