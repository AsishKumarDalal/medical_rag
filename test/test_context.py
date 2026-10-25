"""H3: the 45% stub pass and the 75% compression pass."""

from __future__ import annotations

import pytest

from harness.config import Config
from harness.context import (
    STUB_MARKER,
    SUMMARY_MARKER,
    ContextError,
    ContextManager,
    LLMSummarizer,
    conversation_tokens,
    estimate_tokens,
    render_transcript,
)
from harness.llm import LLMResponse
from harness.loop import run_turn
from harness.messages import Message, ToolCall, assistant, system, tool_result, user

OUTPUT = "x" * 2000


def group(index: int, text: str = OUTPUT) -> list[Message]:
    call = ToolCall(
        id=f"call_{index}",
        name="echo",
        arguments={"text": text},
        raw_arguments='{"text": "..."}',
    )
    return [assistant("", [call]), tool_result(call, f"{text}-{index}")]


def conversation(groups: int) -> list[Message]:
    messages: list[Message] = [system("be useful"), user("the task"), assistant("starting")]
    for index in range(groups):
        messages.extend(group(index))
    messages.append(user("final question"))
    return messages


def manager_for(messages, ratio: float, **kwargs) -> ContextManager:
    tokens = conversation_tokens(messages)
    return ContextManager(budget=int(tokens / ratio), **kwargs)


def assert_grouping_is_valid(messages: list[Message]) -> None:
    """Every tool result follows the assistant call that produced it."""
    open_calls: set[str] = set()
    for message in messages:
        if message.role == "tool":
            assert message.tool_call_id in open_calls, f"orphan tool result {message.tool_call_id}"
            open_calls.discard(message.tool_call_id)
        elif message.role == "assistant":
            for call in message.tool_calls:
                open_calls.add(call.id)


def test_below_threshold_nothing_changes():
    messages = conversation(3)
    manager = manager_for(messages, 0.2)

    budgeted = manager.apply(messages)

    assert budgeted.events == []
    assert budgeted.request == messages
    assert budgeted.conversation == messages


def test_stub_pass_replaces_stale_tool_outputs_for_the_request_only():
    messages = conversation(6)
    manager = manager_for(messages, 0.5)

    budgeted = manager.apply(messages)

    assert [event.kind for event in budgeted.events] == ["stub"]
    # The last six messages are protected, so only the older tool results are cut.
    assert budgeted.request[4].content.startswith(STUB_MARKER)
    assert budgeted.request[6].content.startswith(STUB_MARKER)
    assert budgeted.request[10].content == messages[10].content
    # The canonical conversation still holds the full text.
    assert budgeted.conversation[4].content.endswith("-0")
    assert len(budgeted.conversation[4].content) > 2000
    assert budgeted.tokens < conversation_tokens(messages)


def test_stub_pass_keeps_first_three_and_last_six_untouched():
    messages = conversation(6)
    manager = manager_for(messages, 0.5)

    budgeted = manager.apply(messages)

    assert budgeted.request[:3] == messages[:3]
    assert budgeted.request[-6:] == messages[-6:]


def test_stub_pass_is_idempotent():
    messages = conversation(6)
    manager = manager_for(messages, 0.5)

    first = manager.apply(messages)
    second = manager.apply(messages)

    assert first.request == second.request
    # Running the pass on its own output changes nothing and reports nothing.
    third = manager.apply(first.request)
    assert third.request == first.request
    assert third.events == []


def test_compress_pass_summarizes_the_middle_and_preserves_the_edges():
    messages = conversation(10)
    manager = manager_for(messages, 0.8, summarizer=lambda middle: "short summary of the middle")

    budgeted = manager.apply(messages)

    kinds = [event.kind for event in budgeted.events]
    assert "compress" in kinds
    compress = budgeted.events[0]
    assert compress.before_tokens > compress.after_tokens

    assert budgeted.conversation[:3] == messages[:3]
    assert budgeted.conversation[3].content.startswith(SUMMARY_MARKER)
    assert "short summary of the middle" in budgeted.conversation[3].content
    # The tail is untouched (it may be longer than six messages to keep a call
    # together with its results).
    assert budgeted.conversation[-6:] == messages[-6:]
    assert_grouping_is_valid(budgeted.conversation)


def test_compress_pass_falls_back_when_the_summarizer_fails():
    messages = conversation(10)

    def broken(middle):
        raise RuntimeError("model down")

    manager = manager_for(messages, 0.8, summarizer=broken)

    budgeted = manager.apply(messages)

    assert [event.kind for event in budgeted.events] == ["compress_fallback"]
    summary = budgeted.conversation[3].content
    assert "earlier messages were removed" in summary
    assert_grouping_is_valid(budgeted.conversation)


def test_compress_pass_does_nothing_when_there_is_no_middle():
    messages = [
        system("s"),
        user("u"),
        assistant("a" * 4000),
        tool_result(ToolCall(id="c", name="echo"), "t" * 4000),
        user("again"),
    ]
    manager = ContextManager(budget=10, summarizer=lambda middle: "should not run")

    budgeted = manager.apply(messages)

    assert [event.kind for event in budgeted.events] == []
    assert budgeted.conversation == messages


def test_compress_pass_never_orphans_a_tool_result_at_the_start():
    call = ToolCall(id="c1", name="echo", arguments={"text": "first"})
    messages = [
        system("s"),
        user("u"),
        assistant("", [call]),
        tool_result(call, "first " + OUTPUT),
        assistant("between"),
    ]
    for index in range(2, 7):
        messages.extend(group(index))
    messages.append(user("done"))
    manager = manager_for(messages, 0.8, summarizer=lambda middle: "summary")

    budgeted = manager.apply(messages)

    assert budgeted.conversation[:4] == messages[:4]  # the call and its result stay together
    assert budgeted.conversation[4].content.startswith(SUMMARY_MARKER)
    assert_grouping_is_valid(budgeted.conversation)


def test_compress_pass_never_splits_a_call_from_its_results():
    messages = conversation(6)
    # Make the boundary fall right between an assistant tool call and its result.
    boundary = len(messages) - 6
    assert messages[boundary - 1].role == "assistant"
    manager = manager_for(messages, 0.8, summarizer=lambda middle: "summary")

    budgeted = manager.apply(messages)

    assert_grouping_is_valid(budgeted.conversation)
    assert budgeted.conversation[-1] == messages[-1]


def test_invalid_configuration_is_rejected():
    with pytest.raises(ContextError, match="budget"):
        ContextManager(budget=0)
    with pytest.raises(ContextError, match="stub_at"):
        ContextManager(budget=100, stub_at=0.9, compress_at=0.5)
    with pytest.raises(ContextError, match="keep_first"):
        ContextManager(budget=100, keep_first=-1)


def test_budget_reserves_completion_tokens():
    config = Config(
        base_url="https://x.test/v1",
        api_key="k",
        model="m",
        max_tokens=4096,
        context_window=50_000,
    )

    manager = ContextManager.from_config(config)

    assert manager.budget == 50_000 - 4096


def test_token_estimate_is_cheap_but_sane():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        return self.responses.pop(0)


def test_llm_summarizer_uses_the_client_without_tools():
    client = StubClient([LLMResponse(content="a compact summary")])
    summarizer = LLMSummarizer(client)

    text = summarizer([user("hi"), assistant("there")])

    assert text == "a compact summary"
    messages, tools = client.calls[0]
    assert tools is None
    assert messages[0].role == "system"
    assert "hi" in messages[1].content


def test_llm_summarizer_rejects_an_empty_summary():
    client = StubClient([LLMResponse(content="  ")])

    with pytest.raises(ContextError, match="empty summary"):
        LLMSummarizer(client)([user("hi")])


def test_render_transcript_keeps_both_ends_when_huge():
    messages = [user("start" + "a" * 500), assistant("b" * 500 + "end")]

    text = render_transcript(messages, max_chars=200)

    assert text.startswith("user: start")
    assert "middle omitted" in text
    assert text.endswith("end")


def test_loop_applies_stubbing_before_the_model_call():
    messages = conversation(6)
    manager = manager_for(messages, 0.5)
    client = StubClient([LLMResponse(content="done")])

    result = run_turn(messages, client=client, tools=None, context=manager)

    assert [event.kind for event in result.context_events] == ["stub"]
    sent = client.calls[0][0]
    assert sent[4].content.startswith(STUB_MARKER)
    assert result.messages[4].content.endswith("-0")


def test_loop_returns_the_compressed_transcript():
    messages = conversation(10)
    manager = manager_for(messages, 0.8, summarizer=lambda middle: "summary")
    client = StubClient([LLMResponse(content="done")])

    result = run_turn(messages, client=client, tools=None, context=manager)

    assert [event.kind for event in result.context_events] == ["compress"]
    assert len(result.messages) < len(messages)
    assert result.messages[3].content.startswith(SUMMARY_MARKER)
    assert result.final == "done"
