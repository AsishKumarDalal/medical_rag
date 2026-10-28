"""The TUI's structured events and pure text layout."""

from __future__ import annotations

import pytest

from harness.config import Config
from harness.context import ContextManager
from harness.llm import LLMResponse
from harness.messages import ToolCall
from harness.session import SessionStore
from harness.tools import EchoTool, ToolRegistry
from tui.chat import TuiChat, compact
from tui.render import Entry, scroll_window, wrap_entry

USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


class StubClient:
    def __init__(self, *replies: LLMResponse):
        self.replies = list(replies)

    def complete(self, messages, tools=None):
        return self.replies.pop(0)


@pytest.fixture
def config(tmp_path):
    return Config(
        base_url="https://x.test/v1",
        api_key="k",
        model="test-model",
        db_path=str(tmp_path / "sessions.db"),
    )


@pytest.fixture
def store(config):
    session_store = SessionStore(config.db_path)
    yield session_store
    session_store.close()


def make_chat(config, store, client, events):
    return TuiChat(
        config=config,
        client=client,
        store=store,
        tools=ToolRegistry([EchoTool()]),
        context=ContextManager(budget=100_000),
        events=events.append,
    )


def test_turn_is_published_as_structured_events(config, store):
    events: list[dict] = []
    client = StubClient(
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="echo",
                    arguments={"text": "hi"},
                    raw_arguments='{"text": "hi"}',
                )
            ],
        ),
        LLMResponse(content="the echo said hi", usage=USAGE),
    )
    chat = make_chat(config, store, client, events)

    chat.ask("echo hi")

    kinds = [event["kind"] for event in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "answer" in kinds
    assert "usage" in kinds
    answer = next(event for event in events if event["kind"] == "answer")
    assert answer["text"] == "the echo said hi"
    usage = next(event for event in events if event["kind"] == "usage")
    assert usage["total_tokens"] == 12


def test_command_output_reaches_the_tui_as_notices(config, store):
    events: list[dict] = []
    chat = make_chat(config, store, StubClient(), events)

    chat.handle("/help")

    notices = [event["text"] for event in events if event["kind"] == "notice"]
    assert any("/sessions" in text for text in notices)


def test_wrap_entry_marks_the_speaker_and_indents_continuations():
    lines = wrap_entry(Entry("you", "one two three four"), width=10)

    assert lines[0] == ("you", "you> one")
    assert all(role == "you" for role, _ in lines)
    assert all(line.startswith("     ") for _, line in lines[1:])


def test_wrap_entry_keeps_explicit_newlines():
    lines = wrap_entry(Entry("assistant", "first\nsecond"), width=40)

    assert [line for _, line in lines] == [
        "harness> first",
        "         second",
    ]


def test_scroll_window_is_bottom_anchored_and_clamped():
    assert scroll_window(100, 10, 0) == (90, 100, 0)
    assert scroll_window(100, 10, 5) == (85, 95, 5)
    assert scroll_window(100, 10, 500) == (0, 10, 90)
    assert scroll_window(3, 10, 0) == (0, 3, 0)


def test_compact_collapses_whitespace_and_caps_length():
    assert compact("one \n two") == "one two"
    assert len(compact("x" * 100, limit=10)) == 10
