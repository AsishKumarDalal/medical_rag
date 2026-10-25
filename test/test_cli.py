"""H6: the terminal can hold a conversation and keep it."""

from __future__ import annotations

import pytest

from harness.cli import Chat, main
from harness.config import Config
from harness.context import ContextManager
from harness.llm import LLMResponse
from harness.messages import ToolCall
from harness.session import SessionStore
from harness.tools import EchoTool, ToolRegistry


USAGE = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


class StubClient:
    def __init__(self, *replies: str | LLMResponse):
        self.replies = [
            reply if isinstance(reply, LLMResponse) else LLMResponse(content=reply, usage=USAGE)
            for reply in replies
        ]
        self.calls: list[tuple[list, list | None]] = []

    def complete(self, messages, tools=None):
        self.calls.append((list(messages), tools))
        if self.replies:
            return self.replies.pop(0)
        return LLMResponse(content="the answer", usage=USAGE)


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


def make_chat(config, store, client, out):
    return Chat(
        config=config,
        client=client,
        store=store,
        tools=ToolRegistry([EchoTool()]),
        context=ContextManager(budget=100_000),
        out=out,
    )


def test_ask_stores_the_turn_and_prints_the_answer(config, store):
    out: list[str] = []
    client = StubClient("the answer")
    chat = make_chat(config, store, client, out.append)

    result = chat.ask("What is asthma?")

    assert result.final == "the answer"
    messages = store.load_messages(chat.session_id)
    assert [message.role for message in messages] == ["system", "user", "assistant"]
    assert messages[1].content == "What is asthma?"
    assert any("the answer" in line for line in out)
    assert store.get(chat.session_id).title == "What is asthma?"


def test_tools_are_offered_to_the_model(config, store):
    client = StubClient("ok")
    chat = make_chat(config, store, client, lambda line: None)

    chat.ask("hello")

    specs = client.calls[0][1]
    assert [spec["function"]["name"] for spec in specs] == ["echo"]


def test_tool_activity_is_shown_compactly(config, store):
    out: list[str] = []
    client = StubClient(
        LLMResponse(
            content="",
            tool_calls=[
                ToolCall(id="c1", name="echo", arguments={"text": "hi"}, raw_arguments='{"text": "hi"}')
            ],
        ),
        LLMResponse(content="the echo said hi", usage=USAGE),
    )
    chat = make_chat(config, store, client, out.append)

    chat.ask("echo hi")

    assert any(line.startswith("  -> echo(") for line in out)
    assert any(line.startswith("  <- echo: hi") for line in out)


def test_new_session_and_sessions_listing(config, store):
    out: list[str] = []
    chat = make_chat(config, store, StubClient("one", "two"), out.append)
    first = chat.session_id
    chat.ask("first question")

    assert chat.handle("/new") is True
    second = chat.session_id
    chat.ask("second question")

    assert second != first
    chat.handle("/sessions")
    listing = "\n".join(out)
    assert first in listing
    assert second in listing
    assert "*" in listing


def test_resume_restores_history(config, store):
    out: list[str] = []
    client = StubClient("first answer", "second answer")
    chat = make_chat(config, store, client, out.append)
    original = chat.session_id
    chat.ask("remember this")

    chat.handle("/new")
    assert chat.session_id != original
    chat.handle(f"/resume {original}")
    assert chat.session_id == original
    chat.ask("what did I just ask?")

    last_call, _ = client.calls[-1]
    assert [message.content for message in last_call if message.role == "user"] == [
        "remember this",
        "what did I just ask?",
    ]


def test_resume_unknown_session_is_reported(config, store):
    out: list[str] = []
    chat = make_chat(config, store, StubClient(), out.append)

    assert chat.handle("/resume nope") is True

    assert any("unknown session" in line for line in out)


def test_usage_reports_totals(config, store):
    out: list[str] = []
    chat = make_chat(config, store, StubClient("hi"), out.append)
    chat.ask("hello")
    out.clear()

    chat.handle("/usage")

    assert any("total=12" in line for line in out)


def test_unknown_command_and_exit(config, store):
    out: list[str] = []
    chat = make_chat(config, store, StubClient(), out.append)

    assert chat.handle("/dance") is True
    assert any("unknown command" in line for line in out)
    assert chat.handle("/exit") is False
    assert chat.handle("/quit") is False


def test_main_once_asks_and_exits(config, store):
    out: list[str] = []
    client = StubClient("one-shot answer")

    code = main(
        ["--once", "hello", "--db", config.db_path],
        client=client,
        store=store,
        tools=ToolRegistry([EchoTool()]),
        context=ContextManager(budget=100_000),
        out=out.append,
    )

    assert code == 0
    assert any("one-shot answer" in line for line in out)


def test_main_list_shows_sessions(config, store):
    out: list[str] = []
    chat = make_chat(config, store, StubClient("hi"), out.append)
    chat.ask("hello")
    out.clear()

    code = main(["--list"], store=store, out=out.append)

    assert code == 0
    assert any(chat.session_id in line for line in out)


def test_main_refuses_unknown_session(config, store):
    out: list[str] = []

    code = main(["--session", "missing"], store=store, out=out.append)

    assert code == 2
    assert any("unknown session" in line for line in out)


def test_main_without_configuration_explains_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    out: list[str] = []

    code = main(["--list"], out=out.append)

    assert code == 2
    assert any("LLM_BASE_URL" in line for line in out)
