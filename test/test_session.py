"""H4: sessions survive restarts, isolation, and a crash mid-turn."""

from __future__ import annotations

import pytest

from harness.config import Config
from harness.context import ContextEvent, ContextManager
from harness.llm import LLMError, LLMResponse
from harness.loop import run_turn
from harness.messages import ToolCall, assistant, tool_result, user
from harness.session import SessionError, SessionStore
from harness.tools import EchoTool, ToolRegistry


@pytest.fixture
def store(tmp_path):
    session_store = SessionStore(tmp_path / "sessions.db")
    yield session_store
    session_store.close()


class StubClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, messages, tools=None):
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def echo_call(text: str = "hi") -> ToolCall:
    return ToolCall(id="call_1", name="echo", arguments={"text": text}, raw_arguments='{"text": "hi"}')


def test_create_get_list_and_delete(store):
    session_id = store.create(title="asthma", model="m")

    info = store.get(session_id)
    assert info is not None
    assert info.title == "asthma"
    assert info.messages == 0
    assert [item.id for item in store.sessions()] == [session_id]

    assert store.delete(session_id) is True
    assert store.get(session_id) is None
    assert store.delete(session_id) is False


def test_duplicate_session_id_is_rejected(store):
    store.create(session_id="fixed")

    with pytest.raises(SessionError, match="already exists"):
        store.create(session_id="fixed")


def test_messages_round_trip_with_tool_calls(store):
    session_id = store.create()
    call = echo_call()
    store.append_message(session_id, user("hello"), turn=1)
    store.append_message(session_id, assistant("", [call]), turn=1)
    store.append_message(session_id, tool_result(call, "hi"), turn=1)

    loaded = store.load_messages(session_id)

    assert loaded == [user("hello"), assistant("", [call]), tool_result(call, "hi")]
    assert loaded[1].tool_calls[0].arguments == {"text": "hi"}


def test_tool_log_records_arguments_and_results(store):
    session_id = store.create()
    call = echo_call()
    store.append_message(session_id, assistant("", [call]))
    store.append_message(session_id, tool_result(call, "hi"))

    log = store.tool_log(session_id)

    assert len(log) == 1
    assert log[0]["name"] == "echo"
    assert log[0]["arguments"] == '{"text": "hi"}'
    assert log[0]["result"] == "hi"
    assert log[0]["ok"] == 1


def test_context_events_and_usage_survive(store):
    session_id = store.create()
    event = ContextEvent(
        kind="stub",
        before_tokens=1000,
        after_tokens=400,
        messages_before=12,
        messages_after=12,
        detail="stubbed 2 tool outputs",
    )
    store.record_context_event(session_id, event, turn=1)
    store.record_usage(session_id, {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
    store.record_usage(session_id, {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60})

    events = store.context_events(session_id)
    totals = store.usage_totals(session_id)

    assert events[0]["kind"] == "stub"
    assert events[0]["detail"] == "stubbed 2 tool outputs"
    assert totals == {"prompt_tokens": 150, "completion_tokens": 30, "total_tokens": 180, "calls": 2}


def test_delete_cascades_to_messages(tmp_path):
    path = tmp_path / "sessions.db"
    with SessionStore(path) as store:
        session_id = store.create()
        store.append_message(session_id, user("hi"))
        assert store.delete(session_id)

        with pytest.raises(SessionError):
            store.append_message(session_id, user("again"))


def test_sessions_are_isolated(store):
    first = store.create()
    second = store.create()
    store.append_message(first, user("first"))
    store.append_message(second, user("second"))

    assert [message.content for message in store.load_messages(first)] == ["first"]
    assert [message.content for message in store.load_messages(second)] == ["second"]


def test_reopening_the_file_restores_the_conversation(tmp_path):
    path = tmp_path / "sessions.db"
    store = SessionStore(path)
    session_id = store.create(title="kept")
    store.append_message(session_id, user("still here"), turn=3)
    store.close()

    reopened = SessionStore(path)
    try:
        assert reopened.get(session_id).title == "kept"
        assert [message.content for message in reopened.load_messages(session_id)] == ["still here"]
    finally:
        reopened.close()


def test_sanitizer_redacts_before_writing(tmp_path):
    store = SessionStore(tmp_path / "s.db", sanitize=lambda text: text.replace("sk-live", "[redacted]"))
    try:
        session_id = store.create()
        store.append_message(session_id, user("key is sk-live-123"))

        assert store.load_messages(session_id)[0].content == "key is [redacted]-123"
    finally:
        store.close()


def test_unknown_session_is_reported(store):
    with pytest.raises(SessionError, match="unknown session"):
        store.append_message("nope", user("hi"))


def test_loop_records_each_step_through_the_recorder(store):
    session_id = store.create()
    store.append_message(session_id, user("echo hi"))
    client = StubClient(
        [
            LLMResponse(content="", tool_calls=[echo_call()], usage={"prompt_tokens": 30, "completion_tokens": 5, "total_tokens": 35}),
            LLMResponse(content="done", usage={"prompt_tokens": 40, "completion_tokens": 5, "total_tokens": 45}),
        ]
    )

    result = run_turn(
        store.load_messages(session_id),
        client=client,
        tools=ToolRegistry([EchoTool()]),
        recorder=store.recorder(session_id, turn=1),
    )

    assert result.stop_reason == "answer"
    stored = store.load_messages(session_id)
    assert [message.role for message in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[2].content == "hi"
    assert store.usage_totals(session_id)["total_tokens"] == 80


def test_a_crash_mid_turn_keeps_completed_steps(store):
    session_id = store.create()
    store.append_message(session_id, user("echo twice"))
    client = StubClient([LLMResponse(content="", tool_calls=[echo_call()]), LLMError("connection reset")])

    result = run_turn(
        store.load_messages(session_id),
        client=client,
        tools=ToolRegistry([EchoTool()]),
        recorder=store.recorder(session_id, turn=1),
    )

    assert result.stop_reason == "llm_error"
    stored = store.load_messages(session_id)
    assert [message.role for message in stored] == ["user", "assistant", "tool"]


def test_context_events_are_recorded_while_running(store):
    session_id = store.create()
    store.append_message(session_id, user("start"))
    big = "x" * 4000
    for index in range(6):
        call = ToolCall(id=f"c{index}", name="echo", arguments={"text": big})
        store.append_message(session_id, assistant("", [call]))
        store.append_message(session_id, tool_result(call, big))
    store.append_message(session_id, user("again"))
    conversation = store.load_messages(session_id)
    manager = ContextManager(budget=2_000, summarizer=lambda middle: "summary")
    client = StubClient([LLMResponse(content="ok")])

    run_turn(
        conversation,
        client=client,
        tools=ToolRegistry([EchoTool()]),
        context=manager,
        recorder=store.recorder(session_id, turn=1),
    )

    assert [event["kind"] for event in store.context_events(session_id)]


def test_db_path_comes_from_config():
    config = Config.from_env(
        {"LLM_BASE_URL": "https://x.test/v1", "LLM_API_KEY": "k", "LLM_MODEL": "m"}
    )

    assert config.db_path.endswith(".db")
