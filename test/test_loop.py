"""H1: the loop calls tools, feeds results back, and stops for the right reasons."""

from __future__ import annotations

import json

from harness.llm import LLMError, LLMResponse
from harness.loop import run_turn
from harness.messages import Message, ToolCall, user
from harness.tools import EchoTool, Tool, ToolRegistry


class StubClient:
    """Returns canned responses and records what the loop asked for."""

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.tool_specs: list[list[dict] | None] = []

    def complete(self, messages, tools=None):
        self.calls.append(list(messages))
        self.tool_specs.append(tools)
        if not self.responses:
            raise AssertionError("stub client ran out of responses")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def echo_call(text: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="echo",
        arguments={"text": text},
        raw_arguments=json.dumps({"text": text}),
    )


def registry() -> ToolRegistry:
    return ToolRegistry([EchoTool()])


def test_tool_round_trip_produces_final_answer():
    client = StubClient(
        [
            LLMResponse(content="", tool_calls=[echo_call("loop is alive")]),
            LLMResponse(content="The echo returned: loop is alive"),
        ]
    )

    result = run_turn([user("Echo 'loop is alive'.")], client=client, tools=registry())

    assert result.stop_reason == "answer"
    assert result.answered
    assert result.final == "The echo returned: loop is alive"
    assert result.iterations == 2
    assert [message.role for message in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert result.messages[1].tool_calls[0].name == "echo"
    assert result.messages[2].tool_call_id == "call_1"
    assert result.messages[2].name == "echo"
    assert result.messages[2].content == "loop is alive"
    # The model saw the tool result before answering.
    assert client.calls[1][-1].content == "loop is alive"
    # The echo tool was advertised to the model.
    assert client.tool_specs[0][0]["function"]["name"] == "echo"


def test_user_messages_are_not_mutated():
    conversation = [user("hello")]
    client = StubClient([LLMResponse(content="hi")])

    result = run_turn(conversation, client=client, tools=registry())

    assert len(conversation) == 1
    assert len(result.messages) == 2


def test_unknown_tool_becomes_an_error_result():
    client = StubClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(id="c1", name="shell", arguments={"cmd": "ls"})],
            ),
            LLMResponse(content="That tool does not exist."),
        ]
    )

    result = run_turn([user("Run ls.")], client=client, tools=registry())

    assert result.stop_reason == "answer"
    assert result.messages[2].role == "tool"
    assert result.messages[2].content.startswith("unknown tool 'shell'")
    assert "echo" in result.messages[2].content


def test_unparsable_arguments_become_an_error_result():
    call = ToolCall(id="c1", name="echo", arguments={}, raw_arguments="{not json", parse_error="invalid JSON arguments")
    client = StubClient([LLMResponse(content="", tool_calls=[call]), LLMResponse(content="bad args")])

    result = run_turn([user("echo")], client=client, tools=registry())

    assert result.messages[2].content.startswith("echo: invalid JSON arguments")


def test_tool_exception_becomes_an_error_result():
    class ExplodingTool(Tool):
        name = "boom"
        description = "always fails"

        def run(self, arguments):
            raise RuntimeError("kaput")

    client = StubClient(
        [
            LLMResponse(content="", tool_calls=[ToolCall(id="c1", name="boom")]),
            LLMResponse(content="The tool failed."),
        ]
    )

    result = run_turn([user("break")], client=client, tools=ToolRegistry([ExplodingTool()]))

    assert result.stop_reason == "answer"
    assert result.messages[2].content == "boom raised RuntimeError: kaput"
    assert result.messages[3].content == "The tool failed."


def test_iteration_cap_stops_the_loop():
    responses = [LLMResponse(content="", tool_calls=[echo_call("again", f"c{i}")]) for i in range(5)]
    client = StubClient(responses)

    result = run_turn([user("loop forever")], client=client, tools=registry(), max_iterations=3)

    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3
    assert len(client.calls) == 3
    assert result.final == ""
    assert "3 model calls" in (result.error or "")


def test_model_failure_is_reported_not_raised():
    client = StubClient([LLMError("connection reset")])

    result = run_turn([user("hello")], client=client, tools=registry())

    assert result.stop_reason == "llm_error"
    assert result.error == "connection reset"
    assert result.iterations == 0
    assert [message.role for message in result.messages] == ["user"]


def test_tool_call_without_a_registry_is_an_error_result():
    client = StubClient(
        [
            LLMResponse(content="", tool_calls=[echo_call("hi")]),
            LLMResponse(content="no tools here"),
        ]
    )

    result = run_turn([user("echo")], client=client, tools=None)

    assert result.stop_reason == "answer"
    assert result.messages[2].content == "no tools are registered; cannot call 'echo'"


def test_multiple_tool_calls_in_one_turn_all_run():
    client = StubClient(
        [
            LLMResponse(
                content="",
                tool_calls=[echo_call("one", "c1"), echo_call("two", "c2")],
            ),
            LLMResponse(content="both echoed"),
        ]
    )

    result = run_turn([user("echo twice")], client=client, tools=registry())

    tool_messages = [message for message in result.messages if message.role == "tool"]
    assert [message.content for message in tool_messages] == ["one", "two"]
