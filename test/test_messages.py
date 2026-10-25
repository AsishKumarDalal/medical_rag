"""H1: messages survive the round trip to and from the provider format."""

from __future__ import annotations

import json

from harness.messages import Message, ToolCall, assistant, system, tool_result, user


def test_plain_messages_round_trip():
    for message in (system("be brief"), user("hello"), Message(role="assistant", content="hi")):
        assert Message.from_provider(message.to_provider()) == message


def test_assistant_message_with_tool_calls_round_trips():
    call = ToolCall(
        id="call_9",
        name="echo",
        arguments={"text": "hi"},
        raw_arguments='{"text": "hi"}',
    )
    message = assistant("", [call])

    wire = message.to_provider()
    assert wire["tool_calls"][0]["function"]["arguments"] == '{"text": "hi"}'

    restored = Message.from_provider(wire)
    assert restored == message
    assert restored.tool_calls[0].arguments == {"text": "hi"}


def test_tool_message_round_trips():
    call = ToolCall(id="call_1", name="echo", arguments={"text": "x"})
    message = tool_result(call, "x")

    wire = message.to_provider()
    assert wire == {"role": "tool", "content": "x", "tool_call_id": "call_1", "name": "echo"}
    assert Message.from_provider(wire) == message


def test_invalid_argument_json_is_reported_not_raised():
    parsed = ToolCall.from_provider(
        {"id": "c1", "function": {"name": "echo", "arguments": "{not json"}}
    )

    assert parsed.arguments == {}
    assert parsed.raw_arguments == "{not json"
    assert parsed.parse_error is not None
    assert "invalid JSON" in parsed.parse_error


def test_non_object_arguments_are_rejected():
    parsed = ToolCall.from_provider(
        {"id": "c1", "function": {"name": "echo", "arguments": "[1, 2]"}}
    )

    assert parsed.arguments == {}
    assert parsed.parse_error is not None
    assert "JSON object" in parsed.parse_error


def test_dict_arguments_from_sdk_are_accepted():
    parsed = ToolCall.from_provider(
        {"id": "c1", "function": {"name": "echo", "arguments": {"text": "hi"}}}
    )

    assert parsed.arguments == {"text": "hi"}
    assert parsed.parse_error is None


def test_empty_arguments_are_allowed():
    parsed = ToolCall.from_provider({"id": "c1", "function": {"name": "echo", "arguments": ""}})

    assert parsed.arguments == {}
    assert parsed.parse_error is None


def test_missing_arguments_serialize_as_empty_object():
    call = ToolCall(id="c1", name="noargs")

    assert json.loads(call.to_provider()["function"]["arguments"]) == {}
