"""Message and tool-call types, and their OpenAI-compatible wire form.

The harness stores conversation as these objects and converts at the boundary,
so the provider format never leaks into the loop, the session store or tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str | None]:
    """Parse a tool-call argument string into a dict, reporting failures."""
    if not raw.strip():
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments ({exc.msg} at position {exc.pos})"
    if not isinstance(parsed, dict):
        return {}, f"arguments must be a JSON object, got {type(parsed).__name__}"
    return parsed, None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""
    parse_error: str | None = None

    def to_provider(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments),
            },
        }

    @classmethod
    def from_provider(cls, data: dict[str, Any]) -> "ToolCall":
        function = data.get("function") or {}
        raw = function.get("arguments")
        if isinstance(raw, dict):
            arguments, raw_text, parse_error = raw, json.dumps(raw), None
        else:
            raw_text = "" if raw is None else str(raw)
            arguments, parse_error = _parse_arguments(raw_text)
        return cls(
            id=str(data.get("id") or ""),
            name=str(function.get("name") or ""),
            arguments=arguments,
            raw_arguments=raw_text,
            parse_error=parse_error,
        )


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_provider(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [call.to_provider() for call in self.tool_calls]
        if self.tool_call_id is not None:
            data["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        return data

    @classmethod
    def from_provider(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data.get("content") or "",
            tool_calls=[ToolCall.from_provider(call) for call in data.get("tool_calls") or []],
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


def system(text: str) -> Message:
    return Message(role="system", content=text)


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role="assistant", content=text, tool_calls=tool_calls or [])


def tool_result(call: ToolCall, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call.id, name=call.name)
