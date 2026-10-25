"""The echo tool: proves the loop can call a tool and read its result."""

from __future__ import annotations

from typing import Any

from .base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = (
        "Return the given text unchanged. Used to verify that tool calling works; "
        "it has no side effects and no other purpose."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to return unchanged."},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        text = arguments.get("text")
        if not isinstance(text, str):
            return ToolResult.error("echo: 'text' must be a string")
        return ToolResult.success(text)
