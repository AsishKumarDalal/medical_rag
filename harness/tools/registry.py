"""Tool registry: registration, model-facing specs, and safe execution."""

from __future__ import annotations

from typing import Any, Iterable

from ..messages import ToolCall
from .base import Tool, ToolResult


class ToolRegistry:
    """Name → tool lookup with error containment.

    Execution never raises: unknown tools, unparsable arguments and tool bugs all
    come back as :class:`ToolResult` errors, which the loop hands to the model.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult.error(f"unknown tool {call.name!r}; available tools: {available}")
        if call.parse_error is not None:
            return ToolResult.error(f"{tool.name}: {call.parse_error}")
        try:
            result = tool.run(call.arguments)
        except Exception as exc:  # a tool bug is data for the model, not a crash
            return ToolResult.error(f"{tool.name} raised {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult.error(
                f"{tool.name} returned {type(result).__name__}, expected ToolResult"
            )
        return result

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
