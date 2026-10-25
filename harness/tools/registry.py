"""Tool registry: registration, model-facing specs, and safe execution."""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..messages import ToolCall
from ..security import SecurityPolicy
from .base import Tool, ToolResult
from .validation import validate_arguments

# Tool results are fed back into the prompt, so they are capped before that.
MAX_RESULT_CHARS = 8_000


class ToolRegistry:
    """Name → tool lookup with error containment.

    Execution never raises: unknown tools, unparsable arguments and tool bugs all
    come back as :class:`ToolResult` errors, which the loop hands to the model.
    """

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        max_result_chars: int = MAX_RESULT_CHARS,
        policy: SecurityPolicy | None = None,
    ) -> None:
        self.policy = policy
        if policy is not None:
            max_result_chars = policy.max_result_chars
        if max_result_chars <= 0:
            raise ValueError("max_result_chars must be positive")
        self.max_result_chars = max_result_chars
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
        """Only policy-allowed tools are advertised to the model."""
        if self.policy is not None and self.policy.allowed_tools is not None:
            return [tool.spec() for name, tool in self._tools.items() if self.policy.allows(name)]
        return [tool.spec() for tool in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        if self.policy is not None and not self.policy.allows(call.name):
            return ToolResult.error(f"tool {call.name!r} is not permitted by this session's policy")
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult.error(f"unknown tool {call.name!r}; available tools: {available}")
        if call.parse_error is not None:
            return ToolResult.error(f"{tool.name}: {call.parse_error}")
        if self.policy is not None:
            size = len(call.raw_arguments or json.dumps(call.arguments))
            if size > self.policy.max_argument_chars:
                return ToolResult.error(
                    f"{tool.name}: arguments are {size} characters, above the "
                    f"{self.policy.max_argument_chars}-character limit"
                )
        problems = validate_arguments(tool.parameters, call.arguments)
        if problems:
            return ToolResult.error(
                f"{tool.name}: invalid arguments — " + "; ".join(problems)
            )
        try:
            result = tool.run(call.arguments)
        except Exception as exc:  # a tool bug is data for the model, not a crash
            return ToolResult.error(f"{tool.name} raised {type(exc).__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return ToolResult.error(
                f"{tool.name} returned {type(result).__name__}, expected ToolResult"
            )
        return self._cap(result, tool.name)

    def _cap(self, result: ToolResult, name: str) -> ToolResult:
        """Truncate oversized content, keeping a marker the model can act on."""
        limit = self.max_result_chars
        if len(result.content) <= limit:
            return result
        omitted = len(result.content) - limit
        content = (
            result.content[:limit]
            + f"\n\n[truncated: {omitted} of {len(result.content)} characters omitted from "
            + f"{name}; narrow the request for a smaller result]"
        )
        return ToolResult(ok=result.ok, content=content, data=result.data)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
