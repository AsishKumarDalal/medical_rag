"""The tool contract: what the loop may call, and what it gets back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolResult:
    """Outcome of one tool call, formatted for the model."""

    ok: bool
    content: str
    data: Any = None

    @classmethod
    def success(cls, content: str, data: Any = None) -> "ToolResult":
        return cls(ok=True, content=content, data=data)

    @classmethod
    def error(cls, content: str, data: Any = None) -> "ToolResult":
        return cls(ok=False, content=content, data=data)


class Tool:
    """Base class for a callable tool.

    Subclasses set ``name``, ``description`` and a JSON-schema ``parameters``
    object, and implement :meth:`run`. ``run`` may raise: the registry turns any
    exception into an error result so a broken tool cannot end the turn.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        """The OpenAI-compatible function description sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"
