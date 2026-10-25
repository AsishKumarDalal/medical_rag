"""The RAG tool: the harness's first real tool.

The pipeline does not exist yet (phase G3), so the tool ships as a stub with its
final signature. That lets the system prompt, the registry and the loop be written
once against the contract they will keep.
"""

from __future__ import annotations

from typing import Any, Protocol

from .base import Tool, ToolResult


class RagBackend(Protocol):
    """What the tool needs from the retrieval pipeline."""

    def answer(self, question: str, *, mode: str = "auto", k: int = 8) -> Any: ...


class SearchMedicalLiteratureTool(Tool):
    name = "search_medical_literature"
    description = (
        "Search the indexed medical literature and return a short, cited answer. "
        "Use it for any medical or health question. Answers cite chunk ids; if the "
        "corpus does not contain the answer, the tool says so instead of guessing."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 3,
                "description": "The question to answer from the literature.",
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "local", "global"],
                "description": (
                    "auto picks local (named entities) or global (themes) itself; "
                    "force a mode only when you know better."
                ),
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of evidence items to use (default 8).",
            },
        },
        "required": ["question"],
        "additionalProperties": False,
    }

    def __init__(self, backend: RagBackend | None = None) -> None:
        self._backend = backend

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        question = arguments.get("question", "").strip()
        if not question:
            return ToolResult.error("search_medical_literature: 'question' must not be empty")
        if self._backend is None:
            return ToolResult.error(
                "search_medical_literature: the retrieval pipeline is not wired yet "
                "(phase G3); no corpus is available"
            )

        mode = arguments.get("mode", "auto")
        k = arguments.get("k", 8)
        try:
            result = self._backend.answer(question, mode=mode, k=k)
        except Exception as exc:  # pipeline failures are reported, not fatal
            return ToolResult.error(f"search_medical_literature failed: {type(exc).__name__}: {exc}")

        text = _render(result)
        citations = list(getattr(result, "citations", []) or [])
        return ToolResult.success(text, data={"citations": citations, "mode": getattr(result, "mode", mode)})


def _render(result: Any) -> str:
    answer = str(getattr(result, "answer", result) or "").strip()
    citations = list(getattr(result, "citations", []) or [])
    mode = getattr(result, "mode", None)
    lines = [answer or "The corpus did not yield an answer."]
    if citations:
        lines.append("")
        lines.append("Citations: " + ", ".join(str(citation) for citation in citations))
    if mode:
        lines.append(f"(mode: {mode})")
    return "\n".join(lines)
