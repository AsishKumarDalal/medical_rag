"""The RAG tool: the harness's only real tool.

``GraphRagBackend`` is injected so tests can exercise the tool contract without a
model or network. When no backend is supplied the tool reports that the index is
not wired, rather than pretending an answer exists.
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
                "search_medical_literature: the retrieval pipeline is not wired; "
                "no corpus is available"
            )

        mode = arguments.get("mode", "auto")
        k = arguments.get("k", 8)
        try:
            result = self._backend.answer(question, mode=mode, k=k)
        except Exception as exc:  # pipeline failures are reported, not fatal
            return ToolResult.error(f"search_medical_literature failed: {type(exc).__name__}: {exc}")

        text = _render(result)
        citations = list(getattr(result, "citations", []) or [])
        return ToolResult.success(
            text,
            data={
                "citations": citations,
                "evidence_ids": _evidence_ids(result),
                "mode": getattr(result, "mode", mode),
                "abstained": bool(getattr(result, "abstained", False)),
            },
        )


def _evidence_ids(result: Any) -> list[str]:
    """Stable ids the answer was built from, without sending full evidence back."""
    ids: list[str] = []
    for item in getattr(result, "evidence", []) or []:
        candidates: list[str] = []
        if item.get("kind") == "fact":
            candidates.extend(str(chunk) for chunk in item.get("source_chunks", []) or [])
        elif item.get("id") is not None:
            candidates.append(str(item["id"]))
        for candidate in candidates:
            if candidate not in ids:
                ids.append(candidate)
    return ids


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
