"""Q4's ranker seam.

The default is a deterministic score order. An LLM reranker can be dropped in
behind the same interface; it is shown the shortlist with stable indices and may
only reorder them. Invalid output falls back to the deterministic order, so the
query path never loses evidence because the judge misbehaved.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence

from .llm import GraphLLM, GraphRagLLMError

RERANK_PROMPT = """You rank evidence for a medical literature question.
You are given numbered candidate facts. Return the indices of the most useful
candidates, best first. Use only the supplied list. Prefer facts that directly
answer the question and keep different relations. Never invent an index.
Return ONLY: {"keep": [1, 3, 2]}"""


class Reranker(Protocol):
    def __call__(self, question: str, candidates: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]: ...


def rerank(
    question: str,
    candidates: Sequence[dict[str, Any]],
    k: int,
    *,
    llm: GraphLLM | None = None,
    allow_llm: bool = True,
) -> list[dict[str, Any]]:
    """Return at most ``k`` candidates; LLM output can only reorder the shortlist."""
    ordered = sorted(candidates, key=lambda item: (-float(item.get("base_score", 0.0)), str(item.get("id", ""))))
    if k <= 0:
        return []
    if len(ordered) <= k or llm is None or not allow_llm:
        return ordered[:k]
    shortlist = ordered[: max(k * 2, k + 4)]
    lines = [
        f"{index}. {item.get('text', '')}"
        for index, item in enumerate(shortlist, start=1)
    ]
    try:
        payload = llm.complete_json(
            system=RERANK_PROMPT,
            user=f"Question: {question}\n\nCandidates:\n" + "\n".join(lines) + "\n\nReturn the keep list now.",
            max_tokens=1_500,
            attempts=1,
            expect=dict,
        )
        raw = payload.get("keep") if isinstance(payload, dict) else None
        selected: list[dict[str, Any]] = []
        seen: set[int] = set()
        for value in raw or []:
            try:
                index = int(value) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(shortlist) and index not in seen:
                seen.add(index)
                selected.append(shortlist[index])
        if selected:
            for item in ordered:
                if len(selected) >= k:
                    break
                if item not in selected:
                    selected.append(item)
            return selected[:k]
    except (GraphRagLLMError, ValueError):
        pass
    return ordered[:k]


def make_reranker(llm: GraphLLM | None, *, allow_llm: bool = True) -> Reranker:
    def _rerank(question: str, candidates: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        return rerank(question, candidates, k, llm=llm, allow_llm=allow_llm)

    return _rerank
