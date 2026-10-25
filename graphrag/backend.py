"""The harness-facing adapter for the GraphRAG pipeline.

Importing this module is cheap; the index is loaded on the first question, so the
CLI can start even before the pipeline has been built. A missing index becomes a
tool error ("not built yet") rather than a process failure.
"""

from __future__ import annotations

from typing import Any

from .config import GraphRagConfig
from .llm import GraphLLM
from .query import GraphRagEngine, GraphRagNotReady, QueryResult


_UNSET = object()


class GraphRagBackend:
    """Stable contract used by ``harness.tools.rag``."""

    def __init__(
        self,
        *,
        config: GraphRagConfig | None = None,
        llm: GraphLLM | None | object = _UNSET,
    ) -> None:
        self.config = config or GraphRagConfig.from_env()
        if llm is _UNSET:
            self.llm = GraphLLM(
                self.config.llm,
                embed_model=self.config.embed_model,
                embed_base_url=self.config.embed_base_url,
                embed_api_key=self.config.embed_api_key,
            )
        else:
            self.llm = llm  # type: ignore[assignment]
        self._engine: GraphRagEngine | None = None

    @classmethod
    def from_env(cls) -> "GraphRagBackend":
        return cls()

    @property
    def ready(self) -> bool:
        return all(
            path.is_file()
            for path in (
                self.config.graph_path,
                self.config.index_path,
                self.config.vectors_path,
                self.config.communities_path,
            )
        )

    def engine(self) -> GraphRagEngine:
        if self._engine is None:
            self._engine = GraphRagEngine.load(self.config, self.llm)
        return self._engine

    def answer(self, question: str, *, mode: str = "auto", k: int = 8) -> QueryResult:
        return self.engine().answer(question, mode=mode, k=k)

    def close(self) -> None:
        self._engine = None


__all__ = ["GraphRagBackend", "GraphRagNotReady", "QueryResult", "GraphRagConfig", "GraphLLM"]
