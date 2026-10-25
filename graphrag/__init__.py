"""GraphRAG: corpus → graph → query, with citations back to source chunks."""

from __future__ import annotations

from .backend import GraphRagBackend
from .config import GraphRagConfig, GraphRagConfigError
from .query import GraphRagEngine, GraphRagNotReady, QueryResult, answer, verify_citations

__all__ = [
    "GraphRagBackend",
    "GraphRagConfig",
    "GraphRagConfigError",
    "GraphRagEngine",
    "GraphRagNotReady",
    "QueryResult",
    "answer",
    "verify_citations",
]

__version__ = "0.1.0"
