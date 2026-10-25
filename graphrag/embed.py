"""I11 embeddings: nodes, facts and chunks in one on-disk vector store.

Two backends behind one interface:

* **OpenAI-compatible** — used when ``GRAPHRAG_EMBED_MODEL`` is configured.
* **Deterministic hashing** — a dependency-free lexical fallback used when the
  chat endpoint has no embeddings route. It hashes word tokens and character
  trigrams into a fixed-dimension signed vector and L2-normalises it. It is not
  semantic, but it is stable, offline and good enough to keep the two-door /
  fetch fallbacks working. ``vectors.json`` records which backend produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from .chunk import ChunkIndex
from .config import GraphRagConfig
from .graph import Graph, GraphEdge
from .llm import GraphLLM, GraphRagLLMError

VECTORS_VERSION = 1
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    backend: str
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class HashEmbedder:
    """Signed feature hashing over words and character trigrams."""

    dim: int = 256
    backend: str = "hash-ngram"
    model: str = "deterministic-blake2b"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        lowered = text.lower()
        tokens = _TOKEN.findall(lowered)
        counts = Counter(tokens)
        for token, count in counts.items():
            weight = 1.0 + math.log1p(count)
            _add_feature(vector, token, weight, self.dim)
            padded = f"^{token}$"
            for index in range(max(1, len(padded) - 2)):
                _add_feature(vector, padded[index : index + 3], 0.35, self.dim)
        # Character ngrams over the whole string help with typos and morphology.
        compact = re.sub(r"\s+", " ", lowered)
        for index in range(max(0, len(compact) - 2)):
            _add_feature(vector, compact[index : index + 3], 0.15, self.dim)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [round(value / norm, 7) for value in vector]
        return vector


class GraphEmbedder:
    """Adapter that routes OpenAI-compatible embeddings through GraphLLM."""

    backend = "openai-compatible"

    def __init__(self, llm: GraphLLM, model: str, dim: int) -> None:
        self.llm = llm
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.llm.embed(texts)
        if vectors and len(vectors[0]) != self.dim:
            raise GraphRagLLMError(
                f"embedding dimension is {len(vectors[0])}, expected {self.dim}; "
                "set GRAPHRAG_EMBED_DIM to match the model"
            )
        return vectors


@dataclass(slots=True)
class VectorStore:
    backend: str
    model: str
    dim: int
    nodes: dict[str, list[float]]
    facts: dict[str, list[float]]
    chunks: dict[str, list[float]]
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VECTORS_VERSION,
            "backend": self.backend,
            "model": self.model,
            "dim": self.dim,
            "built_at": _now(),
            "fallback_reason": self.fallback_reason,
            "counts": {
                "nodes": len(self.nodes),
                "facts": len(self.facts),
                "chunks": len(self.chunks),
            },
            "nodes": self.nodes,
            "facts": self.facts,
            "chunks": self.chunks,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorStore":
        if int(payload.get("version", 0)) != VECTORS_VERSION:
            raise ValueError(f"unknown vectors version {payload.get('version')!r}")
        return cls(
            backend=str(payload.get("backend") or "unknown"),
            model=str(payload.get("model") or "unknown"),
            dim=int(payload.get("dim", 0)),
            nodes={key: [float(v) for v in vector] for key, vector in payload.get("nodes", {}).items()},
            facts={key: [float(v) for v in vector] for key, vector in payload.get("facts", {}).items()},
            chunks={key: [float(v) for v in vector] for key, vector in payload.get("chunks", {}).items()},
            fallback_reason=payload.get("fallback_reason"),
        )


def build_vectors(
    config: GraphRagConfig,
    graph: Graph,
    index: ChunkIndex,
    llm: GraphLLM | None,
    *,
    force: bool = False,
) -> VectorStore:
    config.ensure_dirs()
    embedder, fallback_reason = choose_embedder(config, llm)
    previous = None if force else _load_previous(config.vectors_path, embedder)
    node_texts = {key: _node_text(graph.nodes[key]) for key in sorted(graph.nodes)}
    fact_texts = {
        edge.id: edge.render(graph)
        for edge in sorted(graph.edges, key=lambda edge: edge.id)
    }
    chunk_texts = {chunk.id: chunk.text for chunk in index.chunks}

    store = VectorStore(
        backend=embedder.backend,
        model=embedder.model,
        dim=embedder.dim,
        nodes={},
        facts={},
        chunks={},
        fallback_reason=fallback_reason,
    )
    for name, texts, existing in (
        ("nodes", node_texts, previous.nodes if previous else {}),
        ("facts", fact_texts, previous.facts if previous else {}),
        ("chunks", chunk_texts, previous.chunks if previous else {}),
    ):
        mapping: dict[str, list[float]] = {}
        missing = [key for key in texts if key not in existing]
        for key in texts:
            if key in existing:
                mapping[key] = existing[key]
        if missing:
            vectors = embedder.embed([texts[key] for key in missing])
            mapping.update(dict(zip(missing, vectors)))
        setattr(store, name, mapping)
        # Persist after each inventory so a crash can resume the rest.
        _write_store(store, config.vectors_path)
    return store


def load_vectors(path: str | Path) -> VectorStore:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"vectors not found: {target}; run the embed stage first")
    payload = json.loads(target.read_text(encoding="utf-8"))
    return VectorStore.from_dict(payload)


def choose_embedder(config: GraphRagConfig, llm: GraphLLM | None) -> tuple[Embedder, str | None]:
    """Use the configured embeddings route, else the deterministic fallback."""
    if config.embed_model and llm is not None:
        configured = GraphLLM(
            config.llm,
            sdk=getattr(llm, "_sdk", None),
            embed_model=config.embed_model,
            embed_base_url=config.embed_base_url,
            embed_api_key=config.embed_api_key,
        )
        embedder = GraphEmbedder(configured, config.embed_model, config.embed_dim)
        try:
            embedder.embed(["dimension probe"])
            return embedder, None
        except GraphRagLLMError as exc:
            return HashEmbedder(dim=config.embed_dim), f"configured embeddings failed: {exc}"
    return HashEmbedder(dim=config.embed_dim), "no GRAPHRAG_EMBED_MODEL configured"


def embedder_from_store(
    config: GraphRagConfig,
    store: VectorStore,
    llm: GraphLLM | None,
) -> Embedder:
    """Build the embedder that matches the vectors on disk."""
    if store.backend == "openai-compatible":
        if llm is None:
            raise GraphRagLLMError("the index uses OpenAI embeddings but no model client is available")
        embedder = GraphEmbedder(llm, store.model, store.dim)
        if llm.embed_model != store.model:
            embedder = GraphEmbedder(
                GraphLLM(
                    config.llm,
                    sdk=getattr(llm, "_sdk", None),
                    embed_model=store.model,
                    embed_base_url=config.embed_base_url,
                    embed_api_key=config.embed_api_key,
                ),
                store.model,
                store.dim,
            )
        return embedder
    return HashEmbedder(dim=store.dim)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def top_matches(
    query: Sequence[float],
    vectors: dict[str, Sequence[float]],
    *,
    k: int = 8,
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    scored = [(key, cosine(query, vector)) for key, vector in vectors.items()]
    scored = [item for item in scored if item[1] >= threshold]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


def _add_feature(vector: list[float], feature: str, weight: float, dim: int) -> None:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    index = number % dim
    sign = 1.0 if (number >> 8) & 1 else -1.0
    vector[index] += sign * weight


def _node_text(node: Any) -> str:
    aliases = ", ".join(node.aliases)
    return (
        f"{node.display}. Type: {node.type}. Mentions: {node.mentions}. "
        f"Aliases: {aliases or 'none'}."
    )


def _load_previous(path: Path, embedder: Embedder) -> VectorStore | None:
    if not path.is_file():
        return None
    try:
        previous = load_vectors(path)
    except (ValueError, json.JSONDecodeError):
        return None
    if previous.backend != embedder.backend or previous.model != embedder.model or previous.dim != embedder.dim:
        return None
    return previous


def _write_store(store: VectorStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = store.to_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def render_fact(graph: Graph, head: str, relation: str, tail: str) -> str:
    head_name = graph.nodes[head].display if head in graph.nodes else head
    tail_name = graph.nodes[tail].display if tail in graph.nodes else tail
    return f"{head_name} --{relation}--> {tail_name}"


def fact_texts(graph: Graph) -> dict[str, str]:
    return {edge.id: edge.render(graph) for edge in graph.edges}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
