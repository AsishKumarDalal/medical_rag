"""Q0–QG: answer a question from the saved graph, vectors and text store.

The pipeline is deliberately layered:

``probe → route → seeds (names ∪ vectors) → walk ∪ fetch → rank → narrate``

for local questions, and ``map → filter → reduce`` over community summaries for
global ones. Every fallback is explicit: a failed router defaults to local, no
seed falls back to fetch-only retrieval, and no evidence produces an honest
refusal. Every citation is checked against a real chunk or community.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

from harness.security import UNTRUSTED_NOTICE, wrap_untrusted

from .chunk import ChunkIndex, load_index
from .communities import load_communities
from .config import GraphRagConfig
from .embed import Embedder, VectorStore, cosine, embedder_from_store, load_vectors, top_matches
from .graph import Graph, GraphEdge, load_graph
from .llm import GraphLLM, GraphRagLLMError
from .normalize import normalize_entity
from .rerank import Reranker, make_reranker

LOCAL = "local"
GLOBAL = "global"
ABSTAIN = "abstain"
REFUSAL = "the evidence does not contain this"
CITATION = re.compile(r"\[([A-Za-z0-9_]+(?:::\d+)?)\]")
GLOBAL_HINTS = (
    "main themes",
    "themes",
    "overview",
    "summarize the corpus",
    "what is this corpus about",
    "main topics",
    "big picture",
    "across the corpus",
)
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "than",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "won",
    "world",
    "dose",
    "year",
    "take",
    "home",
}
GENERIC_TOKENS = {
    "disease",
    "diseases",
    "disorder",
    "disorders",
    "syndrome",
    "syndromes",
    "cancer",
    "cancers",
    "treatment",
    "treatments",
    "treat",
    "treats",
    "treated",
    "therapy",
    "therapies",
    "health",
    "medical",
    "medicine",
    "chronic",
    "acute",
    "patients",
    "patient",
    "people",
    "adults",
    "children",
    "symptoms",
    "symptom",
    "risk",
    "factors",
    "factor",
    "cause",
    "causes",
    "types",
    "type",
    "about",
    "which",
    "what",
    "does",
    "have",
    "with",
    "from",
    "that",
    "this",
}
ROUTER_PROMPT = """You route questions for a small medical literature corpus.
Choose exactly one route:
- LOCAL: the question names or implies specific entities (diseases, drugs, people,
  organisations, places) and needs facts or connections.
- GLOBAL: the question asks about themes, overview, coverage or the corpus as a whole.
- ABSTAIN: the corpus clearly cannot address it (for example a sports result).
Return ONLY: {"route": "LOCAL|GLOBAL|ABSTAIN", "reason": "short reason"}"""

NARRATOR_PROMPT = """You answer medical questions using ONLY the supplied evidence.
Do not use outside knowledge and do not give personal medical advice.
Every factual sentence must end with one or more citations of the exact chunk ids
shown below, in square brackets, for example [asthma::2]. If the evidence does not
contain the answer, say exactly: "The evidence does not contain this."
Be concise (at most 180 words).""" + "\n" + UNTRUSTED_NOTICE

GLOBAL_MAP_PROMPT = """You score one community summary for a question about a medical corpus.
Return ONLY {"score": 0-100, "points": ["short point", "..."]}.
Score 0 when the summary is unrelated. Do not invent anything outside the summary."""

GLOBAL_REDUCE_PROMPT = """You synthesize an answer from community summaries.
Use only the supplied summaries, cite community ids in square brackets like [C0],
and say when the corpus does not cover the question. At most 200 words.""" + "\n" + UNTRUSTED_NOTICE


class GraphRagNotReady(RuntimeError):
    """The index artifacts are missing or inconsistent."""


@dataclass(slots=True)
class QueryResult:
    answer: str
    citations: list[str]
    evidence: list[dict[str, Any]]
    mode: str
    route: str
    abstained: bool = False
    fallback: bool = False
    verification: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "evidence": self.evidence,
            "mode": self.mode,
            "route": self.route,
            "abstained": self.abstained,
            "fallback": self.fallback,
            "verification": self.verification,
            "stats": self.stats,
            "error": self.error,
        }


class GraphRagEngine:
    """Loaded index + model client; reusable across many questions."""

    def __init__(
        self,
        *,
        config: GraphRagConfig,
        llm: GraphLLM | None,
        graph: Graph,
        index: ChunkIndex,
        vectors: VectorStore,
        communities: dict[str, Any],
        reranker: Reranker | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.graph = graph
        self.index = index
        self.vectors = vectors
        self.communities = communities
        self.embedder: Embedder = embedder_from_store(config, vectors, llm)
        self.reranker = reranker or make_reranker(llm, allow_llm=config.llm_rerank)
        self.chunks = {chunk.id: chunk for chunk in index.chunks}
        self._current_question = ""

    @classmethod
    def load(cls, config: GraphRagConfig, llm: GraphLLM | None = None) -> "GraphRagEngine":
        missing = [
            str(path)
            for path in (config.graph_path, config.index_path, config.vectors_path, config.communities_path)
            if not Path(path).is_file()
        ]
        if missing:
            raise GraphRagNotReady(
                "the GraphRAG index is not built; missing " + ", ".join(missing)
            )
        return cls(
            config=config,
            llm=llm,
            graph=load_graph(config.graph_path),
            index=load_index(config.index_path),
            vectors=load_vectors(config.vectors_path),
            communities=load_communities(config.communities_path),
        )

    def answer(self, question: str, *, mode: str = "auto", k: int = 8) -> QueryResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")
        k = max(1, min(int(k), 20))
        started = monotonic()
        self._current_question = question
        route = self.route(question, mode=mode)
        if route == ABSTAIN:
            return self._abstain(question, route=route, started=started)
        probe = self.embedder.embed([question])[0]
        if route == GLOBAL:
            result = self._answer_global(question, probe, k=k)
        else:
            result = self._answer_local(question, probe, k=k)
        result.stats = {
            "seconds": round(monotonic() - started, 3),
            "llm": self.llm.stats.to_dict() if getattr(self.llm, "stats", None) is not None else None,
            "k": k,
        }
        return result

    # -- Q1 route ---------------------------------------------------------

    def route(self, question: str, *, mode: str = "auto") -> str:
        requested = (mode or "auto").strip().lower()
        if requested in {LOCAL, GLOBAL, ABSTAIN}:
            return requested
        if requested != "auto":
            raise ValueError(f"unknown mode {mode!r}; use auto, local, global or abstain")
        if self.llm is None:
            return GLOBAL if _looks_global(question) else LOCAL
        try:
            payload = self.llm.complete_json(
                system=ROUTER_PROMPT,
                user=f"Question: {question}\n\nReturn the route JSON now.",
                max_tokens=1_600,
                attempts=2,
                expect=dict,
            )
            route = str(payload.get("route") or "").strip().lower()
            if route in {LOCAL, GLOBAL, ABSTAIN}:
                return route
        except (GraphRagLLMError, ValueError):
            pass
        return LOCAL

    # -- local path -------------------------------------------------------

    def _answer_local(self, question: str, probe: list[float], *, k: int) -> QueryResult:
        seeds = self._seed_names(question)
        vector_seeds = self._seed_vectors(probe)
        for key, score in vector_seeds:
            if key not in seeds:
                seeds[key] = score
        seeds = dict(sorted(seeds.items(), key=lambda item: (-item[1], item[0]))[: self.config.max_seeds])

        walk_edges, distances = self._walk(set(seeds), hops=self.config.walk_hops)
        fetched = self._fetch_chunks(
            probe,
            k=max(k * 3, 12),
            min_matches=1 if seeds else 2,
        )
        candidates = self._rank_facts(probe, walk_edges, distances, set(seeds))
        facts = self.reranker(question, candidates, k)

        chunk_order = self._evidence_chunks(facts, fetched, k=k)
        if not chunk_order and not facts:
            return self._abstain(question, route=LOCAL, started=None, fallback=True)
        evidence_chunks = [
            {
                "id": chunk_id,
                "article": self.chunks[chunk_id].article,
                "text": self.chunks[chunk_id].text,
            }
            for chunk_id in chunk_order
            if chunk_id in self.chunks
        ]
        if self.llm is not None:
            try:
                text = self._narrate_local(question, facts, evidence_chunks)
                narration_error = None
            except (GraphRagLLMError, ValueError) as exc:
                text = _fallback_local_answer(facts, evidence_chunks)
                narration_error = f"{type(exc).__name__}: {exc}"
        else:
            text = _fallback_local_answer(facts, evidence_chunks)
            narration_error = None
        if _is_refusal(text):
            return self._abstain(question, route=LOCAL, started=None)

        valid_ids = set(self.chunks) | _community_ids(self.communities)
        citations = _citations(text, valid_ids)
        if not citations and evidence_chunks:
            citations = [chunk["id"] for chunk in evidence_chunks[:k]]
            if self.llm is not None:
                text = text.rstrip() + "\n\nSources: " + ", ".join(f"[{cid}]" for cid in citations)
        verification = verify_citations(text, valid_ids)
        return QueryResult(
            answer=text.strip(),
            citations=citations,
            evidence=[
                {
                    "kind": "fact",
                    "head": item["edge"].head,
                    "relation": item["edge"].relation,
                    "tail": item["edge"].tail,
                    "text": item["text"],
                    "mentions": item["edge"].mentions,
                    "source_chunks": list(item["edge"].source_chunks),
                    "score": round(float(item.get("base_score", 0.0)), 4),
                }
                for item in facts
            ]
            + [{"kind": "chunk", "id": chunk["id"], "text": chunk["text"][:500]} for chunk in evidence_chunks],
            mode=LOCAL,
            route="local_vector" if not seeds else LOCAL,
            fallback=not seeds,
            verification=verification,
            error=narration_error,
        )

    def _seed_names(self, question: str) -> dict[str, float]:
        normalized = normalize_entity(question)
        tokens = set(normalized.split())
        found: dict[str, float] = {}
        for key, node in self.graph.nodes.items():
            display = node.display.lower()
            score = 0.0
            if normalized == key or question.strip().lower() == display:
                score = 1000.0
            elif _phrase_in(key, normalized) or _phrase_in(display, question.lower()):
                score = 600.0 + node.mentions
            else:
                meaningful = [
                    token
                    for token in key.split()
                    if token not in GENERIC_TOKENS and len(token) >= 4
                ]
                if meaningful and all(token in tokens for token in meaningful):
                    score = 250.0 + node.mentions
                elif len(meaningful) == 1 and meaningful[0] in tokens:
                    score = 200.0 + node.mentions
            if score:
                found[key] = score
        return dict(sorted(found.items(), key=lambda item: (-item[1], item[0]))[: self.config.max_seeds])

    def _seed_vectors(self, probe: list[float]) -> list[tuple[str, float]]:
        threshold = _effective(self.vectors, self.config.seed_threshold)
        matches = top_matches(
            probe,
            self.vectors.nodes,
            k=self.config.max_seeds,
            threshold=threshold,
        )
        if self.vectors.backend.startswith("hash"):
            # Hashing is lexical, so a score without any shared content word is
            # collision noise. Real embedding backends do not need this guard.
            matches = [
                (key, score)
                for key, score in matches
                if self._name_overlap(key)
            ]
        return matches

    def _name_overlap(self, key: str) -> bool:
        return bool(_content_tokens(key) & _content_tokens(self._current_question))

    def _walk(self, seeds: set[str], *, hops: int) -> tuple[list[GraphEdge], dict[str, int]]:
        if not seeds:
            return [], {}
        adjacency = self.graph.adjacency()
        visited = set(seeds)
        distances = {key: 0 for key in seeds}
        edges: dict[str, GraphEdge] = {}
        frontier = sorted(seeds)
        for depth in range(1, max(1, hops) + 1):
            next_frontier: list[str] = []
            for key in frontier:
                for neighbour, edge in adjacency.get(key, []):
                    if edge.id not in edges:
                        edges[edge.id] = edge
                    if neighbour not in visited:
                        visited.add(neighbour)
                        distances[neighbour] = depth
                        next_frontier.append(neighbour)
                    elif depth < distances.get(neighbour, depth):
                        distances[neighbour] = depth
            frontier = sorted(next_frontier)
            if not frontier or len(visited) > 120 or len(edges) > 500:
                break
        return list(edges.values()), distances

    def _fetch_chunks(
        self,
        probe: list[float],
        *,
        k: int,
        min_matches: int = 1,
    ) -> list[tuple[str, float]]:
        threshold = _effective(self.vectors, self.config.chunk_threshold)
        candidates = top_matches(
            probe,
            self.vectors.chunks,
            k=max(k * 6, 24),
            threshold=threshold,
        )
        if self.vectors.backend.startswith("hash"):
            # Hashing is lexical, not semantic. Rank the honest signal: shared
            # question/chunk content words, with cosine as a tie-breaker.
            question_tokens = _lexical_tokens(self._current_question)
            rescored: list[tuple[str, float]] = []
            for chunk_id, score in candidates:
                chunk_tokens = _lexical_tokens(self.chunks[chunk_id].text)
                if not question_tokens:
                    continue
                matched = len(question_tokens & chunk_tokens)
                if matched < min(min_matches, len(question_tokens)):
                    continue
                overlap = matched / len(question_tokens)
                rescored.append((chunk_id, 0.45 * score + 0.55 * overlap))
            rescored.sort(key=lambda item: (-item[1], item[0]))
            candidates = rescored
        return candidates[: max(k * 3, 12)]

    def _rank_facts(
        self,
        probe: list[float],
        edges: Sequence[GraphEdge],
        distances: dict[str, int],
        seeds: set[str],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for edge in edges:
            vector = self.vectors.facts.get(edge.id)
            semantic = cosine(probe, vector) if vector else 0.0
            touches_seed = edge.head in seeds or edge.tail in seeds
            distance = min(distances.get(edge.head, 99), distances.get(edge.tail, 99))
            structural = 1.0 if touches_seed else (0.6 if distance <= 2 else 0.2)
            mention = min(1.0, edge.mentions / 10.0)
            score = 0.55 * semantic + 0.30 * structural + 0.10 * mention + 0.05 * min(1.0, len(edge.source_chunks) / 3.0)
            candidates.append(
                {
                    "id": edge.id,
                    "text": edge.render(self.graph),
                    "base_score": score,
                    "edge": edge,
                    "distance": distance,
                }
            )
        return candidates

    def _evidence_chunks(
        self,
        facts: Sequence[dict[str, Any]],
        fetched: Sequence[tuple[str, float]],
        *,
        k: int,
    ) -> list[str]:
        ordered: list[str] = []
        for chunk_id, _score in fetched:
            if chunk_id in self.chunks and chunk_id not in ordered:
                ordered.append(chunk_id)
        for fact in facts:
            for chunk_id in fact["edge"].source_chunks:
                if chunk_id in self.chunks and chunk_id not in ordered:
                    ordered.append(chunk_id)
                    break
        return ordered[:k]

    def _narrate_local(
        self,
        question: str,
        facts: Sequence[dict[str, Any]],
        chunks: Sequence[dict[str, Any]],
    ) -> str:
        fact_lines = [f"- {fact['text']} (sources: {', '.join(fact['edge'].source_chunks)})" for fact in facts]
        chunk_lines = [
            wrap_untrusted(chunk["id"], chunk["text"][:2_500]) for chunk in chunks[:8]
        ]
        user = (
            f"Question: {question}\n\n"
            "Ranked facts:\n"
            + ("\n".join(fact_lines) or "- none")
            + "\n\nSource passages:\n"
            + ("\n".join(chunk_lines) or "(none)")
            + "\n\nWrite the cited answer now. If the evidence does not contain the answer, "
            "say exactly: The evidence does not contain this."
        )
        return self.llm.complete_text(  # type: ignore[union-attr]
            system=NARRATOR_PROMPT,
            user=user,
            max_tokens=2_600,
            temperature=0.0,
        ).strip()

    # -- global path ------------------------------------------------------

    def _answer_global(self, question: str, probe: list[float], *, k: int) -> QueryResult:
        communities = list(self.communities.get("communities", []))
        if not communities:
            return self._abstain(question, route=GLOBAL, started=None, fallback=True)
        scored: list[tuple[float, dict[str, Any], list[str]]] = []
        for community in communities:
            score, points = self._map_community(question, community, probe)
            scored.append((score, community, points))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id"))))
        kept = [item for item in scored if item[0] >= 35][: max(2, min(k, 4))]
        if not kept:
            kept = [item for item in scored[:2] if item[0] > 0]
        if not kept:
            return self._abstain(question, route=GLOBAL, started=None, fallback=True)
        evidence = [
            {
                "kind": "community",
                "id": community["id"],
                "members": community.get("member_names", []),
                "summary": community.get("summary", ""),
                "score": round(score, 1),
                "points": points,
            }
            for score, community, points in kept
        ]
        if self.llm is not None:
            try:
                text = self._reduce_global(question, evidence)
                error = None
            except (GraphRagLLMError, ValueError) as exc:
                text = _fallback_global_answer(evidence)
                error = f"{type(exc).__name__}: {exc}"
        else:
            text = _fallback_global_answer(evidence)
            error = None
        if _is_refusal(text):
            return self._abstain(question, route=GLOBAL, started=None)
        valid_ids = _community_ids(self.communities) | set(self.chunks)
        citations = _citations(text, valid_ids)
        if not citations:
            citations = [item["id"] for item in evidence]
            text = text.rstrip() + "\n\nSources: " + ", ".join(f"[{cid}]" for cid in citations)
        return QueryResult(
            answer=text.strip(),
            citations=citations,
            evidence=evidence,
            mode=GLOBAL,
            route=GLOBAL,
            verification=verify_citations(text, valid_ids),
            error=error,
        )

    def _map_community(
        self,
        question: str,
        community: dict[str, Any],
        probe: list[float],
    ) -> tuple[float, list[str]]:
        if self.llm is not None:
            try:
                payload = self.llm.complete_json(
                    system=GLOBAL_MAP_PROMPT,
                    user=(
                        f"Question: {question}\n\nCommunity {community.get('id')}:\n"
                        f"{community.get('summary', '')}\n\nReturn the score JSON now."
                    ),
                    max_tokens=1_500,
                    attempts=1,
                    expect=dict,
                )
                score = float(payload.get("score", 0))
                points = [str(point) for point in payload.get("points") or [] if str(point).strip()][:5]
                return max(0.0, min(100.0, score)), points
            except (GraphRagLLMError, ValueError, TypeError):
                pass
        score = _lexical_score(question, str(community.get("summary", "")))
        if _looks_global(question):
            score = max(score, 45.0)
        return score, []

    def _reduce_global(self, question: str, evidence: Sequence[dict[str, Any]]) -> str:
        blocks = [
            wrap_untrusted(str(item["id"]), str(item.get("summary", "")), max_chars=2_500)
            for item in evidence
        ]
        return self.llm.complete_text(  # type: ignore[union-attr]
            system=GLOBAL_REDUCE_PROMPT,
            user=f"Question: {question}\n\nCommunity summaries:\n" + "\n\n".join(blocks)
            + "\n\nWrite the synthesized answer now.",
            max_tokens=2_500,
            temperature=0.0,
        ).strip()

    # -- refusal ----------------------------------------------------------

    def _abstain(
        self,
        question: str,
        *,
        route: str,
        started: float | None,
        fallback: bool = False,
    ) -> QueryResult:
        mode = ABSTAIN
        return QueryResult(
            answer="The evidence does not contain this. The corpus does not provide enough information to answer the question.",
            citations=[],
            evidence=[],
            mode=mode,
            route=route,
            abstained=True,
            fallback=fallback,
            stats={"seconds": round(monotonic() - (started or monotonic()), 3), "k": 0},
        )


def answer(
    question: str,
    *,
    mode: str = "auto",
    k: int = 8,
    config: GraphRagConfig | None = None,
    llm: GraphLLM | None = None,
) -> QueryResult:
    """Public entry point used by the harness and ``python -m graphrag.ask``."""
    active = config or GraphRagConfig.from_env()
    client = llm if llm is not None else GraphLLM(
        active.llm,
        embed_model=active.embed_model,
        embed_base_url=active.embed_base_url,
        embed_api_key=active.embed_api_key,
    )
    return GraphRagEngine.load(active, client).answer(question, mode=mode, k=k)


def verify_citations(text: str, valid_ids: set[str]) -> dict[str, Any]:
    """Q6 light check: every bracket citation must resolve to a real artifact."""
    cited = CITATION.findall(text or "")
    unsupported = sorted({citation for citation in cited if citation not in valid_ids})
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text or "") if sentence.strip()]
    uncited = [
        sentence
        for sentence in sentences
        if len(sentence.split()) >= 8 and not CITATION.search(sentence)
    ]
    return {
        "citations": cited,
        "unsupported": unsupported,
        "uncited_sentences": uncited[:10],
        "ok": not unsupported,
    }


def _citations(text: str, valid_ids: set[str]) -> list[str]:
    seen: list[str] = []
    for citation in CITATION.findall(text or ""):
        if citation in valid_ids and citation not in seen:
            seen.append(citation)
    return seen


def _phrase_in(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])", haystack) is not None


def _looks_global(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in GLOBAL_HINTS)


def _effective(store: VectorStore, threshold: float) -> float:
    """Hashing vectors are lexical, so their cosine scale is lower than an embedding model's."""
    return threshold * 0.25 if store.backend.startswith("hash") else threshold


def _community_ids(communities: dict[str, Any]) -> set[str]:
    return {str(community.get("id")) for community in communities.get("communities", [])}


def _is_refusal(text: str) -> bool:
    return REFUSAL in (text or "").lower()


def _lexical_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for token in normalize_entity(text).split():
        if len(token) < 4:
            continue
        token = _stem(token)
        if token not in QUERY_STOPWORDS:
            found.add(token)
    return found


def _content_tokens(text: str) -> set[str]:
    return _lexical_tokens(text) - GENERIC_TOKENS


def _stem(token: str) -> str:
    """A tiny plural stripper, not a linguistic stemmer."""
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _lexical_score(question: str, text: str) -> float:
    q_tokens = set(normalize_entity(question).split())
    t_tokens = set(normalize_entity(text).split())
    if not q_tokens or not t_tokens:
        return 0.0
    return 100.0 * len(q_tokens & t_tokens) / len(q_tokens)


def _fallback_local_answer(
    facts: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
) -> str:
    if facts:
        lines = ["The corpus contains the following evidence:"]
        for fact in facts:
            sources = " ".join(f"[{chunk}]" for chunk in fact["edge"].source_chunks[:2])
            lines.append(f"- {fact['text']} {sources}".rstrip())
        return "\n".join(lines)
    if chunks:
        lines = ["The corpus contains the following passages:"]
        for chunk in chunks[:3]:
            excerpt = " ".join(chunk["text"].split())[:400]
            lines.append(f"- {excerpt} [{chunk['id']}]")
        return "\n".join(lines)
    return "The evidence does not contain this."


def _fallback_global_answer(evidence: Sequence[dict[str, Any]]) -> str:
    lines = ["The corpus covers these themes:"]
    for item in evidence:
        summary = " ".join(str(item.get("summary", "")).split())[:500]
        lines.append(f"- [{item['id']}] {summary}")
    return "\n".join(lines)
