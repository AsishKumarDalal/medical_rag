"""G2: query pipeline, offline with the deterministic embedding fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag.chunk import Chunk, ChunkIndex
from graphrag.communities import build_communities
from graphrag.config import GraphRagConfig
from graphrag.embed import HashEmbedder, VectorStore, cosine
from graphrag.graph import Graph, GraphEdge, GraphNode
from graphrag.llm import GraphRagLLMError
from graphrag.query import GraphRagEngine, verify_citations
from graphrag.rerank import rerank


def config(tmp_path: Path) -> GraphRagConfig:
    from harness.config import Config

    return GraphRagConfig(
        corpus_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        llm=Config(base_url="https://stub.test/v1", api_key="k", model="stub"),
        embed_dim=96,
        community_min_size=2,
        seed_threshold=0.65,
        chunk_threshold=0.30,
        llm_rerank=False,
    )


def fixture_graph() -> Graph:
    nodes = {
        "asthma": GraphNode("asthma", "Asthma", "DISEASE", 6),
        "inhaled corticosteroids": GraphNode(
            "inhaled corticosteroids", "Inhaled corticosteroids", "DRUG", 3
        ),
        "wheezing": GraphNode("wheezing", "Wheezing", "SYMPTOM", 2),
        "diabetes": GraphNode("diabetes", "Diabetes", "DISEASE", 4),
        "insulin": GraphNode("insulin", "Insulin", "DRUG", 2),
        "smoking": GraphNode("smoking", "Smoking", "RISK_FACTOR", 2),
    }
    edges = [
        GraphEdge("inhaled corticosteroids", "TREATS", "asthma", 3, ["asthma::1", "asthma::2"]),
        GraphEdge("asthma", "CAUSES", "wheezing", 2, ["asthma::1"]),
        GraphEdge("smoking", "RISK_FACTOR_FOR", "asthma", 2, ["asthma::2"]),
        GraphEdge("insulin", "TREATS", "diabetes", 2, ["diabetes::1"]),
    ]
    return Graph(nodes=nodes, edges=edges, chunk_count=3, fact_count=9)


def fixture_index() -> ChunkIndex:
    chunks = [
        Chunk("asthma::1", "asthma", "Asthma", 1, "Inhaled corticosteroids are used to treat asthma. Asthma causes wheezing."),
        Chunk("asthma::2", "asthma", "Asthma", 2, "Smoking is a risk factor for asthma. Inhaled corticosteroids reduce airway inflammation."),
        Chunk("diabetes::1", "diabetes", "Diabetes", 1, "Insulin is used to treat diabetes mellitus."),
    ]
    return ChunkIndex(
        articles=[{"id": "asthma", "title": "Asthma", "chunks": []}, {"id": "diabetes", "title": "Diabetes", "chunks": []}],
        chunks=chunks,
        target=400,
        overlap=60,
    )


def fixture_engine(tmp_path: Path, llm=None) -> GraphRagEngine:
    graph = fixture_graph()
    index = fixture_index()
    embedder = HashEmbedder(dim=96)
    nodes = {key: embedder.embed([f"{node.display}. Type: {node.type}. Mentions: {node.mentions}."])[0] for key, node in graph.nodes.items()}
    facts = {edge.id: embedder.embed([edge.render(graph)])[0] for edge in graph.edges}
    chunks = {chunk.id: embedder.embed([chunk.text])[0] for chunk in index.chunks}
    vectors = VectorStore(
        backend=embedder.backend,
        model=embedder.model,
        dim=96,
        nodes=nodes,
        facts=facts,
        chunks=chunks,
    )
    communities = build_communities(graph, None, min_size=2).payload
    return GraphRagEngine(
        config=config(tmp_path),
        llm=llm,
        graph=graph,
        index=index,
        vectors=vectors,
        communities=communities,
    )


def test_local_question_returns_cited_evidence(tmp_path):
    result = fixture_engine(tmp_path).answer("What treats asthma?", k=4)

    assert result.mode == "local"
    assert not result.abstained
    assert "asthma" in result.answer.lower()
    assert result.citations
    assert all(citation in {"asthma::1", "asthma::2", "diabetes::1"} for citation in result.citations)
    fact_text = " ".join(item.get("text", "") for item in result.evidence if item["kind"] == "fact")
    assert "Inhaled corticosteroids" in fact_text
    assert result.verification["ok"] is True


def test_no_seed_falls_back_to_fetch_only(tmp_path):
    result = fixture_engine(tmp_path).answer("airway inflammation", k=3)

    assert result.mode == "local"
    assert result.fallback is True
    assert result.route == "local_vector"
    assert result.citations
    assert "asthma" in result.answer.lower()


def test_unanswerable_question_abstains(tmp_path):
    result = fixture_engine(tmp_path).answer("Who won the 1930 World Cup?", k=4)

    assert result.abstained is True
    assert result.mode == "abstain"
    assert result.citations == []
    assert "does not contain" in result.answer


def test_global_question_reads_community_summaries(tmp_path):
    result = fixture_engine(tmp_path).answer("What are the main themes of the corpus?", k=3)

    assert result.mode == "global"
    assert result.citations
    assert result.citations[0].startswith("C")
    assert any(item["kind"] == "community" for item in result.evidence)
    assert result.verification["ok"] is True


def test_router_failure_defaults_to_local_and_narration_failure_falls_back(tmp_path):
    class BrokenLLM:
        stats = None

        def complete_json(self, **kwargs):
            raise GraphRagLLMError("router down")

        def complete_text(self, **kwargs):
            raise GraphRagLLMError("narrator down")

    engine = fixture_engine(tmp_path, llm=BrokenLLM())
    assert engine.route("What treats asthma?") == "local"
    result = engine.answer("What treats asthma?", k=4)
    assert result.mode == "local"
    assert "narrator down" in (result.error or "")


def test_verification_flags_unknown_citation():
    verification = verify_citations("Asthma is chronic [asthma::1]. It is caused by ghosts [fake::9].", {"asthma::1"})

    assert verification["ok"] is False
    assert verification["unsupported"] == ["fake::9"]


def test_llm_reranker_can_only_reorder_the_shortlist():
    class RerankLLM:
        def complete_json(self, **kwargs):
            return {"keep": [2, 1, 99, "x"]}

    candidates = [
        {"id": "a", "text": "A", "base_score": 3.0},
        {"id": "b", "text": "B", "base_score": 2.0},
        {"id": "c", "text": "C", "base_score": 1.0},
    ]
    ranked = rerank("q", candidates, 2, llm=RerankLLM(), allow_llm=True)

    assert [item["id"] for item in ranked] == ["b", "a"]


def test_bad_reranker_output_falls_back_to_score_order():
    class BadRerankLLM:
        def complete_json(self, **kwargs):
            raise GraphRagLLMError("bad reranker")

    candidates = [
        {"id": "a", "text": "A", "base_score": 3.0},
        {"id": "b", "text": "B", "base_score": 2.0},
        {"id": "c", "text": "C", "base_score": 1.0},
    ]
    ranked = rerank("q", candidates, 2, llm=BadRerankLLM(), allow_llm=True)

    assert [item["id"] for item in ranked] == ["a", "b"]
