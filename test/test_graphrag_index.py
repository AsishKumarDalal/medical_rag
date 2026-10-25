"""G1: the indexing pipeline, offline with a stub model and a mini corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.config import Config
from graphrag.chunk import Chunk, ChunkIndex, build_chunk_index, load_index, write_index
from graphrag.communities import build_communities, load_communities
from graphrag.config import GraphRagConfig
from graphrag.corpus import load_corpus
from graphrag.embed import HashEmbedder, build_vectors, cosine, load_vectors, top_matches
from graphrag.extract import Checkpoint, extract_chunks, load_records, normalize_payload
from graphrag.graph import Graph, GraphEdge, GraphNode, build_graph, load_graph
from graphrag.llm import GraphRagLLMError
from graphrag.resolve import (
    canonical_mapping,
    jaro_winkler,
    nominate,
    rebuild_graph,
    resolve_entities,
    token_overlap,
)
from graphrag.validate import validate_record

FIXTURE_CORPUS = Path(__file__).parent / "data" / "fixtures" / "graph_corpus"
PROMPT_CHUNK = "Chunk id: "


def llm_config() -> Config:
    return Config(base_url="https://stub.test/v1", api_key="k", model="stub")


def graph_config(tmp_path: Path, **overrides) -> GraphRagConfig:
    defaults = dict(
        corpus_dir=FIXTURE_CORPUS,
        artifacts_dir=tmp_path / "artifacts",
        llm=llm_config(),
        chunk_target=400,
        chunk_overlap=60,
        embed_dim=64,
        judge_max_batches=10,
        community_min_size=2,
    )
    defaults.update(overrides)
    return GraphRagConfig(**defaults)


def extraction_payload(chunk_id: str) -> dict:
    if chunk_id.startswith("diabetes"):
        return {
            "entities": [
                {"name": "Diabetes mellitus", "type": "DISEASE"},
                {"name": "Insulin", "type": "DRUG"},
                {"name": "Metformin", "type": "DRUG"},
            ],
            "triples": [
                {"head": "Insulin", "relation": "TREATS", "tail": "Diabetes mellitus"},
                {"head": "Metformin", "relation": "TREATS", "tail": "Diabetes mellitus"},
            ],
        }
    return {
        "entities": [
            {"name": "Asthma", "type": "DISEASE"},
            {"name": "Inhaled corticosteroids", "type": "DRUG"},
            {"name": "Wheezing", "type": "SYMPTOM"},
        ],
        "triples": [
            {"head": "Inhaled corticosteroids", "relation": "TREATS", "tail": "Asthma"},
            {"head": "Asthma", "relation": "CAUSES", "tail": "Wheezing"},
        ],
    }


class FakeLLM:
    def __init__(self, *, fail_on: set[str] | None = None, judge: list[list[str]] | None = None):
        self.fail_on = fail_on or set()
        self.judge = judge or []
        self.extract_calls: list[str] = []
        self.text_calls = 0

    def complete_json(self, *, system, user, max_tokens=None, attempts=2, expect=(dict, list), retry_note=None):
        if user.startswith("Candidate entities:"):
            return {"groups": self.judge}
        chunk_id = user.split(PROMPT_CHUNK, 1)[1].split("\n", 1)[0].strip()
        self.extract_calls.append(chunk_id)
        if chunk_id in self.fail_on:
            raise GraphRagLLMError("stub extraction failure")
        return extraction_payload(chunk_id)

    def complete_text(self, *, system, user, max_tokens=None, temperature=None):
        self.text_calls += 1
        return "A stub summary of this community."

    def embed(self, texts, *, batch_size=64):
        return HashEmbedder(dim=64).embed(texts)


def test_chunking_is_deterministic_and_keeps_citable_ids(tmp_path):
    articles = load_corpus(FIXTURE_CORPUS)

    first = build_chunk_index(articles, target=400, overlap=60)
    second = build_chunk_index(articles, target=400, overlap=60)

    assert [chunk.id for chunk in first.chunks] == [chunk.id for chunk in second.chunks]
    assert [chunk.text for chunk in first.chunks] == [chunk.text for chunk in second.chunks]
    assert len(first.chunks) > len(articles)
    assert all(chunk.id.split("::")[1].isdigit() for chunk in first.chunks)
    assert all(chunk.chars <= 400 + 60 + 2 for chunk in first.chunks)
    assert first.chunks[0].id.endswith("::1")

    path = tmp_path / "index.json"
    write_index(first, path)
    reloaded = load_index(path)
    assert [chunk.id for chunk in reloaded.chunks] == [chunk.id for chunk in first.chunks]


def test_write_index_serializes_directly_constructed_chunks(tmp_path):
    index = ChunkIndex(
        articles=[{"id": "asthma", "title": "Asthma", "chunks": []}],
        chunks=[Chunk("asthma::1", "asthma", "Asthma", 1, "Asthma text.")],
        target=400,
        overlap=60,
    )

    path = tmp_path / "index.json"
    write_index(index, path)
    reloaded = load_index(path)

    assert [chunk.id for chunk in reloaded.chunks] == ["asthma::1"]
    assert reloaded.chunks[0].text == "Asthma text."


def test_checkpoint_resumes_and_retries_failed_chunks(tmp_path):
    config = graph_config(tmp_path)
    index = build_chunk_index(load_corpus(FIXTURE_CORPUS), target=400, overlap=60)
    first_id = index.chunks[0].id
    fake = FakeLLM(fail_on={first_id})

    first = extract_chunks(index, fake, config, limit=2)
    assert first.written == 1
    assert first.failed == 1
    failed = Checkpoint(config.checkpoint_path, config.failed_path).failed()
    assert first_id in failed

    # A plain resume on the same two chunks skips both the done and failed one.
    two = ChunkIndex(articles=index.articles, chunks=index.chunks[:2], target=index.target, overlap=index.overlap)
    second = extract_chunks(two, fake, config)
    assert second.written == 0
    assert first_id not in fake.extract_calls[-1:]

    fake.fail_on.clear()
    third = extract_chunks(two, fake, config, retry_failed=True)
    assert third.failed == 0
    assert first_id in third_record_ids(config)


def third_record_ids(config: GraphRagConfig) -> set[str]:
    return {record["chunk_id"] for record in load_records(config.checkpoint_path)}


def test_extraction_salvages_alternate_keys_and_caps_triples():
    entities, triples = normalize_payload(
        {
            "entities": [{"name": "A", "type": "disease"}],
            "facts": [
                {"subject": "A", "predicate": "causes", "object": "B"},
                {"head": "A", "relation": "causes", "tail": "C"},
            ],
        },
        max_triples=1,
    )

    assert entities == [{"name": "A", "type": "DISEASE"}]
    assert triples == [{"head": "A", "relation": "causes", "tail": "B"}]


def test_validation_gate_rejects_vague_self_loop_and_junk_and_splits_compounds():
    result = validate_record(
        {
            "chunk_id": "asthma::1",
            "entities": [{"name": "Asthma", "type": "DISEASE"}],
            "triples": [
                {"head": "Asthma", "relation": "CAUSES", "tail": "Asthma"},
                {"head": "other diseases", "relation": "CAUSES", "tail": "Asthma"},
                {"head": "Asthma", "relation": "RELATED_TO", "tail": "Allergy"},
                {"head": "Aspirin", "relation": "TREATS", "tail": "pain and fever"},
                {
                    "head": "a very long rambling entity name with far too many tokens",
                    "relation": "TREATS",
                    "tail": "Pain",
                },
                {"head": "Smoking", "relation": "RISK_FACTOR_FOR", "tail": "Asthma"},
            ],
        }
    )

    accepted = {(fact.head, fact.relation, fact.tail) for fact in result.triples}
    assert ("Aspirin", "TREATS", "pain") in accepted
    assert ("Aspirin", "TREATS", "fever") in accepted
    assert ("Smoking", "RISK_FACTOR_FOR", "Asthma") in accepted
    assert all("RELATED_TO" not in fact.relation for fact in result.triples)
    assert any("junk relation" in rejection["reason"] for rejection in result.rejected)
    assert any("self-loop" in rejection["reason"] for rejection in result.rejected)
    assert any("vague group" in rejection["reason"] for rejection in result.rejected)


def test_graph_aggregates_mentions_and_backfills_types():
    records = [
        {
            "chunk_id": "asthma::1",
            "entities": [{"name": "Aspirin", "type": "MISC"}, {"name": "Headache", "type": "SYMPTOM"}],
            "triples": [{"head": "Aspirin", "relation": "TREATS", "tail": "Headache"}],
        },
        {
            "chunk_id": "asthma::2",
            "entities": [{"name": "Aspirin", "type": "MISC"}],
            "triples": [
                {"head": "Aspirin", "relation": "TREATS", "tail": "Headache"},
                {"head": "Aspirin", "relation": "TREATS", "tail": "Fever"},
            ],
        },
    ]

    build = build_graph(records)
    edge = build.graph.edge("aspirin|TREATS|headache")

    assert edge is not None
    assert edge.mentions == 2
    assert edge.source_chunks == ["asthma::1", "asthma::2"]
    assert build.graph.nodes["aspirin"].mentions == 3
    assert build.graph.nodes["aspirin"].type == "DRUG"  # backfilled from repeated TREATS
    assert len(build.graph.edges) == 2


def test_resolution_merges_only_judged_groups_and_audits_skips():
    graph = Graph(
        nodes={
            "tesla": GraphNode("tesla", "Tesla, Inc.", "ORGANIZATION", 10),
            "tesla motors": GraphNode("tesla motors", "Tesla Motors", "ORGANIZATION", 3),
            "tesla energy": GraphNode("tesla energy", "Tesla Energy", "ORGANIZATION", 2),
            "tesla powerwall": GraphNode("tesla powerwall", "Tesla Powerwall", "PRODUCT", 2),
        },
        edges=[
            GraphEdge("tesla", "ACQUIRED", "tesla energy", 2, ["x::1"]),
            GraphEdge("tesla", "MANUFACTURES", "tesla powerwall", 2, ["x::2"]),
            GraphEdge("tesla", "CEO_OF", "tesla motors", 2, ["x::3"]),
        ],
        chunk_count=3,
        fact_count=6,
    )
    fake = FakeLLM(judge=[["tesla", "tesla motors"]])
    result = resolve_entities(graph, fake, max_batches=10)

    assert result.mapping["tesla motors"] == "tesla"
    assert result.mapping["tesla energy"] == "tesla energy"
    assert "tesla energy" in result.graph.nodes
    assert "tesla powerwall" in result.graph.nodes
    assert result.graph.nodes["tesla"].mentions == 13
    assert "Tesla Motors" in result.graph.nodes["tesla"].aliases
    assert any(skip["reason"] == "maker/made edge" for skip in result.log["skipped"])
    assert any(verdict["verdict"] == "merge" for verdict in result.log["verdicts"])


def test_resolution_does_not_merge_when_judge_fails():
    graph = Graph(
        nodes={
            "tesla": GraphNode("tesla", "Tesla", "ORGANIZATION", 4),
            "tesla motors": GraphNode("tesla motors", "Tesla Motors", "ORGANIZATION", 2),
        },
        edges=[GraphEdge("tesla", "ACQUIRED", "tesla motors", 1, ["x::1"])],
        chunk_count=1,
        fact_count=1,
    )

    class BrokenJudge(FakeLLM):
        def complete_json(self, **kwargs):
            raise GraphRagLLMError("judge down")

    result = resolve_entities(graph, BrokenJudge(), max_batches=5)

    assert result.merged == 0
    assert result.graph.nodes.keys() == graph.nodes.keys()
    assert result.log["batches"][0]["error"]


def test_fuzzy_helpers_nominate_similar_names():
    assert jaro_winkler("tesla", "tesla") == 1.0
    assert jaro_winkler("tesla", "tesla motors") > 0.7
    assert token_overlap("tesla motors", "tesla energy") == pytest.approx(1 / 3)
    graph = Graph(
        nodes={
            "tesla": GraphNode("tesla", "Tesla", "ORGANIZATION", 5),
            "tesla motors": GraphNode("tesla motors", "Tesla Motors", "ORGANIZATION", 2),
            "asthma": GraphNode("asthma", "Asthma", "DISEASE", 5),
        }
    )
    candidates, _ = nominate(graph)
    assert [(candidate.a, candidate.b) for candidate in candidates] == [("tesla", "tesla motors")]
    assert canonical_mapping(graph, [["tesla", "tesla motors"]])["tesla motors"] == "tesla"
    assert rebuild_graph(graph, {"tesla motors": "tesla"}).nodes["tesla"].mentions == 7


def test_communities_get_summaries_and_survive_json(tmp_path):
    graph = Graph(
        nodes={
            "a": GraphNode("a", "Alpha", "DISEASE", 4),
            "b": GraphNode("b", "Beta", "DRUG", 4),
            "c": GraphNode("c", "Gamma", "SYMPTOM", 2),
            "d": GraphNode("d", "Delta", "MISC", 1),
        },
        edges=[
            GraphEdge("a", "TREATS", "b", 3, ["x::1"]),
            GraphEdge("b", "TREATS", "a", 2, ["x::2"]),
            GraphEdge("a", "CAUSES", "c", 1, ["x::3"]),
        ],
        chunk_count=3,
        fact_count=6,
    )
    build = build_communities(graph, FakeLLM(), min_size=2)

    assert build.communities
    assert all(community.summary for community in build.communities)
    assert build.payload["count"] == len(build.communities)
    path = tmp_path / "communities.json"
    path.write_text(json.dumps(build.payload), encoding="utf-8")
    assert load_communities(path)["count"] == len(build.communities)


def test_hash_embeddings_are_deterministic_and_lexically_similar():
    embedder = HashEmbedder(dim=128)
    vectors = embedder.embed(["asthma is a disease", "diabetes is a disease", "asthma"])

    assert len(vectors[0]) == 128
    assert cosine(vectors[0], embedder.embed(["asthma is a disease"])[0]) == pytest.approx(1.0)
    assert cosine(vectors[0], vectors[2]) > cosine(vectors[0], embedder.embed(["diabetes"])[0])
    matches = top_matches(vectors[0], {"a": vectors[0], "b": vectors[1]}, k=1)
    assert matches[0][0] == "a"


def test_full_offline_index_pipeline(tmp_path):
    config = graph_config(tmp_path)
    fake = FakeLLM(judge=[])
    index = build_chunk_index(load_corpus(config.corpus_dir), target=400, overlap=60)
    config.ensure_dirs()
    write_index(index, config.index_path)

    stats = extract_chunks(index, fake, config, limit=6)
    assert stats.written == 6
    records = load_records(config.checkpoint_path)
    assert len(records) == 6

    build = build_graph(records)
    assert build.graph.nodes
    from graphrag.graph import save_graph

    save_graph(build.graph, config.graph_path)

    resolution = resolve_entities(build.graph, fake, max_batches=5)
    save_graph(resolution.graph, config.graph_path)

    communities = build_communities(resolution.graph, fake, min_size=2)
    from graphrag.communities import save_communities

    save_communities(communities.payload, config.communities_path)

    store = build_vectors(config, resolution.graph, index, None)
    assert load_vectors(config.vectors_path).dim == 64
    assert len(store.nodes) == len(resolution.graph.nodes)
    assert len(store.chunks) == len(index.chunks)
    assert config.graph_path.is_file()
    assert config.vectors_path.is_file()
