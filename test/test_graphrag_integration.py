"""G3: the harness asks GraphRAG through the real tool contract."""

from __future__ import annotations

from pathlib import Path

from harness.config import Config
from harness.llm import LLMResponse
from harness.loop import run_turn
from harness.messages import Message, ToolCall
from harness.tools import SearchMedicalLiteratureTool, ToolRegistry
from graphrag.backend import GraphRagBackend
from graphrag.chunk import Chunk, ChunkIndex, write_index
from graphrag.communities import build_communities, save_communities
from graphrag.config import GraphRagConfig
from graphrag.embed import build_vectors
from graphrag.graph import Graph, GraphEdge, GraphNode, save_graph


def fixture_config(tmp_path: Path) -> GraphRagConfig:
    return GraphRagConfig(
        corpus_dir=tmp_path / "corpus",
        artifacts_dir=tmp_path / "artifacts",
        llm=Config(base_url="https://stub.test/v1", api_key="k", model="stub"),
        embed_dim=96,
        community_min_size=2,
        seed_threshold=0.65,
        chunk_threshold=0.30,
        llm_rerank=False,
    )


def fixture_pipeline(tmp_path: Path) -> GraphRagConfig:
    config = fixture_config(tmp_path)
    config.ensure_dirs()
    graph = Graph(
        nodes={
            "asthma": GraphNode("asthma", "Asthma", "DISEASE", 5),
            "inhaled corticosteroids": GraphNode(
                "inhaled corticosteroids", "Inhaled corticosteroids", "DRUG", 2
            ),
            "wheezing": GraphNode("wheezing", "Wheezing", "SYMPTOM", 1),
        },
        edges=[
            GraphEdge(
                "inhaled corticosteroids",
                "TREATS",
                "asthma",
                2,
                ["asthma::1"],
            ),
            GraphEdge("asthma", "CAUSES", "wheezing", 1, ["asthma::1"]),
        ],
        chunk_count=1,
        fact_count=2,
    )
    index = ChunkIndex(
        articles=[{"id": "asthma", "title": "Asthma", "chunks": []}],
        chunks=[
            Chunk(
                "asthma::1",
                "asthma",
                "Asthma",
                1,
                "Inhaled corticosteroids treat asthma. Asthma causes wheezing.",
            )
        ],
        target=400,
        overlap=60,
    )
    save_graph(graph, config.graph_path)
    write_index(index, config.index_path)
    save_communities(
        build_communities(graph, None, min_size=2).payload,
        config.communities_path,
    )
    build_vectors(config, graph, index, None)
    return config


def test_tool_answers_from_real_graphrag_artifacts(tmp_path):
    config = fixture_pipeline(tmp_path)
    backend = GraphRagBackend(config=config, llm=None)
    tool = SearchMedicalLiteratureTool(backend)

    result = tool.run({"question": "What treats asthma?", "mode": "local", "k": 3})

    assert result.ok
    assert "Inhaled corticosteroids" in result.content
    assert "asthma::1" in result.content
    assert result.data["citations"] == ["asthma::1"]
    assert result.data["evidence_ids"] == ["asthma::1"]
    assert result.data["mode"] == "local"


def test_missing_index_is_a_tool_error_not_a_traceback(tmp_path):
    backend = GraphRagBackend(config=fixture_config(tmp_path), llm=None)
    result = SearchMedicalLiteratureTool(backend).run({"question": "What treats asthma?"})

    assert not result.ok
    assert "GraphRAG index is not built" in result.content


class StubClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="search_medical_literature",
                        arguments={"question": "What treats asthma?", "mode": "local"},
                        raw_arguments='{"question": "What treats asthma?", "mode": "local"}',
                    )
                ],
            )
        return LLMResponse(content="Inhaled corticosteroids treat asthma [asthma::1].")


def test_question_tool_call_then_cited_answer_end_to_end(tmp_path):
    config = fixture_pipeline(tmp_path)
    backend = GraphRagBackend(config=config, llm=None)
    tools = ToolRegistry([SearchMedicalLiteratureTool(backend)])
    conversation = [
        Message(role="system", content="Answer from the medical literature."),
        Message(role="user", content="What treats asthma?"),
    ]

    result = run_turn(conversation, client=StubClient(), tools=tools)

    assert result.answered
    tool_message = next(message for message in result.messages if message.role == "tool")
    assert "Inhaled corticosteroids" in tool_message.content
    assert "asthma::1" in tool_message.content
    assert result.final == "Inhaled corticosteroids treat asthma [asthma::1]."
