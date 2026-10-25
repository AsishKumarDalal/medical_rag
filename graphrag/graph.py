"""I7 aggregation and I8 type backfill.

One fact stated many times becomes one edge with a mention count and the union of
its source chunks. Nodes accumulate mentions and aliases; nodes still typed MISC
get a type by voting over *repeated* edges only (one-off noisy edges do not vote).
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .normalize import clean_type, normalize_entity
from .validate import ValidationResult, ValidatedFact, validate_record

GRAPH_VERSION = 1
# relation -> (type of head, type of tail). Only used to type MISC nodes.
RELATION_TYPES: dict[str, tuple[str | None, str | None]] = {
    "CEO_OF": ("PERSON", "ORGANIZATION"),
    "FOUNDED_BY": ("ORGANIZATION", "PERSON"),
    "FOUNDED": ("PERSON", "ORGANIZATION"),
    "LOCATED_IN": ("LOCATION", "LOCATION"),
    "ACQUIRED": ("ORGANIZATION", "ORGANIZATION"),
    "ACQUIRED_BY": ("ORGANIZATION", "ORGANIZATION"),
    "SUBSIDIARY_OF": ("ORGANIZATION", "ORGANIZATION"),
    "PARENT_OF": ("ORGANIZATION", "ORGANIZATION"),
    "COMPETES_WITH": ("ORGANIZATION", "ORGANIZATION"),
    "PARTNER_WITH": ("ORGANIZATION", "ORGANIZATION"),
    "MANUFACTURES": ("ORGANIZATION", "PRODUCT"),
    "PRODUCES": ("ORGANIZATION", "PRODUCT"),
    "DEVELOPS": ("ORGANIZATION", "PRODUCT"),
    "TREATS": ("DRUG", "DISEASE"),
    "PREVENTS": ("DRUG", "DISEASE"),
    "USED_TO_TREAT": ("DRUG", "DISEASE"),
    "CAUSES": ("ORGANISM", "DISEASE"),
    "CAUSED_BY": ("DISEASE", "ORGANISM"),
    "SYMPTOM_OF": ("SYMPTOM", "DISEASE"),
    "DIAGNOSED_BY": ("DISEASE", "PROCEDURE"),
    "RISK_FACTOR_FOR": ("RISK_FACTOR", "DISEASE"),
    "AFFECTS": ("DISEASE", "ANATOMY"),
    "OCCURS_IN": ("DISEASE", "ANATOMY"),
    "FOUND_IN": ("ORGANISM", "LOCATION"),
    "TRANSMITTED_BY": ("DISEASE", "ORGANISM"),
    "APPROVED_BY": ("DRUG", "ORGANIZATION"),
}
# These labels are directionally unreliable in extraction, so their tails do not
# vote on a node's type. This is the I8 lesson from the upstream spec.
TAIL_VOTE_EXCLUDED = {"FOUNDED_BY", "ACQUIRED_BY", "CAUSED_BY", "MANUFACTURED_BY"}


@dataclass(slots=True)
class GraphNode:
    key: str
    display: str
    type: str = "MISC"
    mentions: int = 0
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display": self.display,
            "type": self.type,
            "mentions": self.mentions,
            "aliases": self.aliases,
        }


@dataclass(slots=True)
class GraphEdge:
    head: str
    relation: str
    tail: str
    mentions: int = 0
    source_chunks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "mentions": self.mentions,
            "source_chunks": self.source_chunks,
        }

    @property
    def id(self) -> str:
        return f"{self.head}|{self.relation}|{self.tail}"

    def render(self, graph: "Graph") -> str:
        head = graph.node(self.head).display if graph.node(self.head) else self.head
        tail = graph.node(self.tail).display if graph.node(self.tail) else self.tail
        return f"{head} --{self.relation}--> {tail}"


@dataclass(slots=True)
class Graph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    chunk_count: int = 0
    fact_count: int = 0

    def node(self, key: str) -> GraphNode | None:
        return self.nodes.get(key)

    def edge(self, edge_id: str) -> GraphEdge | None:
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        return None

    def adjacency(self) -> dict[str, list[tuple[str, GraphEdge]]]:
        """Undirected adjacency: node -> [(neighbour, edge)]."""
        result: dict[str, list[tuple[str, GraphEdge]]] = {key: [] for key in self.nodes}
        for edge in self.edges:
            if edge.head in result:
                result[edge.head].append((edge.tail, edge))
            if edge.tail in result:
                result[edge.tail].append((edge.head, edge))
        return result

    def touching(self, key: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.head == key or edge.tail == key]

    def to_dict(self) -> dict[str, Any]:
        nodes = [self.nodes[key].to_dict() for key in sorted(self.nodes)]
        edges = [edge.to_dict() for edge in sorted(self.edges, key=lambda e: (e.head, e.relation, e.tail))]
        return {
            "version": GRAPH_VERSION,
            "stats": {
                "nodes": len(nodes),
                "edges": len(edges),
                "chunks": self.chunk_count,
                "facts": self.fact_count,
                "mentions": sum(edge.mentions for edge in self.edges),
            },
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Graph":
        if int(payload.get("version", 0)) != GRAPH_VERSION:
            raise ValueError(f"unknown graph version {payload.get('version')!r}")
        nodes = {
            str(record["key"]): GraphNode(
                key=str(record["key"]),
                display=str(record.get("display") or record["key"]),
                type=clean_type(str(record.get("type") or "MISC")),
                mentions=int(record.get("mentions", 0)),
                aliases=[str(alias) for alias in record.get("aliases", [])],
            )
            for record in payload.get("nodes", [])
        }
        edges = [
            GraphEdge(
                head=str(record["head"]),
                relation=str(record["relation"]),
                tail=str(record["tail"]),
                mentions=int(record.get("mentions", 0)),
                source_chunks=[str(chunk) for chunk in record.get("source_chunks", [])],
            )
            for record in payload.get("edges", [])
        ]
        stats = payload.get("stats", {})
        return cls(
            nodes=nodes,
            edges=edges,
            chunk_count=int(stats.get("chunks", 0)),
            fact_count=int(stats.get("facts", 0)),
        )


@dataclass(slots=True)
class GraphBuild:
    graph: Graph
    validation: ValidationResult

    @property
    def stats(self) -> dict[str, int]:
        return {
            **self.validation.stats,
            "nodes": len(self.graph.nodes),
            "edges": len(self.graph.edges),
            "facts": self.graph.fact_count,
        }


def build_graph(records: Iterable[dict[str, Any]], *, max_triples: int = 20) -> GraphBuild:
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str], GraphEdge] = {}
    alias_surfaces: dict[str, Counter[str]] = {}
    type_votes: dict[str, Counter[str]] = {}
    validation = ValidationResult()
    chunks_seen = 0

    for record in records:
        chunks_seen += 1
        one = validate_record(record, max_triples=max_triples)
        validation.rejected.extend(one.rejected)
        chunk_id = str(record["chunk_id"])
        for entity in one.entities:
            key = normalize_entity(entity["name"])
            type_votes.setdefault(key, Counter())[clean_type(entity["type"])] += 1
        for fact in one.triples:
            validation.triples.append(fact)
            _accumulate_fact(fact, nodes, edges, alias_surfaces, type_votes)

    for key, node in nodes.items():
        votes = type_votes.get(key) or Counter()
        node.type = _majority(votes, default="MISC")
        surfaces = alias_surfaces.get(key) or Counter()
        if surfaces:
            node.display = _best_surface(surfaces, fallback=node.display)
            node.aliases = sorted({surface for surface in surfaces if surface != node.display})

    graph = Graph(nodes=nodes, edges=list(edges.values()), chunk_count=chunks_seen, fact_count=len(validation.triples))
    backfill_types(graph)
    return GraphBuild(graph=graph, validation=validation)


def _accumulate_fact(
    fact: ValidatedFact,
    nodes: dict[str, GraphNode],
    edges: dict[tuple[str, str, str], GraphEdge],
    alias_surfaces: dict[str, Counter[str]],
    type_votes: dict[str, Counter[str]],
) -> None:
    head = normalize_entity(fact.head)
    tail = normalize_entity(fact.tail)
    if not head or not tail or head == tail:
        return
    for key, surface in ((head, fact.head), (tail, fact.tail)):
        node = nodes.get(key)
        if node is None:
            node = GraphNode(key=key, display=surface)
            nodes[key] = node
        node.mentions += 1
        alias_surfaces.setdefault(key, Counter())[surface] += 1
        type_votes.setdefault(key, Counter())
    edge = edges.get((head, fact.relation, tail))
    if edge is None:
        edge = GraphEdge(head=head, relation=fact.relation, tail=tail)
        edges[(head, fact.relation, tail)] = edge
    edge.mentions += 1
    if fact.source_chunk not in edge.source_chunks:
        edge.source_chunks.append(fact.source_chunk)


def backfill_types(graph: Graph) -> None:
    """Give MISC nodes a type by voting over repeated edges."""
    votes: dict[str, Counter[str]] = {key: Counter() for key in graph.nodes}
    for edge in graph.edges:
        if edge.mentions < 2:
            continue
        head_type, tail_type = RELATION_TYPES.get(edge.relation, (None, None))
        if head_type:
            votes[edge.head][head_type] += edge.mentions
        if tail_type and edge.relation not in TAIL_VOTE_EXCLUDED:
            votes[edge.tail][tail_type] += edge.mentions
    for key, node in graph.nodes.items():
        if node.type != "MISC":
            continue
        inferred = _majority(votes.get(key) or Counter(), default="MISC")
        if inferred != "MISC":
            node.type = inferred
        node.aliases = sorted(set(node.aliases))


def _majority(votes: Counter[str], *, default: str) -> str:
    if not votes:
        return default
    # Counter.most_common is insertion-ordered on ties; sort by count then name.
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0] if ranked[0][1] > 0 else default


def _best_surface(surfaces: Counter[str], *, fallback: str) -> str:
    ranked = sorted(surfaces.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return ranked[0][0] if ranked else fallback


def save_graph(graph: Graph, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_graph(path: str | Path) -> Graph:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"graph not found: {target}; run the graph stage first")
    return Graph.from_dict(json.loads(target.read_text(encoding="utf-8")))


def save_rejections(rejections: Iterable[dict[str, Any]], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for rejection in rejections:
            handle.write(json.dumps(rejection, ensure_ascii=False) + "\n")
    return target
