"""I10 communities: weighted label propagation + one summary per cluster.

The graph collapses to undirected links weighted ``1 + log(total mentions)`` so a
hub is loud but not deafening. Labels propagate in a fixed order, then one small
LLM call writes a summary for each community large enough to matter. A
deterministic summary is used when the model fails, so a run never ends in
silence.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .graph import Graph, GraphEdge
from .llm import GraphLLM, GraphRagLLMError

COMMUNITIES_VERSION = 1
MAX_PASSES = 25

SUMMARY_PROMPT = """You summarize one community of a medical knowledge graph.
Use only the entities and relations listed. Do not invent facts, do not give
medical advice, and do not mention the graph mechanics.
Write one plain paragraph of at most 180 words that says what this community is
about: its main entities, the strongest relations, and any border links to other
communities. No preamble, no headings, no citations."""


@dataclass(slots=True)
class Community:
    id: str
    label: str
    members: list[str]
    summary: str
    method: str = "llm"
    top_edges: list[str] = field(default_factory=list)

    def to_dict(self, graph: Graph) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "size": len(self.members),
            "members": self.members,
            "member_names": [
                graph.nodes[key].display if key in graph.nodes else key for key in self.members
            ],
            "top_edges": self.top_edges,
            "summary": self.summary,
            "method": self.method,
        }


@dataclass(slots=True)
class CommunityBuild:
    communities: list[Community]
    payload: dict[str, Any]


def build_communities(
    graph: Graph,
    llm: GraphLLM | None,
    *,
    min_size: int = 3,
) -> CommunityBuild:
    weights = _undirected_weights(graph)
    labels = label_propagation(graph, weights)
    grouped: dict[str, list[str]] = defaultdict(list)
    for key in sorted(labels):
        grouped[labels[key]].append(key)
    groups = [
        (label, sorted(members))
        for label, members in grouped.items()
        if len(members) >= min_size
    ]
    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    communities: list[Community] = []
    for number, (label, members) in enumerate(groups):
        edges = _cross_edges(graph, members)
        if llm is None:
            summary, method, error = _fallback_summary(graph, members, edges), "fallback", "no model"
        else:
            try:
                summary = _summarize(graph, members, edges, llm)
                method, error = "llm", None
            except (GraphRagLLMError, ValueError) as exc:
                summary, method, error = _fallback_summary(graph, members, edges), "fallback", str(exc)
        community = Community(
            id=f"C{number}",
            label=label,
            members=members,
            summary=summary,
            method=method,
            top_edges=[edge for edge, _ in edges[:8]],
        )
        communities.append(community)

    unassigned = sorted(
        key for key in graph.nodes if key not in {member for group in groups for member in group[1]}
    )
    payload = {
        "version": COMMUNITIES_VERSION,
        "generated_at": _now(),
        "min_size": min_size,
        "graph_stats": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "count": len(communities),
        "communities": [community.to_dict(graph) for community in communities],
        "unassigned": unassigned,
    }
    return CommunityBuild(communities=communities, payload=payload)


def label_propagation(graph: Graph, weights: dict[tuple[str, str], float]) -> dict[str, str]:
    """Deterministic asynchronous label propagation with a fixed visit order."""
    adjacency: dict[str, list[tuple[str, float]]] = {key: [] for key in graph.nodes}
    for (left, right), weight in weights.items():
        adjacency[left].append((right, weight))
        adjacency[right].append((left, weight))
    for key in adjacency:
        adjacency[key].sort()

    labels = {key: key for key in sorted(graph.nodes)}
    keys = sorted(graph.nodes)
    for _ in range(MAX_PASSES):
        changes = 0
        for key in keys:
            votes: Counter[str] = Counter()
            for neighbour, weight in adjacency[key]:
                votes[labels[neighbour]] += weight
            if not votes:
                continue
            winner = sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0]
            if winner != labels[key]:
                labels[key] = winner
                changes += 1
        if changes == 0:
            break
    # Normalize chains so every member points at one terminal label.
    for key in keys:
        seen = {key}
        label = labels[key]
        while labels.get(label, label) != label and label not in seen:
            seen.add(label)
            label = labels[label]
        labels[key] = label
    return labels


def _undirected_weights(graph: Graph) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for edge in graph.edges:
        if edge.head == edge.tail:
            continue
        pair = tuple(sorted((edge.head, edge.tail)))
        totals[pair] += edge.mentions
    return {pair: 1.0 + math.log(total) for pair, total in totals.items() if total > 0}


def _cross_edges(graph: Graph, members: list[str]) -> list[tuple[str, GraphEdge]]:
    inside = set(members)
    selected: list[tuple[str, GraphEdge]] = []
    for edge in graph.edges:
        if edge.head in inside and edge.tail in inside:
            selected.append((f"{edge.head} --{edge.relation}--> {edge.tail} (x{edge.mentions})", edge))
        elif edge.head in inside or edge.tail in inside:
            selected.append(
                (f"{edge.head} --{edge.relation}--> {edge.tail} (x{edge.mentions}) [border]", edge)
            )
    selected.sort(key=lambda item: (-item[1].mentions, item[0]))
    return selected


def _summarize(
    graph: Graph,
    members: list[str],
    edges: list[tuple[str, GraphEdge]],
    llm: GraphLLM,
) -> str:
    lines = ["Members:"]
    for key in sorted(members, key=lambda key: (-graph.nodes[key].mentions, key))[:40]:
        node = graph.nodes[key]
        lines.append(f"- {node.display} ({node.type}, {node.mentions} mentions)")
    lines.append("")
    lines.append("Relations:")
    for rendered, _ in edges[:60]:
        lines.append(f"- {rendered}")
    text = llm.complete_text(
        system=SUMMARY_PROMPT,
        user="\n".join(lines) + "\n\nWrite the community summary now.",
        max_tokens=1_800,
    )
    return text.strip()


def _fallback_summary(graph: Graph, members: list[str], edges: list[tuple[str, GraphEdge]]) -> str:
    top = sorted(members, key=lambda key: (-graph.nodes[key].mentions, key))[:8]
    names = ", ".join(graph.nodes[key].display for key in top)
    relations = "; ".join(rendered for rendered, _ in edges[:5])
    return (
        f"Community of {len(members)} entities. Main entities: {names}. "
        f"Key relations: {relations or 'none recorded'}."
    )


def save_communities(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_communities(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"communities not found: {target}; run the communities stage first")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != COMMUNITIES_VERSION:
        raise ValueError(f"unknown communities version {payload.get('version')!r}")
    return payload


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
