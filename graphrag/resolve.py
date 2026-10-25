"""I9 entity resolution: cheap nomination, an informed judge, deterministic rebuild.

Blocking and fuzzy scoring *nominate* pairs; they never decide. One LLM call per
small batch sees names, types, mention counts and real edges. Code validates the
answer, picks the canonical key by mention count, rebuilds the graph and writes
every candidate and verdict to ``merge_log.json``.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from .graph import Graph, GraphEdge, GraphNode, backfill_types
from .llm import GraphLLM, GraphRagLLMError

NOMINATION_THRESHOLD = 0.50
BUCKET_CAP = 60
BATCH_SIZE = 8
_MAKER_RELATIONS = {
    "MANUFACTURES",
    "PRODUCES",
    "DEVELOPS",
    "LAUNCHES",
    "LAUNCHED",
    "SELLS",
    "CREATES",
    "INVENTED",
    "MADE_BY",
}
_STOPWORDS = {
    "of",
    "and",
    "the",
    "a",
    "an",
    "in",
    "to",
    "for",
    "with",
    "by",
    "on",
    "at",
    "from",
    "as",
    "is",
    "its",
    "their",
    "this",
    "that",
}

JUDGE_PROMPT = """You are the entity-resolution judge for a medical knowledge graph.
You are shown a batch of entity clusters nominated by fuzzy string similarity.
Decide which of these names refer to the SAME real-world entity in this corpus.

Rules:
- If unsure, do NOT group. A wrong merge is much worse than a missed merge.
- Products, drugs, procedures, divisions, factories and locations are different
  from the organisation that makes, owns or uses them.
- A surname only is not enough; use the types, mention counts and edges.
- Respect the evidence: a maker -> made edge proves two names are different.
- Copy entity keys exactly as given. Do not invent keys.
Return ONLY this JSON object:
{"groups": [["key_a", "key_b"], ...]}
Each group must contain at least two keys and every key at most once. An empty
groups list is a valid and common answer."""


@dataclass(slots=True)
class Candidate:
    a: str
    b: str
    score: float
    jaro_winkler: float
    token_overlap: float
    reason: str = ""


@dataclass(slots=True)
class Resolution:
    graph: Graph
    mapping: dict[str, str] = field(default_factory=dict)
    log: dict[str, Any] = field(default_factory=dict)

    @property
    def merged(self) -> int:
        return sum(1 for key, target in self.mapping.items() if key != target)


def resolve_entities(
    graph: Graph,
    llm: GraphLLM | None,
    *,
    max_batches: int = 40,
    threshold: float = NOMINATION_THRESHOLD,
) -> Resolution:
    candidates, skipped = nominate(graph, threshold=threshold)
    components = _components([candidate for candidate in candidates if candidate.reason == ""])
    batches = _batch_components(components, limit=max_batches)
    judged: list[dict[str, Any]] = []
    approved_groups: list[list[str]] = []
    for number, batch in enumerate(batches, start=1):
        if llm is None:
            judged.append({"batch": number, "keys": batch, "groups": [], "error": "no judge configured"})
            continue
        try:
            groups = judge_batch(graph, batch, llm)
            error = None
        except (GraphRagLLMError, ValueError) as exc:
            groups = []
            error = f"{type(exc).__name__}: {exc}"
        judged.append({"batch": number, "keys": batch, "groups": groups, "error": error})
        approved_groups.extend(groups)
    mapping = canonical_mapping(graph, approved_groups)
    rebuilt = rebuild_graph(graph, mapping)
    verdicts = _verdicts(candidates, approved_groups)
    log = {
        "generated_at": _now(),
        "threshold": threshold,
        "components": len(components),
        "batches": judged,
        "candidates": [asdict(candidate) for candidate in candidates],
        "skipped": skipped,
        "verdicts": verdicts,
        "mapping": {key: target for key, target in sorted(mapping.items()) if key != target},
        "merged_nodes": sum(1 for key, target in mapping.items() if key != target),
        "stats_before": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "stats_after": {"nodes": len(rebuilt.nodes), "edges": len(rebuilt.edges)},
    }
    return Resolution(graph=rebuilt, mapping=mapping, log=log)


def nominate(graph: Graph, *, threshold: float = NOMINATION_THRESHOLD) -> tuple[list[Candidate], list[dict[str, Any]]]:
    keys = sorted(graph.nodes)
    buckets: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        for token in _tokens(key):
            buckets[token].append(key)
    pair_keys: set[tuple[str, str]] = set()
    for token in sorted(buckets):
        members = sorted(buckets[token], key=lambda key: (-graph.nodes[key].mentions, key))[:BUCKET_CAP]
        for left, right in itertools.combinations(sorted(members), 2):
            pair_keys.add((left, right))

    candidates: list[Candidate] = []
    skipped: list[dict[str, Any]] = []
    for left, right in sorted(pair_keys):
        if " and " in left or " and " in right:
            skipped.append({"a": left, "b": right, "reason": "compound name"})
            continue
        if _maker_made(graph, left, right):
            skipped.append({"a": left, "b": right, "reason": "maker/made edge"})
            continue
        jw = jaro_winkler(left, right)
        overlap = token_overlap(left, right)
        score = 0.6 * jw + 0.4 * overlap
        if score < threshold:
            continue
        candidates.append(Candidate(left, right, round(score, 4), round(jw, 4), round(overlap, 4)))
    return candidates, skipped


def judge_batch(graph: Graph, keys: list[str], llm: GraphLLM) -> list[list[str]]:
    payload = llm.complete_json(
        system=JUDGE_PROMPT,
        user=_judge_user(graph, keys),
        max_tokens=1_800,
        attempts=2,
        expect=dict,
    )
    raw_groups = payload.get("groups") if isinstance(payload, dict) else None
    if not isinstance(raw_groups, list):
        raise ValueError("judge response has no groups list")
    allowed = set(keys)
    by_display = {graph.nodes[key].display.lower(): key for key in keys}
    groups: list[list[str]] = []
    used: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list):
            continue
        group: list[str] = []
        for raw_key in raw_group:
            key = str(raw_key).strip()
            if key not in allowed:
                key = by_display.get(key.lower(), "")
            if not key or key in used or key in group:
                continue
            group.append(key)
        if len(group) >= 2:
            groups.append(group)
            used.update(group)
    return groups


def canonical_mapping(graph: Graph, groups: Iterable[list[str]]) -> dict[str, str]:
    mapping = {key: key for key in graph.nodes}
    for group in groups:
        valid = [key for key in group if key in graph.nodes]
        if len(valid) < 2:
            continue
        canonical = max(
            valid,
            key=lambda key: (
                graph.nodes[key].mentions,
                len(graph.nodes[key].display),
                key,
            ),
        )
        for key in valid:
            if key != canonical:
                mapping[key] = canonical
    # Collapse chains defensively, though groups are disjoint.
    for key in list(mapping):
        target = mapping[key]
        seen = {key}
        while mapping.get(target, target) != target and target not in seen:
            seen.add(target)
            target = mapping[target]
        mapping[key] = target
    return mapping


def rebuild_graph(graph: Graph, mapping: dict[str, str]) -> Graph:
    """Rebuild nodes and edges under the approved canonical keys."""
    type_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for key, node in graph.nodes.items():
        target = mapping.get(key, key)
        if node.type != "MISC":
            type_votes[target][node.type] += max(1, node.mentions)
    nodes: dict[str, GraphNode] = {}
    for key, node in graph.nodes.items():
        target = mapping.get(key, key)
        if target not in nodes:
            source = graph.nodes.get(target, node)
            inferred = source.type
            if inferred == "MISC":
                votes = type_votes.get(target) or Counter()
                inferred = _weighted_type(votes) or "MISC"
            nodes[target] = GraphNode(
                key=target,
                display=source.display,
                type=inferred,
                mentions=0,
                aliases=[],
            )
        merged = nodes[target]
        merged.mentions += node.mentions
        surfaces = {node.display, *node.aliases}
        if target == key:
            surfaces.add(nodes[target].display)
        merged.aliases = sorted({alias for alias in set(merged.aliases) | surfaces if alias != merged.display})

    edges: dict[tuple[str, str, str], GraphEdge] = {}
    dropped = 0
    for edge in graph.edges:
        head = mapping.get(edge.head, edge.head)
        tail = mapping.get(edge.tail, edge.tail)
        if head == tail:
            dropped += edge.mentions
            continue
        key = (head, edge.relation, tail)
        merged_edge = edges.get(key)
        if merged_edge is None:
            merged_edge = GraphEdge(head=head, relation=edge.relation, tail=tail)
            edges[key] = merged_edge
        merged_edge.mentions += edge.mentions
        for chunk in edge.source_chunks:
            if chunk not in merged_edge.source_chunks:
                merged_edge.source_chunks.append(chunk)

    rebuilt = Graph(
        nodes=nodes,
        edges=list(edges.values()),
        chunk_count=graph.chunk_count,
        fact_count=max(0, graph.fact_count - dropped),
    )
    for edge in rebuilt.edges:
        edge.source_chunks.sort()
    # Re-vote only for nodes still MISC; canonical display/type is retained.
    backfill_types(rebuilt)
    return rebuilt


def jaro_winkler(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    jaro = _jaro(a, b)
    prefix = 0
    for left, right in zip(a, b):
        if left != right or prefix >= 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1.0 - jaro)


def token_overlap(a: str, b: str) -> float:
    left, right = set(_tokens(a)), set(_tokens(b))
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _jaro(a: str, b: str) -> float:
    match_distance = max(len(a), len(b)) // 2 - 1
    match_distance = max(0, match_distance)
    a_matches = [False] * len(a)
    b_matches = [False] * len(b)
    matches = 0
    for index, char in enumerate(a):
        start = max(0, index - match_distance)
        end = min(index + match_distance + 1, len(b))
        for other in range(start, end):
            if not b_matches[other] and b[other] == char:
                a_matches[index] = True
                b_matches[other] = True
                matches += 1
                break
    if not matches:
        return 0.0
    transpositions = 0
    b_index = 0
    for index, char in enumerate(a):
        if not a_matches[index]:
            continue
        while not b_matches[b_index]:
            b_index += 1
        if char != b[b_index]:
            transpositions += 1
        b_index += 1
    transpositions //= 2
    return (
        matches / len(a)
        + matches / len(b)
        + (matches - transpositions) / matches
    ) / 3.0


def _tokens(key: str) -> list[str]:
    return [token for token in key.split() if token not in _STOPWORDS and len(token) > 1]


def _maker_made(graph: Graph, left: str, right: str) -> bool:
    for edge in graph.edges:
        if edge.relation not in _MAKER_RELATIONS:
            continue
        if {edge.head, edge.tail} == {left, right}:
            return True
    return False


def _components(pairs: list[Candidate]) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for candidate in pairs:
        union(candidate.a, candidate.b)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in parent:
        groups[find(key)].append(key)
    return [sorted(group) for group in sorted(groups.values()) if len(group) >= 2]


def _batch_components(components: list[list[str]], *, limit: int) -> list[list[str]]:
    batches: list[list[str]] = []
    for component in components:
        for start in range(0, len(component), BATCH_SIZE):
            batches.append(component[start : start + BATCH_SIZE])
            if len(batches) >= limit:
                return batches
    return batches


def _judge_user(graph: Graph, keys: list[str]) -> str:
    lines = ["Candidate entities:", ""]
    for key in keys:
        node = graph.nodes[key]
        lines.append(
            f"- key: {key}\n  name: {node.display}\n  type: {node.type}\n"
            f"  mentions: {node.mentions}\n  aliases: {', '.join(node.aliases) or 'none'}"
        )
        edges = sorted(
            graph.touching(key),
            key=lambda edge: (-edge.mentions, edge.id),
        )[:8]
        if edges:
            lines.append("  edges:")
            for edge in edges:
                direction = "out" if edge.head == key else "in"
                other = edge.tail if edge.head == key else edge.head
                lines.append(
                    f"    {direction} {edge.relation} -> {graph.nodes[other].display} "
                    f"(x{edge.mentions})"
                )
        else:
            lines.append("  edges: none")
        lines.append("")
    lines.append("Return the groups JSON now.")
    return "\n".join(lines)


def _weighted_type(votes: Counter[str]) -> str | None:
    if not votes:
        return None
    ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0] if ranked[0][1] > 0 else None


def _verdicts(candidates: list[Candidate], groups: list[list[str]]) -> list[dict[str, Any]]:
    merged_pairs = {tuple(sorted(pair)) for group in groups for pair in itertools.combinations(group, 2)}
    verdicts: list[dict[str, Any]] = []
    for candidate in candidates:
        pair = tuple(sorted((candidate.a, candidate.b)))
        verdicts.append(
            {
                "a": candidate.a,
                "b": candidate.b,
                "score": candidate.score,
                "jaro_winkler": candidate.jaro_winkler,
                "token_overlap": candidate.token_overlap,
                "verdict": "merge" if pair in merged_pairs else "keep separate",
            }
        )
    return verdicts


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
