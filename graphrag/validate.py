"""I5 validation gate: cheap, deterministic rejections before normalization.

The gate filters; it does not fix. A rejected fact is recorded with a reason so a
run can be audited, and compound ``A and B`` mentions are split into separate
facts instead of silently becoming one node.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .normalize import clean_type, normalize_entity, normalize_relation

MAX_NAME_CHARS = 60
MAX_FIELD_CHARS = 100
MAX_NAME_TOKENS = 7

VAGUE_PREFIX = re.compile(
    r"^(?:other|various|several|many|some|numerous|multiple|different|a\s+few|"
    r"a\s+number\s+of|most|all|both|either)\b",
    re.IGNORECASE,
)
VAGUE_NAMES = {
    "people",
    "persons",
    "patients",
    "individuals",
    "researchers",
    "scientists",
    "doctors",
    "physicians",
    "healthcare providers",
    "healthcare professionals",
    "health professionals",
    "the public",
    "general population",
    "society",
    "others",
    "things",
}
JUNK_RELATIONS = {
    "RELATED_TO",
    "ASSOCIATED_WITH",
    "LINKED_TO",
    "CONNECTED_TO",
    "INVOLVES",
    "INVOLVED_IN",
    "HAS",
    "HAS_A",
    "HAVE",
    "IS",
    "ARE",
    "WAS",
    "WERE",
    "IS_A",
    "IS_AN",
    "CAN_BE",
    "MAY_BE",
    "EXISTS",
    "EXIST",
}
_COMPOUND = re.compile(r"\s+and\s+", re.IGNORECASE)


@dataclass(slots=True)
class ValidatedFact:
    head: str
    relation: str
    tail: str
    source_chunk: str

    def to_dict(self) -> dict[str, str]:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "source_chunk": self.source_chunk,
        }


@dataclass(slots=True)
class ValidationResult:
    triples: list[ValidatedFact] = field(default_factory=list)
    entities: list[dict[str, str]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "accepted": len(self.triples),
            "entities": len(self.entities),
            "rejected": len(self.rejected),
        }


def validate_record(record: dict[str, Any], *, max_triples: int = 20) -> ValidationResult:
    """Validate one checkpoint record (one chunk)."""
    chunk_id = str(record.get("chunk_id") or "")
    if not chunk_id:
        raise ValueError("checkpoint record has no chunk_id")
    result = ValidationResult()

    entities: list[dict[str, str]] = []
    for entity in record.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        problem = _name_problem(name)
        if problem:
            result.rejected.append({"chunk_id": chunk_id, "kind": "entity", "name": name, "reason": problem})
            continue
        entities.append({"name": name, "type": clean_type(str(entity.get("type") or ""))})
    # Deduplicate entities by normalized key, keeping the first (longest) surface form.
    seen_entities: dict[str, dict[str, str]] = {}
    for entity in entities:
        key = normalize_entity(entity["name"])
        if key and key not in seen_entities:
            seen_entities[key] = entity
        elif key and len(entity["name"]) > len(seen_entities[key]["name"]):
            seen_entities[key] = entity
    result.entities = sorted(seen_entities.values(), key=lambda item: normalize_entity(item["name"]))

    accepted: list[ValidatedFact] = []
    for raw in record.get("triples") or []:
        if not isinstance(raw, dict):
            result.rejected.append({"chunk_id": chunk_id, "kind": "triple", "reason": "not an object"})
            continue
        head = str(raw.get("head") or "").strip()
        relation = normalize_relation(str(raw.get("relation") or ""))
        tail = str(raw.get("tail") or "").strip()
        for fact in _validate_triple(chunk_id, head, relation, tail, result):
            accepted.append(fact)
        if len(accepted) >= max_triples:
            break

    deduped: list[ValidatedFact] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in accepted[:max_triples]:
        key = (normalize_entity(fact.head), fact.relation, normalize_entity(fact.tail))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(fact)
    result.triples = deduped
    return result


def validate_records(records: Iterable[dict[str, Any]], *, max_triples: int = 20) -> ValidationResult:
    combined = ValidationResult()
    for record in records:
        one = validate_record(record, max_triples=max_triples)
        combined.triples.extend(one.triples)
        combined.entities.extend(one.entities)
        combined.rejected.extend(one.rejected)
    return combined


def _validate_triple(
    chunk_id: str,
    head: str,
    relation: str,
    tail: str,
    result: ValidationResult,
) -> list[ValidatedFact]:
    if not head or not relation or not tail:
        result.rejected.append(
            {
                "chunk_id": chunk_id,
                "kind": "triple",
                "triple": {"head": head, "relation": relation, "tail": tail},
                "reason": "head, relation and tail must all be non-empty",
            }
        )
        return []
    if len(relation) > MAX_FIELD_CHARS:
        result.rejected.append(
            {
                "chunk_id": chunk_id,
                "kind": "triple",
                "triple": {"head": head, "relation": relation, "tail": tail},
                "reason": f"relation is longer than {MAX_FIELD_CHARS} characters",
            }
        )
        return []
    if relation in JUNK_RELATIONS:
        result.rejected.append(
            {
                "chunk_id": chunk_id,
                "kind": "triple",
                "triple": {"head": head, "relation": relation, "tail": tail},
                "reason": f"junk relation {relation}",
            }
        )
        return []

    heads = _split_compound(head)
    tails = _split_compound(tail)
    facts: list[ValidatedFact] = []
    for one_head in heads:
        for one_tail in tails:
            problem = _name_problem(one_head) or _name_problem(one_tail)
            if problem:
                result.rejected.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "triple",
                        "triple": {"head": one_head, "relation": relation, "tail": one_tail},
                        "reason": problem,
                    }
                )
                continue
            if normalize_entity(one_head) == normalize_entity(one_tail):
                result.rejected.append(
                    {
                        "chunk_id": chunk_id,
                        "kind": "triple",
                        "triple": {"head": one_head, "relation": relation, "tail": one_tail},
                        "reason": "self-loop after normalization",
                    }
                )
                continue
            facts.append(ValidatedFact(one_head, relation, one_tail, chunk_id))
    return facts


def _split_compound(name: str) -> list[str]:
    if not _COMPOUND.search(name):
        return [name]
    parts = [part.strip() for part in _COMPOUND.split(name) if part.strip()]
    return parts or [name]


def _name_problem(name: str) -> str | None:
    if not name:
        return "name is empty"
    if len(name) > MAX_NAME_CHARS:
        return f"name is longer than {MAX_NAME_CHARS} characters"
    if len(name.split()) > MAX_NAME_TOKENS:
        return f"name has more than {MAX_NAME_TOKENS} tokens"
    lowered = name.strip().lower()
    if lowered in VAGUE_NAMES:
        return "vague group, not an identifiable entity"
    if VAGUE_PREFIX.search(lowered):
        return "vague group, not an identifiable entity"
    if not normalize_entity(name):
        return "name has no alphanumeric identity"
    return None
