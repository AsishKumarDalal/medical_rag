"""I3 extraction and I4 checkpointing.

One LLM call per chunk. The result is appended to ``triples.jsonl`` *before* any
graph work, so an interrupted run resumes by chunk id. Permanently failed chunks
are recorded in ``failed_chunks.json`` and are not retried unless explicitly asked.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .chunk import Chunk, ChunkIndex
from .config import GraphRagConfig
from .llm import GraphLLM, GraphRagLLMError

SYSTEM_PROMPT = """You are a precise entity and relationship extraction engine.
You respond with ONLY one valid JSON object, no commentary and no markdown.

Extract entities and the relationships explicitly stated in the text.
Rules:
1. Entity names use the most complete form found in the text.
2. Relations are UPPER_SNAKE_CASE: TREATS, CAUSES, SYMPTOM_OF, DIAGNOSED_BY,
   RISK_FACTOR_FOR, AFFECTS, PREVENTS, LOCATED_IN, PART_OF, ASSOCIATED_WITH, etc.
3. Extract ONLY facts explicitly stated in the text. Never invent or infer.
4. Entity types: PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT, MISC, DISEASE,
   DRUG, PROCEDURE, SYMPTOM, ANATOMY, ORGANISM, GENE, RISK_FACTOR.
5. At most 20 triples. Skip trivial facts and generic groups such as "patients".
6. Keep entity names short (2-5 words) and do not list every symptom separately.
Return exactly:
{"entities": [{"name": "...", "type": "..."}],
 "triples": [{"head": "...", "relation": "...", "tail": "..."}]}"""


@dataclass(slots=True)
class ExtractionStats:
    attempted: int = 0
    written: int = 0
    skipped: int = 0
    failed: int = 0
    triples: int = 0
    entities: int = 0


class Checkpoint:
    """Append-only JSONL plus a small failed-chunk ledger."""

    def __init__(self, path: str | Path, failed_path: str | Path) -> None:
        self.path = Path(path)
        self.failed_path = Path(failed_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.failed_path.parent.mkdir(parents=True, exist_ok=True)

    def done_ids(self) -> set[str]:
        ids: set[str] = set()
        if not self.path.is_file():
            return ids
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk_id = str(record.get("chunk_id") or "")
            if chunk_id:
                ids.add(chunk_id)
        return ids

    def failed(self) -> dict[str, dict[str, Any]]:
        if not self.failed_path.is_file():
            return {}
        try:
            payload = json.loads(self.failed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        records = payload.get("chunks", payload) if isinstance(payload, dict) else payload
        return {
            str(record["chunk_id"]): dict(record)
            for record in records
            if isinstance(record, dict) and record.get("chunk_id")
        }

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def mark_failed(self, chunk_id: str, error: str) -> dict[str, Any]:
        failed = self.failed()
        previous = failed.get(chunk_id, {})
        record = {
            "chunk_id": chunk_id,
            "attempts": int(previous.get("attempts", 0)) + 1,
            "error": error[:500],
            "at": _now(),
        }
        failed[chunk_id] = record
        self._write_failed(failed)
        return record

    def clear_failed(self, chunk_ids: Iterable[str]) -> None:
        failed = self.failed()
        removed = False
        for chunk_id in chunk_ids:
            if failed.pop(chunk_id, None) is not None:
                removed = True
        if removed:
            self._write_failed(failed)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()
        if self.failed_path.exists():
            self.failed_path.unlink()

    def _write_failed(self, failed: dict[str, dict[str, Any]]) -> None:
        records = sorted(failed.values(), key=lambda record: record["chunk_id"])
        payload = {"count": len(records), "chunks": records}
        tmp = self.failed_path.with_suffix(self.failed_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.failed_path)


def extract_chunks(
    index: ChunkIndex,
    llm: GraphLLM,
    config: GraphRagConfig,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
    workers: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> ExtractionStats:
    checkpoint = Checkpoint(config.checkpoint_path, config.failed_path)
    if force:
        checkpoint.reset()
    done = checkpoint.done_ids()
    failed = checkpoint.failed()
    if retry_failed:
        checkpoint.clear_failed(failed.keys())
        failed = {}

    chunks = [chunk for chunk in index.chunks if chunk.id not in done and chunk.id not in failed]
    if limit is not None:
        chunks = chunks[: max(0, limit)]
    stats = ExtractionStats(skipped=len(index.chunks) - len(chunks))
    active_workers = max(1, workers if workers is not None else config.extract_workers)
    if active_workers == 1:
        for position, chunk in enumerate(chunks, start=1):
            if progress is not None:
                progress(position, len(chunks), chunk.id)
            stats.attempted += 1
            started = time.monotonic()
            try:
                record = extract_chunk(
                    chunk,
                    llm,
                    max_triples=config.max_triples_per_chunk,
                    max_tokens=config.extract_max_tokens,
                )
            except (GraphRagLLMError, ValueError) as exc:
                stats.failed += 1
                checkpoint.mark_failed(chunk.id, f"{type(exc).__name__}: {exc}")
                continue
            _record_success(checkpoint, stats, record, started)
        return stats

    # The model call is slow; the checkpoint is still written by one thread only.
    with ThreadPoolExecutor(max_workers=active_workers) as pool:
        futures = {
            pool.submit(
                _timed_extract,
                chunk,
                llm,
                config.max_triples_per_chunk,
                config.extract_max_tokens,
            ): chunk
            for chunk in chunks
        }
        for position, future in enumerate(as_completed(futures), start=1):
            chunk = futures[future]
            stats.attempted += 1
            if progress is not None:
                progress(position, len(chunks), chunk.id)
            try:
                record, seconds = future.result()
            except (GraphRagLLMError, ValueError) as exc:
                stats.failed += 1
                checkpoint.mark_failed(chunk.id, f"{type(exc).__name__}: {exc}")
                continue
            except Exception as exc:  # a worker bug must still leave a checkpoint
                stats.failed += 1
                checkpoint.mark_failed(chunk.id, f"{type(exc).__name__}: {exc}")
                continue
            record["seconds"] = round(seconds, 3)
            _record_success(checkpoint, stats, record, None)
    return stats


def _timed_extract(chunk: Chunk, llm: GraphLLM, max_triples: int, max_tokens: int):
    started = time.monotonic()
    record = extract_chunk(chunk, llm, max_triples=max_triples, max_tokens=max_tokens)
    return record, time.monotonic() - started


def _record_success(
    checkpoint: Checkpoint,
    stats: ExtractionStats,
    record: dict[str, Any],
    started: float | None,
) -> None:
    if started is not None and not record.get("seconds"):
        record["seconds"] = round(time.monotonic() - started, 3)
    checkpoint.append(record)
    stats.written += 1
    stats.triples += len(record["triples"])
    stats.entities += len(record["entities"])


def extract_chunk(chunk: Chunk, llm: GraphLLM, *, max_triples: int = 20, max_tokens: int = 4_096) -> dict[str, Any]:
    user = _prompt(chunk, max_triples=max_triples)
    payload = llm.complete_json(
        system=SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        attempts=2,
        expect=(dict, list),
        retry_note="Cap the result at 12 entities and 12 triples.",
    )
    entities, triples = normalize_payload(payload, max_triples=max_triples)
    return {
        "chunk_id": chunk.id,
        "doc": chunk.article,
        "title": chunk.title,
        "entities": entities,
        "triples": triples,
        "extracted_at": _now(),
    }


def normalize_payload(payload: Any, *, max_triples: int = 20) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Accept the requested shape and the common near-misses, without trusting them."""
    entities_raw: Any = []
    triples_raw: Any = []
    if isinstance(payload, dict):
        entities_raw = payload.get("entities") or []
        triples_raw = payload.get("triples") or payload.get("relationships") or payload.get("facts") or []
    elif isinstance(payload, list):
        triples_raw = payload
    else:
        raise ValueError(f"extraction output must be an object or list, got {type(payload).__name__}")

    entities: list[dict[str, str]] = []
    for entity in entities_raw if isinstance(entities_raw, list) else []:
        if not isinstance(entity, dict):
            continue
        name = str(entity.get("name") or "").strip()
        kind = str(entity.get("type") or "MISC").strip().upper()
        if name:
            entities.append({"name": name, "type": kind})

    triples: list[dict[str, str]] = []
    for triple in triples_raw if isinstance(triples_raw, list) else []:
        if not isinstance(triple, dict):
            continue
        head = str(triple.get("head") or triple.get("subject") or "").strip()
        relation = str(triple.get("relation") or triple.get("predicate") or triple.get("relationship") or "").strip()
        tail = str(triple.get("tail") or triple.get("object") or "").strip()
        if head and relation and tail:
            triples.append({"head": head, "relation": relation, "tail": tail})
        if len(triples) >= max_triples:
            break
    return entities, triples


def load_records(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"extraction checkpoint not found: {target}")
    records: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk_id = str(record.get("chunk_id") or "")
        if chunk_id:
            records[chunk_id] = record
    return [records[key] for key in sorted(records)]


def _prompt(chunk: Chunk, *, max_triples: int) -> str:
    return (
        f"Article: {chunk.title}\n"
        f"Chunk id: {chunk.id}\n"
        f"Maximum triples: {max_triples}\n\n"
        "TEXT:\n<<<\n"
        f"{chunk.text}\n"
        ">>>\n\n"
        "Return the JSON object now."
    )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
