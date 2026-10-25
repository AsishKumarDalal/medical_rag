"""G4 — measure the query pipeline instead of trusting impressions.

    python test/eval/run_eval.py --offline          # no model, deterministic vectors
    python test/eval/run_eval.py --live --k 8       # real router/reranker/narrator

Metrics per question: retrieval recall@k over hand-marked chunks, citation
coverage against real artifacts, abstain accuracy for out-of-scope questions,
latency and model calls. Writes ``test/eval/results.json`` and prints a markdown
table for ``docs/eval.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.config import GraphRagConfig  # noqa: E402
from graphrag.llm import GraphLLM  # noqa: E402
from graphrag.query import GraphRagEngine  # noqa: E402

QUESTIONS_PATH = Path(__file__).with_name("questions.json")
RESULTS_PATH = Path(__file__).with_name("results.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the GraphRAG query pipeline.")
    parser.add_argument("--offline", action="store_true", help="no model; hashed vectors and deterministic answers")
    parser.add_argument("--live", action="store_true", help="use the configured model endpoint")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--ids", nargs="+", help="run only these question ids (live smoke)")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse rows written by an interrupted run (same k and artifacts)",
    )
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args(argv)
    if args.offline and args.live:
        parser.error("choose --offline or --live, not both")
    live = not args.offline

    config = GraphRagConfig.from_env()
    if args.artifacts:
        config = replace(config, artifacts_dir=args.artifacts)
    if args.corpus:
        config = replace(config, corpus_dir=args.corpus)
    config = replace(config, llm_rerank=live)
    llm = None
    if live:
        llm = GraphLLM(
            config.llm,
            embed_model=config.embed_model,
            embed_base_url=config.embed_base_url,
            embed_api_key=config.embed_api_key,
        )
    engine = GraphRagEngine.load(config, llm)
    payload = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = payload["questions"]
    if args.ids:
        wanted = set(args.ids)
        questions = [item for item in questions if item.get("id") in wanted]
        missing = wanted - {item.get("id") for item in questions}
        if missing:
            parser.error(f"unknown question ids: {', '.join(sorted(missing))}")
    valid_ids = set(engine.chunks) | {
        str(community.get("id")) for community in engine.communities.get("communities", [])
    }

    partial_path = args.out.parent / (args.out.stem + ".partial.jsonl")
    reused: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.is_file():
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id"):
                reused[str(row["id"])] = row

    rows: list[dict[str, Any]] = []
    for item in questions:
        if item["id"] in reused:
            rows.append(reused[item["id"]])
            continue
        before_calls = llm.stats.calls if llm is not None else 0
        started = time.monotonic()
        try:
            result = engine.answer(item["question"], mode=item.get("mode", "auto"), k=args.k)
            error = result.error
        except Exception as exc:  # an eval run records failures rather than dying
            result = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        calls = (llm.stats.calls - before_calls) if llm is not None else 0
        row: dict[str, Any] = {
            "id": item["id"],
            "shape": item["shape"],
            "question": item["question"],
            "mode": result.mode if result else "error",
            "route": result.route if result else "error",
            "abstained": bool(result.abstained) if result else False,
            "citations": list(result.citations) if result else [],
            "seconds": round(elapsed, 3),
            "llm_calls": calls,
            "error": error,
        }
        relevant = [chunk for chunk in item.get("relevant_chunks", []) if chunk in engine.chunks]
        retrieved = _retrieved_chunks(result, args.k) if result else set()
        row["recall_at_k"] = (
            round(len(set(relevant) & retrieved) / len(relevant), 4) if relevant else None
        )
        row["citation_coverage"] = _citation_coverage(row["citations"], valid_ids)
        row["abstain_correct"] = (
            row["abstained"] == bool(item.get("must_abstain"))
            if item["shape"] == "unanswerable"
            else None
        )
        rows.append(row)
        # The live run can take hours on a slow endpoint; keep each finished row so
        # an interrupted run resumes with --resume instead of re-billing every call.
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = _summary(rows)
    snapshot = _snapshot(config)
    output = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "offline": not live,
        "k": args.k,
        "corpus_snapshot": snapshot,
        "summary": summary,
        "questions": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(_markdown(output))
    print(f"\nresults: {args.out}")
    return 0


def _retrieved_chunks(result: Any, k: int) -> set[str]:
    """The first ``k`` distinct chunks the pipeline actually used, in rank order."""
    found: list[str] = []
    for item in result.evidence:
        candidates: list[str] = []
        if item.get("kind") == "chunk" and item.get("id"):
            candidates.append(str(item["id"]))
        candidates.extend(str(chunk) for chunk in item.get("source_chunks", []) or [])
        for candidate in candidates:
            if candidate not in found:
                found.append(candidate)
            if len(found) >= k:
                return set(found)
    return set(found)


def _citation_coverage(citations: list[str], valid_ids: set[str]) -> float | None:
    if not citations:
        return None
    return round(sum(1 for citation in citations if citation in valid_ids) / len(citations), 4)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    local = [row for row in rows if row["shape"] == "local" and row["recall_at_k"] is not None]
    unanswerable = [row for row in rows if row["shape"] == "unanswerable"]
    coverages = [row["citation_coverage"] for row in rows if row["citation_coverage"] is not None]
    return {
        "questions": len(rows),
        "local_recall_at_k": round(sum(row["recall_at_k"] for row in local) / len(local), 4) if local else None,
        "abstain_accuracy": round(
            sum(1 for row in unanswerable if row["abstain_correct"]) / len(unanswerable), 4
        ) if unanswerable else None,
        "citation_coverage": round(sum(coverages) / len(coverages), 4) if coverages else None,
        "errors": sum(1 for row in rows if row["error"]),
        "avg_seconds": round(sum(row["seconds"] for row in rows) / len(rows), 3) if rows else 0,
        "llm_calls": sum(row["llm_calls"] for row in rows),
    }


def _snapshot(config: GraphRagConfig) -> dict[str, str | int]:
    index_path = config.index_path
    data = index_path.read_bytes() if index_path.is_file() else b""
    manifest = config.corpus_dir / "manifest.json"
    manifest_bytes = manifest.read_bytes() if manifest.is_file() else b""
    return {
        "index_sha256": hashlib.sha256(data).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "index_chunks": _count_chunks(index_path),
        "corpus_dir": str(config.corpus_dir),
    }


def _count_chunks(index_path: Path) -> int:
    if not index_path.is_file():
        return 0
    try:
        return int(json.loads(index_path.read_text(encoding="utf-8")).get("chunk_count", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _markdown(output: dict[str, Any]) -> str:
    summary = output["summary"]
    lines = [
        "| id | shape | mode | recall@k | cite coverage | abstain | sec | calls |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in output["questions"]:
        recall = "—" if row["recall_at_k"] is None else f"{row['recall_at_k']:.2f}"
        coverage = "—" if row["citation_coverage"] is None else f"{row['citation_coverage']:.2f}"
        abstain = "—" if row["abstain_correct"] is None else ("yes" if row["abstain_correct"] else "NO")
        lines.append(
            f"| {row['id']} | {row['shape']} | {row['mode']} | {recall} | {coverage} | "
            f"{abstain} | {row['seconds']:.1f} | {row['llm_calls']} |"
        )
    lines.extend(
        [
            "",
            f"**Summary:** recall@k={summary['local_recall_at_k']}, "
            f"abstain={summary['abstain_accuracy']}, citations={summary['citation_coverage']}, "
            f"errors={summary['errors']}, avg={summary['avg_seconds']}s, "
            f"calls={summary['llm_calls']}",
            "",
            f"Snapshot: `{output['corpus_snapshot']['index_sha256']}` "
            f"({output['corpus_snapshot']['index_chunks']} chunks)",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
