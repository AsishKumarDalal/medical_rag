"""Stage CLI for the indexing pipeline.

    python -m graphrag.build --stage extract --limit 20
    python -m graphrag.build --stage graph
    python -m graphrag.build --stage resolve
    python -m graphrag.build --stage communities
    python -m graphrag.build --stage embed
    python -m graphrag.build --stage all

Each stage reads the previous stage's artifact and writes its own. Extraction is
the only paid stage; every later stage can be rebuilt from ``triples.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .chunk import ChunkIndex, load_or_build
from .communities import build_communities, save_communities
from .config import GraphRagConfig, GraphRagConfigError
from .embed import build_vectors
from .extract import ExtractionStats, extract_chunks, load_records
from .graph import build_graph, load_graph, save_graph, save_rejections
from .llm import GraphLLM
from .resolve import resolve_entities, write_merge_audit

STAGES = ("extract", "graph", "resolve", "communities", "embed", "all")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the GraphRAG index.")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--corpus", type=Path, help="corpus directory")
    parser.add_argument("--artifacts", type=Path, help="artifact directory")
    parser.add_argument("--limit", type=int, help="extract at most N pending chunks")
    parser.add_argument("--resume", action="store_true", help="resume extraction (the default)")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed chunks")
    parser.add_argument("--workers", type=int, help="parallel extraction calls")
    parser.add_argument("--force", action="store_true", help="rebuild this stage from scratch")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = GraphRagConfig.from_env()
    except GraphRagConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    if args.corpus:
        config = replace(config, corpus_dir=args.corpus)
    if args.artifacts:
        config = replace(config, artifacts_dir=args.artifacts)
    config.ensure_dirs()

    llm = GraphLLM(
        config.llm,
        embed_model=config.embed_model,
        embed_base_url=config.embed_base_url,
        embed_api_key=config.embed_api_key,
    )
    log = (lambda line: None) if args.quiet else (lambda line: print(line, flush=True))
    started = time.monotonic()
    try:
        if args.stage in ("extract", "all"):
            stage_extract(
                config,
                llm,
                limit=args.limit,
                force=args.force,
                retry_failed=args.retry_failed,
                workers=args.workers,
                log=log,
            )
        if args.stage in ("graph", "all"):
            stage_graph(config, log=log)
        if args.stage in ("resolve", "all"):
            stage_resolve(config, llm, log=log)
        if args.stage in ("communities", "all"):
            stage_communities(config, llm, log=log)
        if args.stage in ("embed", "all"):
            stage_embed(config, llm, force=args.force, log=log)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[build] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    log(f"[build] stage={args.stage} done in {time.monotonic() - started:.1f}s; stats={json.dumps(llm.stats.to_dict())}")
    return 0


def stage_extract(
    config: GraphRagConfig,
    llm: GraphLLM,
    *,
    limit: int | None = None,
    force: bool = False,
    retry_failed: bool = False,
    workers: int | None = None,
    log=lambda line: print(line, flush=True),
) -> ChunkIndex:
    index = load_or_build(config)
    log(f"[index] {index.count} chunks from {len(index.articles)} articles")

    def progress(position: int, total: int, chunk_id: str) -> None:
        log(f"[extract] {position}/{total} {chunk_id}")

    stats = extract_chunks(
        index,
        llm,
        config,
        limit=limit,
        force=force,
        retry_failed=retry_failed,
        workers=workers,
        progress=progress,
    )
    _print_extract_stats(stats, log)
    return index


def stage_graph(config: GraphRagConfig, *, log=lambda line: print(line, flush=True)):
    records = load_records(config.checkpoint_path)
    if not records:
        raise ValueError("no checkpoint records; run --stage extract first")
    build = build_graph(records, max_triples=config.max_triples_per_chunk)
    save_graph(build.graph, config.graph_path)
    save_rejections(build.validation.rejected, config.rejections_path)
    log(
        f"[graph] {len(build.graph.nodes)} nodes, {len(build.graph.edges)} edges, "
        f"{build.graph.fact_count} facts, {len(build.validation.rejected)} rejections"
    )
    return build.graph


def stage_resolve(config: GraphRagConfig, llm: GraphLLM | None, *, log=lambda line: print(line, flush=True)):
    graph = load_graph(config.graph_path)
    resolution = resolve_entities(graph, llm, max_batches=config.judge_max_batches)
    save_graph(resolution.graph, config.graph_path)
    config.merge_log_path.write_text(json.dumps(resolution.log, indent=2) + "\n", encoding="utf-8")
    write_merge_audit(resolution, config.merge_audit_path)
    log(
        f"[resolve] {resolution.merged} nodes merged; "
        f"{len(graph.nodes)} -> {len(resolution.graph.nodes)} nodes; "
        f"{len(resolution.log.get('batches', []))} judge batches; "
        f"{len(resolution.log.get('verdicts', []))} candidates audited"
    )
    return resolution


def stage_communities(config: GraphRagConfig, llm: GraphLLM | None, *, log=lambda line: print(line, flush=True)):
    graph = load_graph(config.graph_path)
    build = build_communities(graph, llm, min_size=config.community_min_size)
    save_communities(build.payload, config.communities_path)
    log(
        f"[communities] {len(build.communities)} communities "
        f"({sum(len(c.members) for c in build.communities)} assigned nodes, "
        f"{len(build.payload['unassigned'])} unassigned)"
    )
    return build


def stage_embed(
    config: GraphRagConfig,
    llm: GraphLLM | None,
    *,
    force: bool = False,
    log=lambda line: print(line, flush=True),
):
    graph = load_graph(config.graph_path)
    index = load_or_build(config)
    store = build_vectors(config, graph, index, llm, force=force)
    log(
        f"[embed] backend={store.backend} model={store.model} dim={store.dim} "
        f"nodes={len(store.nodes)} facts={len(store.facts)} chunks={len(store.chunks)}"
    )
    if store.fallback_reason:
        log(f"[embed] fallback: {store.fallback_reason}")
    return store


def _print_extract_stats(stats: ExtractionStats, log) -> None:
    log(
        f"[extract] attempted={stats.attempted} written={stats.written} "
        f"failed={stats.failed} skipped={stats.skipped} "
        f"triples={stats.triples} entities={stats.entities}"
    )


if __name__ == "__main__":
    sys.exit(main())
