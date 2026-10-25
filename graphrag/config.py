"""Environment-driven settings for the GraphRAG pipeline.

The corpus is read from ``test/data/corpus`` by default and every derived artifact
is written under ``graphrag/artifacts``. Both are configurable so tests can use a
tiny fixture corpus and a temporary artifact directory.

Model settings deliberately reuse the harness environment (``LLM_*``): there is
one endpoint and one model name in this project. Embeddings are separate because
a chat endpoint does not necessarily serve an embeddings route.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from harness.config import Config, ConfigError, load_env_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "test" / "data" / "corpus"
DEFAULT_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

DEFAULT_CHUNK_TARGET = 1_000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MAX_TRIPLES = 20
DEFAULT_EXTRACT_MAX_TOKENS = 4_096
DEFAULT_EXTRACT_WORKERS = 1
DEFAULT_EXTRACT_ATTEMPTS = 2
DEFAULT_EMBED_DIM = 256
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_BATCH = 64
DEFAULT_JUDGE_MAX_BATCHES = 40
DEFAULT_COMMUNITY_MIN_SIZE = 3
DEFAULT_WALK_HOPS = 2
DEFAULT_SEED_THRESHOLD = 0.65
DEFAULT_CHUNK_THRESHOLD = 0.30
DEFAULT_RERANK_KEEP = 15
DEFAULT_LLM_RERANK = True


class GraphRagConfigError(RuntimeError):
    """A pipeline path or setting is unusable."""


@dataclass(frozen=True, slots=True)
class GraphRagConfig:
    corpus_dir: Path
    artifacts_dir: Path
    llm: Config
    chunk_target: int = DEFAULT_CHUNK_TARGET
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    max_triples_per_chunk: int = DEFAULT_MAX_TRIPLES
    extract_max_tokens: int = DEFAULT_EXTRACT_MAX_TOKENS
    extract_workers: int = DEFAULT_EXTRACT_WORKERS
    extract_attempts: int = DEFAULT_EXTRACT_ATTEMPTS
    embed_model: str | None = None
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_dim: int = DEFAULT_EMBED_DIM
    embed_batch: int = DEFAULT_EMBED_BATCH
    judge_max_batches: int = DEFAULT_JUDGE_MAX_BATCHES
    community_min_size: int = DEFAULT_COMMUNITY_MIN_SIZE
    walk_hops: int = DEFAULT_WALK_HOPS
    seed_threshold: float = DEFAULT_SEED_THRESHOLD
    chunk_threshold: float = DEFAULT_CHUNK_THRESHOLD
    rerank_keep: int = DEFAULT_RERANK_KEEP
    llm_rerank: bool = DEFAULT_LLM_RERANK
    max_seeds: int = 6

    def __post_init__(self) -> None:
        if self.chunk_target < 200:
            raise GraphRagConfigError("chunk target must be at least 200 characters")
        if not 0 <= self.chunk_overlap < self.chunk_target:
            raise GraphRagConfigError("chunk overlap must be smaller than the chunk target")
        if self.max_triples_per_chunk <= 0:
            raise GraphRagConfigError("max triples per chunk must be positive")
        if self.extract_workers <= 0:
            raise GraphRagConfigError("extract workers must be positive")
        if self.extract_attempts <= 0:
            raise GraphRagConfigError("extract attempts must be positive")
        if self.embed_dim < 16:
            raise GraphRagConfigError("embedding dimension must be at least 16")
        if not 0 < self.seed_threshold <= 1 or not 0 <= self.chunk_threshold <= 1:
            raise GraphRagConfigError("similarity thresholds must be between 0 and 1")

    @property
    def index_path(self) -> Path:
        return self.artifacts_dir / "index.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.artifacts_dir / "triples.jsonl"

    @property
    def failed_path(self) -> Path:
        return self.artifacts_dir / "failed_chunks.json"

    @property
    def rejections_path(self) -> Path:
        return self.artifacts_dir / "rejections.jsonl"

    @property
    def graph_path(self) -> Path:
        return self.artifacts_dir / "graph.json"

    @property
    def merge_log_path(self) -> Path:
        return self.artifacts_dir / "merge_log.json"

    @property
    def communities_path(self) -> Path:
        return self.artifacts_dir / "communities.json"

    @property
    def vectors_path(self) -> Path:
        return self.artifacts_dir / "vectors.json"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_dotenv: bool = True,
    ) -> "GraphRagConfig":
        if env is None and load_dotenv:
            load_env_file()
        src: Mapping[str, str] = os.environ if env is None else env
        try:
            llm = Config.from_env(env=src, load_dotenv=False)
        except ConfigError as exc:
            raise GraphRagConfigError(str(exc)) from exc
        corpus = Path(src.get("GRAPHRAG_CORPUS", "").strip() or DEFAULT_CORPUS)
        artifacts = Path(src.get("GRAPHRAG_ARTIFACTS", "").strip() or DEFAULT_ARTIFACTS)
        embed_model = src.get("GRAPHRAG_EMBED_MODEL", "").strip() or None
        return cls(
            corpus_dir=corpus,
            artifacts_dir=artifacts,
            llm=llm,
            chunk_target=_int(src, "GRAPHRAG_CHUNK_TARGET", DEFAULT_CHUNK_TARGET),
            chunk_overlap=_int(src, "GRAPHRAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
            max_triples_per_chunk=_int(src, "GRAPHRAG_MAX_TRIPLES", DEFAULT_MAX_TRIPLES),
            extract_max_tokens=_int(src, "GRAPHRAG_EXTRACT_MAX_TOKENS", DEFAULT_EXTRACT_MAX_TOKENS),
            extract_workers=_int(src, "GRAPHRAG_EXTRACT_WORKERS", DEFAULT_EXTRACT_WORKERS),
            extract_attempts=_int(src, "GRAPHRAG_EXTRACT_ATTEMPTS", DEFAULT_EXTRACT_ATTEMPTS),
            embed_model=embed_model,
            embed_base_url=src.get("GRAPHRAG_EMBED_BASE_URL", "").strip() or None,
            embed_api_key=src.get("GRAPHRAG_EMBED_API_KEY", "").strip() or None,
            embed_dim=_int(src, "GRAPHRAG_EMBED_DIM", DEFAULT_EMBED_DIM),
            embed_batch=_int(src, "GRAPHRAG_EMBED_BATCH", DEFAULT_EMBED_BATCH),
            judge_max_batches=_int(src, "GRAPHRAG_JUDGE_MAX_BATCHES", DEFAULT_JUDGE_MAX_BATCHES),
            community_min_size=_int(src, "GRAPHRAG_COMMUNITY_MIN_SIZE", DEFAULT_COMMUNITY_MIN_SIZE),
            walk_hops=_int(src, "GRAPHRAG_WALK_HOPS", DEFAULT_WALK_HOPS),
            seed_threshold=_float(src, "GRAPHRAG_SEED_THRESHOLD", DEFAULT_SEED_THRESHOLD),
            chunk_threshold=_float(src, "GRAPHRAG_CHUNK_THRESHOLD", DEFAULT_CHUNK_THRESHOLD),
            rerank_keep=_int(src, "GRAPHRAG_RERANK_KEEP", DEFAULT_RERANK_KEEP),
            llm_rerank=_bool(src, "GRAPHRAG_LLM_RERANK", DEFAULT_LLM_RERANK),
            max_seeds=_int(src, "GRAPHRAG_MAX_SEEDS", 6),
        )

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise GraphRagConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise GraphRagConfigError(f"{key} must be a boolean, got {raw!r}")


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise GraphRagConfigError(f"{key} must be a number, got {raw!r}") from exc
