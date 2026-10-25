"""I2 chunking: deterministic, paragraph-aware windows with overlap.

The chunk id is the citation. Re-running this module on the same corpus must
produce the same chunks and ids forever, so every decision here is deterministic:
greedy packing in input order, sentence boundaries for oversized paragraphs, and
a character-counted overlap prepended from the previous chunk.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import GraphRagConfig
from .corpus import Article

INDEX_VERSION = 1
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class ChunkError(RuntimeError):
    """The article or chunk index cannot be built."""


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    article: str
    title: str
    position: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class ChunkIndex:
    articles: list[dict[str, Any]]
    chunks: list[Chunk] = field(default_factory=list)
    target: int = 1_000
    overlap: int = 150

    @property
    def count(self) -> int:
        return len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        articles: list[dict[str, Any]] = []
        for article in self.articles:
            record = dict(article)
            if not record.get("chunks"):
                # Keep ``self.chunks`` authoritative for callers that construct an
                # index directly; the on-disk shape stays one chunk list per article.
                record["chunks"] = [
                    {
                        "id": chunk.id,
                        "article": chunk.article,
                        "title": chunk.title,
                        "position": chunk.position,
                        "chars": chunk.chars,
                        "text": chunk.text,
                    }
                    for chunk in self.chunks
                    if chunk.article == record["id"]
                ]
                record["chunk_count"] = len(record["chunks"])
                record["chunk_ids"] = [chunk["id"] for chunk in record["chunks"]]
            articles.append(record)
        return {
            "version": INDEX_VERSION,
            "chunk_target": self.target,
            "chunk_overlap": self.overlap,
            "article_count": len(articles),
            "chunk_count": len(self.chunks),
            "total_chars": sum(chunk.chars for chunk in self.chunks),
            "articles": articles,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkIndex":
        chunks: list[Chunk] = []
        articles: list[dict[str, Any]] = []
        for article in payload.get("articles", []):
            record = dict(article)
            records = record.pop("chunks", [])
            articles.append(record)
            for chunk in records:
                chunks.append(
                    Chunk(
                        id=str(chunk["id"]),
                        article=str(chunk.get("article") or record["id"]),
                        title=str(chunk.get("title") or record.get("title") or ""),
                        position=int(chunk["position"]),
                        text=str(chunk["text"]),
                    )
                )
        return cls(
            articles=articles,
            chunks=chunks,
            target=int(payload.get("chunk_target", 1_000)),
            overlap=int(payload.get("chunk_overlap", 150)),
        )


def chunk_article(article: Article, *, target: int = 1_000, overlap: int = 150) -> list[Chunk]:
    """Split one article into stable, overlapping chunks."""
    paragraphs = _paragraphs(article.text)
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= target:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_paragraph(paragraph, target))

    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if current and len(candidate) > target:
            packed.append(current)
            current = piece
        else:
            current = candidate
    if current:
        packed.append(current)

    chunks: list[Chunk] = []
    previous = ""
    for position, body in enumerate(packed, start=1):
        prefix = _overlap_tail(previous, overlap)
        text = f"{prefix}\n\n{body}" if prefix else body
        chunks.append(
            Chunk(
                id=f"{article.id}::{position}",
                article=article.id,
                title=article.title,
                position=position,
                text=text,
            )
        )
        previous = body
    return chunks


def build_chunk_index(
    articles: Iterable[Article],
    *,
    target: int = 1_000,
    overlap: int = 150,
) -> ChunkIndex:
    if target < 200:
        raise ChunkError("chunk target must be at least 200 characters")
    if not 0 <= overlap < target:
        raise ChunkError("chunk overlap must be smaller than the target")
    article_records: list[dict[str, Any]] = []
    chunks: list[Chunk] = []
    for article in articles:
        article_chunks = chunk_article(article, target=target, overlap=overlap)
        if not article_chunks:
            raise ChunkError(f"article {article.id!r} produced no chunks")
        article_records.append(
            {
                "id": article.id,
                "title": article.title,
                "source": article.source,
                "url": article.url,
                "chars": article.chars,
                "chunk_count": len(article_chunks),
                "chunk_ids": [chunk.id for chunk in article_chunks],
                "chunks": [
                    {
                        "id": chunk.id,
                        "article": chunk.article,
                        "title": chunk.title,
                        "position": chunk.position,
                        "chars": chunk.chars,
                        "text": chunk.text,
                    }
                    for chunk in article_chunks
                ],
            }
        )
        chunks.extend(article_chunks)
    return ChunkIndex(articles=article_records, chunks=chunks, target=target, overlap=overlap)


def write_index(index: ChunkIndex, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def load_index(path: str | Path) -> ChunkIndex:
    target = Path(path)
    if not target.is_file():
        raise ChunkError(f"chunk index not found: {target}; run the chunk stage first")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChunkError(f"{target} is not valid JSON: {exc}") from exc
    if int(payload.get("version", 0)) != INDEX_VERSION:
        raise ChunkError(f"{target} has unknown version {payload.get('version')!r}")
    return ChunkIndex.from_dict(payload)


def load_or_build(config: GraphRagConfig, *, force: bool = False) -> ChunkIndex:
    if config.index_path.is_file() and not force:
        return load_index(config.index_path)
    from .corpus import load_corpus

    articles = load_corpus(config.corpus_dir)
    index = build_chunk_index(
        articles,
        target=config.chunk_target,
        overlap=config.chunk_overlap,
    )
    write_index(index, config.index_path)
    return index


def chunk_texts(index: ChunkIndex) -> dict[str, str]:
    return {chunk.id: chunk.text for chunk in index.chunks}


def _paragraphs(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        raise ChunkError("article has no paragraphs")
    return paragraphs


def _split_paragraph(paragraph: str, target: int) -> list[str]:
    """Split an oversized paragraph at sentence boundaries, then hard-cut."""
    sentences = [part for part in _SENTENCE_END.split(paragraph) if part]
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > target:
            cut = sentence.rfind(" ", 0, target)
            cut = cut if cut > target // 2 else target
            head, sentence = sentence[:cut].rstrip(), sentence[cut:].lstrip()
            if current:
                pieces.append(current)
                current = ""
            if head:
                pieces.append(head)
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > target:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(previous: str, overlap: int) -> str:
    if not previous or overlap <= 0:
        return ""
    tail = previous[-overlap:]
    space = tail.find(" ")
    if space >= 0:
        tail = tail[space + 1 :]
    return tail.strip()


def chunks_for_articles(index: ChunkIndex, article_ids: Sequence[str]) -> list[Chunk]:
    wanted = set(article_ids)
    return [chunk for chunk in index.chunks if chunk.article in wanted]
