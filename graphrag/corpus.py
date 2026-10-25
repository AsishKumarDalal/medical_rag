"""I1 acquisition: one reproducible article record per corpus file."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "manifest.json"
_BLANKS = re.compile(r"\n{3,}")


class CorpusError(RuntimeError):
    """The corpus directory is missing or unusable."""


@dataclass(frozen=True, slots=True)
class Article:
    id: str
    title: str
    source: str
    url: str
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)


def load_corpus(corpus_dir: str | Path) -> list[Article]:
    """Load every ``*.txt`` article, sorted by id, with manifest title if present."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise CorpusError(f"corpus directory not found: {root}")
    manifest = _read_manifest(root)
    articles: list[Article] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.txt")):
        article_id = path.stem
        if article_id in seen:
            raise CorpusError(f"duplicate article id {article_id!r}")
        seen.add(article_id)
        text = _clean_text(path.read_text(encoding="utf-8"))
        if not text:
            raise CorpusError(f"article {path.name} is empty")
        entry = manifest.get(path.name, {})
        articles.append(
            Article(
                id=article_id,
                title=str(entry.get("title") or article_id.replace("_", " ")),
                source=str(entry.get("source") or "unknown"),
                url=str(entry.get("url") or ""),
                text=text,
            )
        )
    if not articles:
        raise CorpusError(f"no .txt articles in {root}")
    return articles


def read_manifest(corpus_dir: str | Path) -> dict:
    return _read_manifest(Path(corpus_dir))


def _read_manifest(root: Path) -> dict[str, dict]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path} is not valid JSON: {exc}") from exc
    return {
        str(entry.get("file")): entry
        for entry in payload.get("articles", [])
        if isinstance(entry, dict) and entry.get("file")
    }


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    return _BLANKS.sub("\n\n", cleaned).strip()
