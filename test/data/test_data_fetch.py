"""Fetch a small, reproducible medical corpus.

Source: the Wikipedia extracts API (plain text, no HTML boilerplate, no JavaScript).
24 topics, capped per article so one long page cannot dominate the graph. Everything
is written as one file per article plus a manifest with sizes and hashes, so the
corpus in use is identifiable after the fact.

    python test/data/test_data_fetch.py                 # fetch what is missing
    python test/data/test_data_fetch.py --force         # refetch everything
    python test/data/test_data_fetch.py --limit 5       # smaller run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

API = "https://en.wikipedia.org/w/api.php"
SOURCE = "en.wikipedia.org"
USER_AGENT = "medical-rag-harness/0.1 (https://github.com/AsishKumarDalal/medical_rag)"
DEFAULT_OUT = Path(__file__).resolve().parent / "corpus"
MANIFEST_NAME = "manifest.json"
DEFAULT_MAX_CHARS = 12_000

TOPICS: list[str] = [
    "Asthma",
    "Diabetes",
    "Hypertension",
    "Myocardial infarction",
    "Stroke",
    "Pneumonia",
    "Tuberculosis",
    "Malaria",
    "HIV/AIDS",
    "Sepsis",
    "Breast cancer",
    "Lung cancer",
    "Colorectal cancer",
    "Alzheimer's disease",
    "Parkinson's disease",
    "Epilepsy",
    "Migraine",
    "Major depressive disorder",
    "Generalized anxiety disorder",
    "Chronic obstructive pulmonary disease",
    "Chronic kidney disease",
    "Cirrhosis",
    "Osteoporosis",
    "Influenza",
]

_HEADING = re.compile(r"^=+\s*(.+?)\s*=+$", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]+")


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.strip().lower())
    return slug.strip("_")


def clean_extract(text: str) -> str:
    """Normalize an extract: heading markers become plain lines, blanks collapse."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HEADING.sub(r"\1", text)
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    return _BLANKS.sub("\n\n", cleaned).strip()


def truncate(text: str, max_chars: int) -> str:
    """Cut at a paragraph boundary so an article never ends mid-sentence."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < max_chars // 2:
        cut = text.rfind(". ", 0, max_chars)
        cut = cut + 1 if cut > 0 else max_chars
    return text[:cut].rstrip()


def parse_pages(payload: dict) -> list[dict[str, str]]:
    """Extract usable pages from an API response, skipping missing ones."""
    pages = (payload.get("query") or {}).get("pages") or []
    if isinstance(pages, dict):  # older formatversion=1 shape
        pages = list(pages.values())
    result: list[dict[str, str]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        if page.get("missing") or page.get("invalid"):
            continue
        title = str(page.get("title") or "").strip()
        extract = page.get("extract")
        if not title or not isinstance(extract, str) or not extract.strip():
            continue
        result.append({"title": title, "extract": extract, "pageid": str(page.get("pageid") or "")})
    return result


def fetch_extracts(
    titles: Iterable[str],
    *,
    pause: float = 2.0,
    retries: int = 4,
    timeout: float = 30.0,
) -> list[dict[str, str]]:
    """Fetch plain-text extracts, one request per title, with bounded retries.

    The API caps *full* article extracts at one per request (it answers an
    ``exlimit`` of 20 with a warning and lowers it), so batching titles does not
    work here. ``exintro`` would allow batches, at the cost of only the lead
    section; not worth it for a corpus that needs real paragraphs.
    """
    wanted = [title for title in titles if title.strip()]
    pages: list[dict[str, str]] = []
    for index, title in enumerate(wanted):
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
            "titles": title,
        }
        url = f"{API}?{urllib.parse.urlencode(params)}"
        try:
            payload = _get_json(url, retries=retries, timeout=timeout)
        except RuntimeError as exc:
            # One dead title should not cost the whole corpus; the manifest records
            # exactly what made it.
            print(f"  failed {title!r}: {exc}", file=sys.stderr)
            continue
        found = parse_pages(payload)
        if not found:
            print(f"  skipped {title!r} (no extract)", file=sys.stderr)
        pages.extend(found)
        if index + 1 < len(wanted):
            time.sleep(pause)
    return pages


def _get_json(url: str, *, retries: int, timeout: float) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if attempt >= retries:
                break
            if exc.code == 429:
                # Wikipedia asks for backoff; a token bucket is not worth writing.
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 10.0
                time.sleep(delay * attempt)
            else:
                time.sleep(2.0 * attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last_error}")


def build_articles(
    topics: Iterable[str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    **fetch_kwargs: object,
) -> list[dict[str, object]]:
    topics = list(topics)
    articles: list[dict[str, object]] = []
    for page in fetch_extracts(topics, **fetch_kwargs):  # type: ignore[arg-type]
        text = truncate(clean_extract(page["extract"]), max_chars)
        if len(text) < 200:
            continue
        articles.append(
            {
                "id": slugify(page["title"]),
                "title": page["title"],
                "url": "https://en.wikipedia.org/wiki/"
                + urllib.parse.quote(page["title"].replace(" ", "_")),
                "source": SOURCE,
                "text": text,
            }
        )
    return articles


def write_corpus(
    articles: Iterable[dict[str, object]],
    out_dir: Path,
    *,
    merge: bool = False,
) -> Path:
    """Write one file per article plus a manifest describing exactly what was used."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    fresh = list(articles)
    if merge:
        manifest_path = out_dir / MANIFEST_NAME
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            new_ids = {str(article["id"]) for article in fresh}
            entries.extend(
                entry
                for entry in previous.get("articles", [])
                if entry.get("id") not in new_ids and (out_dir / str(entry["file"])).exists()
            )
    for article in fresh:
        text = str(article["text"])
        path = out_dir / f"{article['id']}.txt"
        path.write_text(text, encoding="utf-8")
        entries.append(
            {
                "id": article["id"],
                "title": article["title"],
                "url": article["url"],
                "source": article["source"],
                "file": path.name,
                "chars": len(text),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    entries.sort(key=lambda entry: str(entry["id"]))
    manifest = {
        "source": SOURCE,
        "api": API,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "max_chars": DEFAULT_MAX_CHARS,
        "count": len(entries),
        "total_chars": sum(int(entry["chars"]) for entry in entries),
        "articles": entries,
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the medical test corpus.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    parser.add_argument("--limit", type=int, default=len(TOPICS), help="how many topics")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--force", action="store_true", help="refetch even if present")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between requests")
    args = parser.parse_args(argv)

    manifest_path = args.out / MANIFEST_NAME
    present: dict[str, dict] = {}
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        present = {
            str(entry["id"]): entry
            for entry in previous.get("articles", [])
            if (args.out / str(entry["file"])).exists()
        }

    topics = TOPICS[: max(0, args.limit)]
    pending = [topic for topic in topics if slugify(topic) not in present]
    if not pending:
        ready = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"corpus already present: {ready['count']} articles, "
            f"{ready['total_chars']} chars in {args.out} (use --force to refetch)"
        )
        return 0

    print(f"fetching {len(pending)} of {len(topics)} topics from {SOURCE} ...")
    articles = build_articles(pending, max_chars=args.max_chars, pause=args.pause)
    if not articles:
        print("no articles fetched (see warnings above)", file=sys.stderr)
        return 1
    written = write_corpus(articles, args.out, merge=not args.force)
    total = sum(len(str(article["text"])) for article in articles)
    print(f"wrote {len(articles)} articles ({total} chars) to {args.out}")
    print(f"manifest: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
