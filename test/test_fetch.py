"""G0: the corpus fetch is testable without the network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from test_data_fetch import (  # noqa: E402
    clean_extract,
    parse_pages,
    slugify,
    truncate,
    write_corpus,
)

FIXTURE = Path(__file__).parent / "data" / "fixtures" / "wiki_extract.json"


@pytest.fixture
def payload():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parse_pages_skips_missing_articles(payload):
    pages = parse_pages(payload)

    assert [page["title"] for page in pages] == ["Asthma", "Malaria"]
    assert pages[0]["pageid"] == "12345"


def test_parse_pages_skips_empty_extracts():
    payload = {"query": {"pages": [{"title": "Empty", "extract": "   "}]}}

    assert parse_pages(payload) == []


def test_clean_extract_flattens_headings_and_blank_runs(payload):
    raw = parse_pages(payload)[0]["extract"]

    cleaned = clean_extract(raw)

    assert "== Signs and symptoms ==" not in cleaned
    assert "Signs and symptoms" in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.endswith("Symptoms are usually worse at night.")


def test_slugify_makes_stable_file_names():
    assert slugify("Alzheimer's disease") == "alzheimer_s_disease"
    assert slugify("HIV/AIDS") == "hiv_aids"
    assert slugify("  Chronic kidney disease  ") == "chronic_kidney_disease"


def test_truncate_cuts_on_a_paragraph_boundary():
    text = "first paragraph. " * 20 + "\n\n" + "second paragraph. " * 20

    cut = truncate(text, 400)

    assert len(cut) <= 400
    assert cut.rstrip().endswith("first paragraph.")


def test_truncate_leaves_short_text_alone():
    assert truncate("short", 100) == "short"


def test_write_corpus_writes_files_and_a_manifest(tmp_path):
    articles = [
        {
            "id": "asthma",
            "title": "Asthma",
            "url": "https://en.wikipedia.org/wiki/Asthma",
            "source": "en.wikipedia.org",
            "text": "Asthma is a disease.",
        }
    ]

    manifest_path = write_corpus(articles, tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == 1
    assert manifest["articles"][0]["chars"] == len("Asthma is a disease.")
    stored = (tmp_path / "asthma.txt").read_text(encoding="utf-8")
    assert stored == "Asthma is a disease."
    assert manifest["articles"][0]["sha256"] == hashlib.sha256(stored.encode()).hexdigest()


def test_every_topic_has_a_plausible_file_name():
    from test_data_fetch import TOPICS

    slugs = [slugify(topic) for topic in TOPICS]

    assert 20 <= len(TOPICS) <= 30
    assert len(set(slugs)) == len(slugs)
    assert all(slug and slug[0].isalnum() for slug in slugs)
