"""Verify that the data_test bundle is complete enough to answer questions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_TEST = Path(__file__).resolve().parent
CORPUS = DATA_TEST / "corpus"
ARTIFACTS = DATA_TEST / "artifacts"


def fail(msg: str) -> int:
    print("FAIL:", msg)
    return 1


def main() -> int:
    if not CORPUS.is_dir():
        return fail("missing corpus/")
    manifest_path = CORPUS / "manifest.json"
    if not manifest_path.is_file():
        return fail("missing corpus/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    count = int(manifest.get("count", 0))
    if count < 50:
        return fail(f"corpus has {count} articles, expected at least 50")
    txts = list(CORPUS.glob("*.txt"))
    if len(txts) < 50:
        return fail(f"only {len(txts)} .txt files found")

    required = ["index.json", "graph.json", "communities.json", "vectors.json", "triples.jsonl"]
    for name in required:
        if not (ARTIFACTS / name).is_file():
            return fail(f"missing artifacts/{name}")

    try:
        vectors = json.loads((ARTIFACTS / "vectors.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return fail(f"invalid vectors.json: {exc}")
    if "backend" not in vectors:
        return fail("vectors.json missing backend metadata")

    print(f"OK: corpus={count} articles, artifacts present, backend={vectors.get('backend')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
