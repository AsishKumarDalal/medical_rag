"""Make the three-article mini corpus used by the live G1/G4 run.

    .venv/bin/python test/eval/make_mini_corpus.py

The full corpus is fetched by ``test/data/test_data_fetch.py`` and git-ignored.
This script deliberately copies only the articles the live run extracted, so the
index and the evaluation snapshot describe the same 40 chunks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "test" / "data" / "corpus"
DEFAULT_TARGET = ROOT / "test" / "data" / "corpus_mini"
DEFAULT_IDS = ("alzheimer_s_disease", "asthma", "breast_cancer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--ids", nargs="+", default=list(DEFAULT_IDS))
    args = parser.parse_args()

    source_manifest = args.source / "manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    wanted = set(args.ids)
    entries = [entry for entry in payload.get("articles", []) if entry.get("id") in wanted]
    found = {entry["id"] for entry in entries}
    missing = wanted - found
    if missing:
        raise SystemExit(f"missing from {source_manifest}: {', '.join(sorted(missing))}")

    args.target.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        shutil.copyfile(args.source / entry["file"], args.target / entry["file"])

    manifest = dict(payload)
    manifest["articles"] = entries
    manifest["count"] = len(entries)
    manifest["total_chars"] = sum(int(entry.get("chars", 0)) for entry in entries)
    manifest["derived_from"] = str(source_manifest)
    (args.target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(entries)} articles, {manifest['total_chars']} chars to {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
