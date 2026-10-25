"""Ask the indexed corpus one question.

    python -m graphrag.ask "What is asthma?"
    python -m graphrag.ask --mode global --k 4 "What are the main themes?"
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .config import GraphRagConfig, GraphRagConfigError
from .query import GraphRagNotReady, answer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the medical GraphRAG index.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--mode", choices=["auto", "local", "global", "abstain"], default="auto")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    try:
        result = answer(question, mode=args.mode, k=args.k)
    except GraphRagConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    except GraphRagNotReady as exc:
        print(f"[index] {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print(result.answer)
    if result.citations:
        print("\nCitations: " + ", ".join(result.citations))
    print(f"(mode: {result.mode})")
    if result.error:
        print(f"[warning] {result.error}", file=sys.stderr)
    return 0 if not result.error else 1


if __name__ == "__main__":
    sys.exit(main())
