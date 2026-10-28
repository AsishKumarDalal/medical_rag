"""Build the GraphRAG index for the data_test bundle.

Usage:
    .venv/bin/python data_test/build_index.py
    .venv/bin/python data_test/build_index.py --limit 10
    .venv/bin/python data_test/build_index.py --retry-failed
    .venv/bin/python data_test/build_index.py --stage extract
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

STAGES = ("extract", "graph", "resolve", "communities", "embed")


def env() -> dict[str, str]:
    e = dict(os.environ)
    # Keep Kaggle's system sitecustomize out of the venv interpreter.
    e.pop("PYTHONPATH", None)
    e.pop("UV_SYSTEM_PYTHON", None)
    e["PYTHONNOUSERSITE"] = "1"
    e.setdefault("GRAPHRAG_CORPUS", str(DATA_TEST / "corpus"))
    e.setdefault("GRAPHRAG_ARTIFACTS", str(DATA_TEST / "artifacts"))
    return e


def run(args: list[str]) -> int:
    print("+ " + " ".join(args), flush=True)
    return subprocess.call(args, cwd=str(ROOT), env=env())


def stage_command(stage: str, args: argparse.Namespace) -> list[str]:
    cmd = [str(PYTHON), "-m", "graphrag.build", "--stage", stage]
    if stage == "extract":
        cmd += ["--workers", str(args.workers)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.retry_failed:
            cmd += ["--retry-failed"]
        if args.force:
            cmd += ["--force"]
    elif stage == "embed" and args.force:
        cmd.append("--force")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    artifacts = Path(env()["GRAPHRAG_ARTIFACTS"])
    corpus = Path(env()["GRAPHRAG_CORPUS"])
    artifacts.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if run(stage_command(stage, args)) != 0:
            return 1

    print("Build complete." if len(stages) > 1 else f"Stage {args.stage} complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
