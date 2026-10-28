"""Run a question through the GraphRAG using the data_test bundle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or "What is asthma?"

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("UV_SYSTEM_PYTHON", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("GRAPHRAG_CORPUS", str(DATA_TEST / "corpus"))
    env.setdefault("GRAPHRAG_ARTIFACTS", str(DATA_TEST / "artifacts"))

    cmd = [str(PYTHON), "-m", "graphrag.ask", "--mode", "auto", question]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
