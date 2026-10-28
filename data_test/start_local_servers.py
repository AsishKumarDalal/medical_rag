"""Start local OpenAI-compatible servers for chat (GGUF) and embeddings (BGE).

Usage:
    .venv/bin/python data_test/start_local_servers.py
    # keeps running until Ctrl-C
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("UV_SYSTEM_PYTHON", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def wait_ready(url: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-port", type=int, default=8080)
    parser.add_argument("--embed-port", type=int, default=8081)
    parser.add_argument("--model", default=str(DATA_TEST / "models" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))
    parser.add_argument("--embed-model", default=str(DATA_TEST / "embeddings" / "bge-small-en-v1.5"))
    args = parser.parse_args()

    children: list[subprocess.Popen] = []

    def cleanup(*_):
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Chat server (llama-cpp-python)
    if not Path(args.model).is_file():
        print(f"Chat model not found: {args.model}", file=sys.stderr)
        return 1
    try:
        chat_proc = subprocess.Popen(
            [str(PYTHON), "-m", "llama_cpp.server", "--model", args.model, "--host", "127.0.0.1", "--port", str(args.chat_port)],
            cwd=str(ROOT),
            env=clean_env(),
        )
        children.append(chat_proc)
        print(f"Started chat server on port {args.chat_port}", flush=True)
    except FileNotFoundError:
        print("llama-cpp-python not installed; install it to serve the GGUF.", file=sys.stderr)
        return 1

    # Embedding server
    if not Path(args.embed_model).is_dir():
        print(f"Embedding model not found: {args.embed_model}", file=sys.stderr)
        return 1
    embed_proc = subprocess.Popen(
        [str(PYTHON), str(DATA_TEST / "embedding_server.py"), "--model", args.embed_model, "--port", str(args.embed_port)],
        cwd=str(ROOT),
        env=clean_env(),
    )
    children.append(embed_proc)
    print(f"Started embedding server on port {args.embed_port}", flush=True)

    chat_ready = wait_ready(f"http://127.0.0.1:{args.chat_port}/v1/models", 180)
    embed_ready = wait_ready(f"http://127.0.0.1:{args.embed_port}/v1/models", 180)
    print(f"Chat ready: {chat_ready}; Embed ready: {embed_ready}", flush=True)

    # Keep running
    while True:
        time.sleep(5)
        for p in children:
            if p.poll() is not None:
                print("A server exited; shutting down.", file=sys.stderr)
                cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
