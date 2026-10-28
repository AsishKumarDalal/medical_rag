"""Ensure Ollama is running and required models are pulled.

Usage:
    .venv/bin/python data_test/setup_ollama.py

Requires the `ollama` binary to be installed. On Kaggle/Linux:
    curl -fsSL https://ollama.com/install.sh | sh
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3:4b")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _install_zstd() -> None:
    if shutil.which("zstd"):
        return
    print("zstd missing; trying to install it...", flush=True)
    for cmd in (
        "sudo apt-get update -qq && sudo apt-get install -y -qq zstd",
        "apt-get update -qq && apt-get install -y -qq zstd",
    ):
        try:
            subprocess.run(cmd, shell=True, check=True)
            return
        except Exception:
            pass


def _install_binary() -> bool:
    """Fallback: download the standalone binary into ~/.local/bin (no root needed)."""
    home = Path.home()
    target_dir = home / ".local" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "ollama"
    url = "https://ollama.com/download/ollama-linux-amd64"
    print(f"Downloading Ollama binary from {url} ...", flush=True)
    try:
        subprocess.run(["curl", "-fsSL", url, "-o", str(target)], check=True)
        target.chmod(0o755)
    except Exception as exc:
        print(f"Binary download failed: {exc}", file=sys.stderr)
        return False
    os.environ["PATH"] = str(target_dir) + os.pathsep + os.environ.get("PATH", "")
    return shutil.which("ollama") is not None


def install_ollama() -> bool:
    if shutil.which("ollama"):
        return True
    print("Ollama not found; attempting install...", flush=True)
    _install_zstd()
    try:
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    except Exception as exc:
        print(f"Ollama installer failed: {exc}", file=sys.stderr)
        print("Trying standalone binary fallback...", flush=True)
        return _install_binary()
    return shutil.which("ollama") is not None


def ollama_binary() -> str | None:
    if not shutil.which("ollama"):
        install_ollama()
    return shutil.which("ollama")


def is_running() -> bool:
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_serve() -> subprocess.Popen | None:
    binary = ollama_binary()
    if not binary:
        print("Ollama binary not found. Install it first.", file=sys.stderr)
        return None
    if is_running():
        print("Ollama already running.", flush=True)
        return None
    print("Starting `ollama serve`...", flush=True)
    proc = subprocess.Popen([binary, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # wait
    deadline = time.time() + 60
    while time.time() < deadline:
        if is_running():
            print("Ollama is ready.", flush=True)
            return proc
        time.sleep(2)
    print("Ollama did not become ready.", file=sys.stderr)
    return proc


def pull(model: str) -> int:
    binary = ollama_binary()
    if not binary:
        return 1
    print(f"Pulling {model} ...", flush=True)
    return subprocess.call([binary, "pull", model])


def main() -> int:
    if not ollama_binary():
        print("Ollama is not installed.", file=sys.stderr)
        print("Install with: curl -fsSL https://ollama.com/install.sh | sh", file=sys.stderr)
        return 1

    proc = start_serve()
    rc1 = pull(CHAT_MODEL)
    rc2 = pull(EMBED_MODEL)
    if rc1 != 0 or rc2 != 0:
        print("Model pull failed.", file=sys.stderr)
        return 1

    print("Ollama setup complete.")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embed model: {EMBED_MODEL}")
    if proc is not None:
        print("Ollama server is running in the background.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
