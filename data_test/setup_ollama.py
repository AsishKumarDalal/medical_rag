"""Ensure Ollama is running with both GPUs and the required models are pulled.

Usage:
    .venv/bin/python data_test/setup_ollama.py
    OLLAMA_RESTART=1 .venv/bin/python data_test/setup_ollama.py

Requires the `ollama` binary. On Kaggle/Linux:
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

DATA_TEST = Path(__file__).resolve().parent
LOG_PATH = DATA_TEST / "ollama_server.log"

# Exact Qwen3-4B-Instruct-2507 GGUF from Unsloth, served through Ollama.
CHAT_MODEL = os.environ.get(
    "OLLAMA_CHAT_MODEL",
    "hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M",
)
MODEL_ALIAS = os.environ.get("OLLAMA_MODEL_ALIAS", "qwen3-4b-instruct-2507")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Use both GPUs: spread layers instead of filling one GPU first.
# A 4B Q4 model fits on one T4, so Ollama may still choose a single GPU; that is
# usually faster than a split. Set OLLAMA_SCHED_SPREAD=0 to disable spreading.
OLLAMA_ENV = {
    "OLLAMA_HOST": "127.0.0.1:11434",
    "OLLAMA_CONTEXT_LENGTH": os.environ.get("OLLAMA_CONTEXT_LENGTH", "32768"),
    "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_KV_CACHE_TYPE": os.environ.get("OLLAMA_KV_CACHE_TYPE", "q8_0"),
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
    "OLLAMA_KEEP_ALIVE": "-1",
    "OLLAMA_SCHED_SPREAD": os.environ.get("OLLAMA_SCHED_SPREAD", "1"),
    "OLLAMA_ORIGINS": "*",
}


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
    target_dir = Path.home() / ".local" / "bin"
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


def stop_ollama() -> None:
    subprocess.run(["pkill", "-9", "-x", "ollama"], check=False)
    # ollama serve may leave its llama-server child holding the GPUs.
    subprocess.run(["pkill", "-9", "-x", "llama-server"], check=False)
    time.sleep(2)


def start_serve() -> subprocess.Popen | None:
    binary = ollama_binary()
    if not binary:
        print("Ollama binary not found. Install it first.", file=sys.stderr)
        return None
    if is_running():
        print("Ollama already running.", flush=True)
        return None
    print("Starting `ollama serve` with GPU settings...", flush=True)
    env = {**os.environ, **OLLAMA_ENV}
    log = open(LOG_PATH, "a")
    proc = subprocess.Popen(
        [binary, "serve"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if is_running():
            print("Ollama is ready.", flush=True)
            return proc
        time.sleep(2)
    print(f"Ollama did not become ready; see {LOG_PATH}", file=sys.stderr)
    return proc


def _client_env() -> dict[str, str]:
    return {**os.environ, "OLLAMA_HOST": OLLAMA_ENV["OLLAMA_HOST"]}


def pull(model: str) -> int:
    binary = ollama_binary()
    if not binary:
        return 1
    print(f"Pulling {model} ...", flush=True)
    return subprocess.call([binary, "pull", model], env=_client_env())


def alias(src: str, dst: str) -> int:
    binary = ollama_binary()
    if not binary:
        return 1
    print(f"Aliasing {src} -> {dst} ...", flush=True)
    return subprocess.call([binary, "cp", src, dst], env=_client_env())


def main() -> int:
    if not ollama_binary():
        print("Ollama is not installed.", file=sys.stderr)
        print("Install with: curl -fsSL https://ollama.com/install.sh | sh", file=sys.stderr)
        return 1

    if os.environ.get("OLLAMA_RESTART", "").strip() == "1":
        print("OLLAMA_RESTART=1: restarting Ollama to apply GPU settings.", flush=True)
        stop_ollama()

    proc = start_serve()

    rc1 = pull(CHAT_MODEL)
    if rc1 != 0:
        print(f"Could not pull {CHAT_MODEL}; falling back to qwen3:4b.", file=sys.stderr)
        rc1 = pull("qwen3:4b")
        if rc1 == 0:
            rc1 = alias("qwen3:4b", MODEL_ALIAS)

    rc2 = pull(EMBED_MODEL)
    if rc1 != 0 or rc2 != 0:
        print("Model pull failed.", file=sys.stderr)
        return 1

    if CHAT_MODEL != MODEL_ALIAS:
        alias(CHAT_MODEL, MODEL_ALIAS)

    print("Ollama setup complete.")
    print(f"Chat model: {CHAT_MODEL} (alias: {MODEL_ALIAS})")
    print(f"Embed model: {EMBED_MODEL}")
    print("GPU settings:", ", ".join(f"{k}={v}" for k, v in OLLAMA_ENV.items() if k.startswith("OLLAMA_SCHED") or k.startswith("OLLAMA_KV")))
    if proc is not None:
        print("Ollama server is running in the background.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
