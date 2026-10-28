"""Prepare the local environment for the data_test pipeline.

Run this once with system Python:
    python data_test/prepare_environment.py

It will:
  - create .venv if missing
  - install requirements.txt + huggingface_hub
  - write a .env file with data_test paths and (optional) HF token
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def create_venv() -> None:
    if VENV_PYTHON.is_file():
        print("Virtual environment already exists.", flush=True)
        ensure_pip()
        return
    # Prefer uv if available. --seed installs pip/setuptools/wheel.
    try:
        subprocess.check_call(["uv", "venv", "--seed", str(ROOT / ".venv")], cwd=str(ROOT))
        ensure_pip()
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    ensure_pip()


def ensure_pip() -> None:
    """Make sure the venv has pip, even if it was created by a bare `uv venv`."""
    if subprocess.call([str(VENV_PYTHON), "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return
    print("pip missing in venv; bootstrapping...", flush=True)
    # Try uv first (does not need pip inside the venv)
    try:
        subprocess.check_call(["uv", "pip", "install", "--python", str(VENV_PYTHON), "pip"], cwd=str(ROOT))
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Fallback to ensurepip
    subprocess.call([str(VENV_PYTHON), "-m", "ensurepip", "--upgrade"])


def install_deps() -> None:
    pip = [str(VENV_PYTHON), "-m", "pip"]
    run(pip + ["install", "-q", "-r", "requirements.txt"])
    run(
        pip
        + [
            "install",
            "-q",
            "-U",
            "huggingface_hub",
            "sentence-transformers",
            "fastapi",
            "uvicorn",
            "wrapt",
        ]
    )
    # llama-cpp-python may need a wheel; try, but do not fail the whole setup if it errors
    try:
        run(pip + ["install", "-q", "-U", "llama-cpp-python"])
    except Exception:
        print("llama-cpp-python install failed; chat server may be unavailable.", flush=True)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # Kaggle secrets
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    return ""


def write_env() -> None:
    env_path = ROOT / ".env"
    lines = [
        f'GRAPHRAG_CORPUS={DATA_TEST / "corpus"}',
        f'GRAPHRAG_ARTIFACTS={DATA_TEST / "artifacts"}',
        "LLM_BASE_URL=http://127.0.0.1:11434/v1",
        "LLM_MODEL=qwen3:4b",
        "LLM_MAX_TOKENS=4096",
        "HARNESS_CONTEXT_WINDOW=50000",
        "GRAPHRAG_EMBED_BASE_URL=http://127.0.0.1:11434/v1",
        "GRAPHRAG_EMBED_MODEL=nomic-embed-text",
        "GRAPHRAG_EMBED_API_KEY=ollama",
        "GRAPHRAG_EMBED_DIM=768",
    ]
    token = get_hf_token()
    if token:
        lines.append(f"HF_TOKEN={token}")
    # Only write if not already present, to avoid clobbering user settings
    if not env_path.is_file():
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote {env_path}", flush=True)
    else:
        print(f".env already exists at {env_path}; leaving it unchanged.", flush=True)


def main() -> int:
    create_venv()
    install_deps()
    write_env()
    if os.environ.get("USE_OLLAMA", "").strip() == "1":
        print("USE_OLLAMA=1: setting up Ollama...", flush=True)
        rc = subprocess.call([str(VENV_PYTHON), str(DATA_TEST / "setup_ollama.py")], cwd=str(ROOT))
        if rc != 0:
            print("Ollama setup failed; environment marker not written.", flush=True)
            return rc
    (DATA_TEST / ".prepared").write_text("ok\n", encoding="utf-8")
    print("Environment prepared.")
    print(f"Next: {VENV_PYTHON} data_test/auto_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
