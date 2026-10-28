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
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    # Kaggle sets UV_SYSTEM_PYTHON=true, which makes `uv venv` warn and can make
    # `uv pip` ignore the venv we are explicitly targeting.
    env.pop("UV_SYSTEM_PYTHON", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=clean_env())


def uv_binary() -> str | None:
    return shutil.which("uv")


def create_venv() -> None:
    if VENV_PYTHON.is_file():
        print("Virtual environment already exists.", flush=True)
    else:
        # Prefer uv if available. --seed installs pip/setuptools/wheel.
        try:
            subprocess.check_call(
                ["uv", "venv", "--seed", str(ROOT / ".venv")],
                cwd=str(ROOT),
                env=clean_env(),
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    ensure_pip()
    bootstrap_wrapt()


def ensure_pip() -> None:
    """Make sure the venv has pip, even if it was created by a bare `uv venv`."""
    uv = uv_binary()
    if uv:
        # uv installs into the venv without starting its interpreter, so Kaggle's
        # system sitecustomize cannot fail before pip runs.
        subprocess.call(
            [uv, "pip", "install", "--python", str(VENV_PYTHON), "-q", "pip"],
            cwd=str(ROOT),
            env=clean_env(),
        )
        return
    if subprocess.call(
        [str(VENV_PYTHON), "-m", "pip", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=clean_env(),
    ) == 0:
        return
    print("pip missing in venv; bootstrapping...", flush=True)
    subprocess.call([str(VENV_PYTHON), "-m", "ensurepip", "--upgrade"], env=clean_env())


def install_packages(args: list[str]) -> int:
    """Install into the venv, preferring uv so the venv interpreter is not started."""
    uv = uv_binary()
    if uv:
        cmd = [uv, "pip", "install", "--python", str(VENV_PYTHON), *args]
    else:
        cmd = [str(VENV_PYTHON), "-m", "pip", "install", *args]
    return run(cmd)


def bootstrap_wrapt() -> None:
    """Install wrapt before any venv Python call.

    Kaggle's system ``sitecustomize`` imports wrapt; when it is missing, every
    venv interpreter prints ``ModuleNotFoundError: No module named 'wrapt'``
    before running the real command. uv installs it without starting Python.
    """
    if install_packages(["-q", "wrapt"]) != 0:
        print("warning: could not preinstall wrapt (sitecustomize may be noisy)", flush=True)


def install_deps() -> None:
    install_packages(["-q", "-r", "requirements.txt"])
    install_packages(["-q", "-U", "huggingface_hub", "wrapt"])
    # The default Ollama path does not use sentence-transformers or llama-cpp:
    # installing them pulls torch / compiles C++ and was the slow step on Kaggle.
    if os.environ.get("USE_LOCAL_MODELS", "").strip() != "1":
        return
    install_packages(["-q", "-U", "sentence-transformers", "fastapi", "uvicorn"])
    # llama-cpp-python may need a wheel; try, but do not fail the whole setup if it errors
    if install_packages(["-q", "-U", "llama-cpp-python"]) != 0:
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
    defaults = [
        f'GRAPHRAG_CORPUS={DATA_TEST / "corpus"}',
        f'GRAPHRAG_ARTIFACTS={DATA_TEST / "artifacts"}',
        "LLM_BASE_URL=http://127.0.0.1:11434/v1",
        "LLM_API_KEY=ollama",
        "LLM_MODEL=qwen3-4b-instruct-2507",
        "LLM_MAX_TOKENS=4096",
        "HARNESS_CONTEXT_WINDOW=50000",
        "GRAPHRAG_EMBED_BASE_URL=http://127.0.0.1:11434/v1",
        "GRAPHRAG_EMBED_MODEL=nomic-embed-text",
        "GRAPHRAG_EMBED_API_KEY=ollama",
        "GRAPHRAG_EMBED_DIM=768",
    ]
    token = get_hf_token()
    if token:
        defaults.append(f"HF_TOKEN={token}")

    existing = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    # Upgrade the old default model name if it is still there.
    upgraded = [
        "LLM_MODEL=qwen3-4b-instruct-2507" if line.strip() == "LLM_MODEL=qwen3:4b" else line
        for line in existing
    ]
    model_upgraded = upgraded != existing
    existing = upgraded

    present = {
        line.partition("=")[0].strip()
        for line in existing
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }
    added = [line for line in defaults if line.partition("=")[0].strip() not in present]
    if not added and not model_upgraded:
        print(f".env already has all managed keys at {env_path}.", flush=True)
        return
    # Merge rather than clobber: keep every user value that is already there.
    env_path.write_text("\n".join(existing + added) + "\n", encoding="utf-8")
    details = []
    if added:
        details.append("added: " + ", ".join(line.partition("=")[0] for line in added))
    if model_upgraded:
        details.append("model=qwen3-4b-instruct-2507")
    print(f"Updated {env_path} ({'; '.join(details)})", flush=True)


def main() -> int:
    create_venv()
    install_deps()
    write_env()
    if os.environ.get("USE_OLLAMA", "").strip() == "1":
        print("USE_OLLAMA=1: setting up Ollama...", flush=True)
        rc = subprocess.call(
            [str(VENV_PYTHON), str(DATA_TEST / "setup_ollama.py")],
            cwd=str(ROOT),
            env=clean_env(),
        )
        if rc != 0:
            print("Ollama setup failed; environment marker not written.", flush=True)
            return rc
    (DATA_TEST / ".prepared").write_text("ok\n", encoding="utf-8")
    print("Environment prepared.")
    print(f"Next: {VENV_PYTHON} data_test/auto_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
