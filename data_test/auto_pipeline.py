"""One-command, resumable pipeline for the data_test bundle.

Runs on Kaggle or locally:
  1. Check/install prerequisite Python modules.
  2. Download the local LLM (Unsloth GGUF) if missing.
  3. Download the small embedding model if missing.
  4. Resume/complete the medical corpus to at least --min-articles.
  5. Run the GraphRAG index pipeline (extract -> graph -> resolve -> communities -> embed).
  6. Verify the bundle.
  7. Run a demo question.

Usage:
    .venv/bin/python data_test/auto_pipeline.py
    .venv/bin/python data_test/auto_pipeline.py --min-articles 60 --retry-failed
    .venv/bin/python data_test/auto_pipeline.py --skip-models --skip-build
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

FETCH_SCRIPT = ROOT / "test" / "data" / "test_data_fetch.py"


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    # Kaggle injects a system sitecustomize via PYTHONPATH that can break venv
    # interpreters; the venv does not need it.
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run(cmd: list[str], env: dict[str, str] | None = None) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT), env=env or clean_env())


def ensure_module(name: str, package: str | None = None) -> None:
    try:
        __import__(name)
        return
    except ImportError:
        pkg = package or name
        print(f"Installing missing module: {pkg}", flush=True)
        subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-q", pkg])


def download_file_hf(repo_id: str, filename: str, local_dir: Path) -> None:
    ensure_module("huggingface_hub")
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id}/{filename} -> {local_dir}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download_model() -> None:
    model_dir = DATA_TEST / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    if target.is_file():
        print(f"Model already present: {target}", flush=True)
        return
    download_file_hf(
        "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        model_dir,
    )


def download_embed_model() -> None:
    embed_dir = DATA_TEST / "embeddings" / "bge-small-en-v1.5"
    if (embed_dir / "config.json").is_file():
        print(f"Embedding model already present: {embed_dir}", flush=True)
        return
    ensure_module("huggingface_hub")
    from huggingface_hub import snapshot_download

    embed_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BAAI/bge-small-en-v1.5 -> {embed_dir}", flush=True)
    snapshot_download(
        repo_id="BAAI/bge-small-en-v1.5",
        local_dir=str(embed_dir),
        resume_download=True,
    )


EXTRA_TOPICS = [
    "Acute myocardial infarction",
    "Type 2 diabetes mellitus",
    "Essential hypertension",
    "Community-acquired pneumonia",
    "Pulmonary tuberculosis",
    "Plasmodium falciparum malaria",
    "Human immunodeficiency virus",
    "Sepsis syndrome",
    "Invasive breast carcinoma",
    "Non-small cell lung cancer",
    "Colorectal adenocarcinoma",
    "Alzheimer disease",
    "Parkinson disease",
    "Temporal lobe epilepsy",
    "Migraine disorder",
    "Major depressive disorder",
    "Generalized anxiety disorder",
    "Chronic obstructive pulmonary disease",
    "Chronic kidney disease",
    "Liver cirrhosis",
    "Postmenopausal osteoporosis",
    "Seasonal influenza",
    "Acute stroke",
    "Asthma bronchiale",
    "Rheumatoid arthritis",
    "Osteoarthritis",
    "Systemic lupus erythematosus",
    "Multiple sclerosis",
    "Amyotrophic lateral sclerosis",
    "Schizophrenia",
    "Bipolar disorder",
    "Attention deficit hyperactivity disorder",
    "Autism spectrum disorder",
    "Obsessive compulsive disorder",
    "Post traumatic stress disorder",
    "Anorexia nervosa",
    "Bulimia nervosa",
    "Substance use disorder",
    "Alcohol use disorder",
    "Tobacco use disorder",
    "Obesity",
    "Metabolic syndrome",
    "Hyperlipidemia",
    "Gout",
    "Osteoporosis",
    "Osteopenia",
    "Vitamin D deficiency",
    "Iron deficiency anemia",
    "Sickle cell disease",
    "Thalassemia",
]


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("test_data_fetch", FETCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {FETCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_count() -> int:
    manifest = DATA_TEST / "corpus" / "manifest.json"
    if not manifest.is_file():
        return 0
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return int(data.get("count", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def ensure_corpus(min_articles: int) -> None:
    fetch = load_fetch_module()
    corpus_dir = DATA_TEST / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    manifest_path = corpus_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_ids = {str(e.get("id")) for e in prev.get("articles", [])}
        except json.JSONDecodeError:
            existing_ids = set()

    desired_topics = list(fetch.TOPICS) + EXTRA_TOPICS
    missing = [t for t in desired_topics if fetch.slugify(t) not in existing_ids]

    current = corpus_count()
    print(f"Corpus currently has {current} articles (target {min_articles})", flush=True)

    if current >= min_articles:
        print("Corpus target already reached.", flush=True)
        return

    # Fetch only what is missing, in batches, until target is reached.
    batch: list[str] = []
    for topic in missing:
        batch.append(topic)
        if len(batch) >= 10:
            print(f"Fetching batch of {len(batch)} topics...", flush=True)
            articles = fetch.build_articles(batch, max_chars=12000, pause=2.0, retries=4, timeout=30.0)
            fetch.write_corpus(articles, corpus_dir, merge=True)
            batch = []
            if corpus_count() >= min_articles:
                break

    if batch:
        print(f"Fetching final batch of {len(batch)} topics...", flush=True)
        articles = fetch.build_articles(batch, max_chars=12000, pause=2.0, retries=4, timeout=30.0)
        fetch.write_corpus(articles, corpus_dir, merge=True)

    print(f"Corpus now has {corpus_count()} articles.", flush=True)


def setup_ollama() -> None:
    if os.environ.get("USE_OLLAMA", "").strip() != "1":
        return
    print("Setting up Ollama...", flush=True)
    rc = subprocess.call([str(PYTHON), str(DATA_TEST / "setup_ollama.py")], cwd=str(ROOT), env=clean_env())
    if rc != 0:
        print("Ollama setup failed.", file=sys.stderr)
        raise SystemExit(rc)


def start_local_servers() -> subprocess.Popen | None:
    if os.environ.get("USE_LOCAL_MODELS", "").strip() != "1":
        return None
    print("Starting local model servers...", flush=True)
    proc = subprocess.Popen(
        [str(PYTHON), str(DATA_TEST / "start_local_servers.py")],
        cwd=str(ROOT),
    )
    # Wait for readiness
    import time
    from urllib.request import urlopen

    for url in ("http://127.0.0.1:8080/v1/models", "http://127.0.0.1:8081/v1/models"):
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                with urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
    return proc


def stop_local_servers(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def build_index(retry_failed: bool, workers: int, force: bool) -> int:
    env = clean_env()
    env["GRAPHRAG_CORPUS"] = str(DATA_TEST / "corpus")
    env["GRAPHRAG_ARTIFACTS"] = str(DATA_TEST / "artifacts")
    # Optional: if an embedding endpoint is provided, keep it; otherwise hash fallback.

    cmd = [str(PYTHON), str(DATA_TEST / "build_index.py")]
    if retry_failed:
        cmd.append("--retry-failed")
    if force:
        cmd.append("--force")
    cmd += ["--workers", str(workers)]
    return run(cmd, env=env)


def verify() -> int:
    return run([str(PYTHON), str(DATA_TEST / "verify_bundle.py")])


def demo() -> int:
    env = clean_env()
    env["GRAPHRAG_CORPUS"] = str(DATA_TEST / "corpus")
    env["GRAPHRAG_ARTIFACTS"] = str(DATA_TEST / "artifacts")
    return run([str(PYTHON), str(DATA_TEST / "run_demo.py"), "What is asthma?"], env=env)


def push_dataset(repo_id: str | None) -> int:
    repo_id = (repo_id or os.environ.get("HF_REPO_ID", "")).strip()
    if not repo_id:
        print("Skipping upload: set HF_REPO_ID or pass --repo-id to push the dataset.", flush=True)
        return 0
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Skipping upload: HF_TOKEN is not set.", file=sys.stderr)
        return 1

    ensure_module("huggingface_hub")
    from huggingface_hub import HfApi

    print(f"Uploading data_test/ to Hugging Face dataset {repo_id} ...", flush=True)
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(DATA_TEST),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=[".prepared", "__pycache__/*", "*.pyc"],
    )
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}", flush=True)
    return 0


MARKER = DATA_TEST / ".prepared"


def ensure_environment() -> None:
    os.environ.setdefault("USE_OLLAMA", "1")
    venv_python = ROOT / ".venv" / "bin" / "python"
    needs_prepare = not MARKER.is_file()
    if os.environ.get("USE_OLLAMA", "").strip() == "1" and not shutil.which("ollama"):
        needs_prepare = True
    if needs_prepare:
        py = venv_python if venv_python.is_file() else Path(sys.executable)
        print("Preparing environment...", flush=True)
        subprocess.check_call([str(py), str(DATA_TEST / "prepare_environment.py")], env=clean_env())
        if not venv_python.is_file():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def main() -> int:
    ensure_environment()

    # Load .env if present
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if sep and key not in os.environ:
                os.environ[key] = value

    parser = argparse.ArgumentParser()
    parser.add_argument("--min-articles", type=int, default=50)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-corpus", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--repo-id", default=None, help="Hugging Face dataset repo, e.g. user/medical-graphrag-data-test")
    args = parser.parse_args()

    # 1. prerequisites
    ensure_module("huggingface_hub")
    ensure_module("requests", "requests")

    # 1.5 Ollama (optional)
    setup_ollama()

    # 2. models
    if not args.skip_models and os.environ.get("USE_OLLAMA", "").strip() != "1":
        download_model()
        download_embed_model()

    # 3. corpus
    if not args.skip_corpus:
        ensure_corpus(args.min_articles)

    server_proc = start_local_servers()

    try:
        # 4. build
        if not args.skip_build:
            rc = build_index(args.retry_failed, args.workers, args.force)
            if rc != 0:
                print("Build failed.", file=sys.stderr)
                return rc

        # 5. verify
        rc = verify()
        if rc != 0:
            print("Verification failed.", file=sys.stderr)
            return rc

        # 6. upload
        if not args.skip_push:
            rc = push_dataset(args.repo_id)
            if rc != 0:
                print("Upload failed.", file=sys.stderr)
                return rc

        # 7. demo
        if not args.skip_demo:
            demo()

        print("Auto pipeline complete.")
        return 0
    finally:
        stop_local_servers(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
