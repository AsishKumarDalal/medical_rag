"""One-command, resumable pipeline for the data_test bundle.

Runs on Kaggle or locally:
  1. Restore the latest HF mirror so a restarted session resumes from there.
  2. Check/install prerequisite Python modules.
  3. Download the local LLM (Unsloth GGUF) if missing.
  4. Download the small embedding model if missing.
  5. Resume/complete the medical corpus to at least --min-articles.
  6. Run the GraphRAG index pipeline (extract -> graph -> resolve -> communities -> embed).
  7. Upload a crash-safe mirror to HF immediately, after each phase, and every
     --checkpoint-seconds.
  8. Verify the bundle and upload the final bundle.
  9. Run a demo question.

Usage:
    .venv/bin/python data_test/auto_pipeline.py
    .venv/bin/python data_test/auto_pipeline.py --min-articles 60 --retry-failed
    .venv/bin/python data_test/auto_pipeline.py --skip-models --skip-build
    .venv/bin/python data_test/auto_pipeline.py --skip-restore
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_TEST = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)

FETCH_SCRIPT = ROOT / "test" / "data" / "test_data_fetch.py"

PHASES = ("extract", "graph", "resolve", "communities", "embed")
RESTORE_PATTERNS = ["corpus/*", "artifacts/*"]
STAGE_ORDER = {
    "index": 0,
    "corpus": 1,
    "extract": 2,
    "graph": 3,
    "resolve": 4,
    "communities": 5,
    "embed": 6,
    "complete": 6,
}

UPLOAD_IGNORE = [
    ".prepared",
    "__pycache__/*",
    "*.py",
    "*.pyc",
    "*.log",
    "models/*",
    "embeddings/*",
    "*.gguf",
]


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    # Kaggle injects a system sitecustomize via PYTHONPATH that can break venv
    # interpreters; the venv does not need it.
    env.pop("PYTHONPATH", None)
    env.pop("UV_SYSTEM_PYTHON", None)
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
        subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-q", pkg], env=clean_env())


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


def build_index(
    retry_failed: bool,
    workers: int,
    force: bool,
    *,
    on_phase: Callable[[str, int], None] | None = None,
) -> int:
    env = clean_env()
    env["GRAPHRAG_CORPUS"] = str(DATA_TEST / "corpus")
    env["GRAPHRAG_ARTIFACTS"] = str(DATA_TEST / "artifacts")
    # Optional: if an embedding endpoint is provided, keep it; otherwise hash fallback.

    for phase in PHASES:
        cmd = [
            str(PYTHON),
            str(DATA_TEST / "build_index.py"),
            "--stage",
            phase,
            "--workers",
            str(workers),
        ]
        if phase == "extract":
            if retry_failed:
                cmd.append("--retry-failed")
            if force:
                cmd.append("--force")
        elif phase == "embed" and force:
            cmd.append("--force")
        rc = run(cmd, env=env)
        if on_phase is not None:
            try:
                on_phase(phase, rc)
            except Exception as exc:
                print(
                    f"[checkpoint] phase mirror failed ({type(exc).__name__}: {exc})",
                    file=sys.stderr,
                    flush=True,
                )
        if rc != 0:
            return rc
    return 0


def verify() -> int:
    return run([str(PYTHON), str(DATA_TEST / "verify_bundle.py")])


def demo() -> int:
    env = clean_env()
    env["GRAPHRAG_CORPUS"] = str(DATA_TEST / "corpus")
    env["GRAPHRAG_ARTIFACTS"] = str(DATA_TEST / "artifacts")
    return run([str(PYTHON), str(DATA_TEST / "run_demo.py"), "What is asthma?"], env=env)


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _jsonl_counts(path: Path) -> tuple[int, int]:
    records = triples = 0
    if not path.is_file():
        return records, triples
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                records += 1
                values = record.get("triples")
                if isinstance(values, list):
                    triples += len(values)
    except OSError:
        pass
    return records, triples


def collect_progress(data_dir: Path | None = None, started_at: float | None = None) -> dict:
    data_dir = data_dir or DATA_TEST
    artifacts = data_dir / "artifacts"
    index = _read_json(artifacts / "index.json")
    graph = _read_json(artifacts / "graph.json")
    communities = _read_json(artifacts / "communities.json")
    vectors = _read_json(artifacts / "vectors.json")
    failed_payload = _read_json(artifacts / "failed_chunks.json")
    records, triples = _jsonl_counts(artifacts / "triples.jsonl")

    failed = _as_int(failed_payload.get("count"))
    if not failed and isinstance(failed_payload.get("chunks"), list):
        failed = len(failed_payload["chunks"])
    total = _as_int(index.get("chunk_count"))
    done = records + failed

    nodes_value = graph.get("nodes") or {}
    edges_value = graph.get("edges") or []
    nodes = len(nodes_value) if isinstance(nodes_value, (dict, list)) else 0
    edges = len(edges_value) if isinstance(edges_value, list) else 0

    community_count = _as_int(communities.get("count"))
    community_list = communities.get("communities")
    if not community_count and isinstance(community_list, (list, dict)):
        community_count = len(community_list)

    if vectors:
        stage = "embed"
    elif communities:
        stage = "communities"
    elif (artifacts / "merge_log.json").is_file():
        stage = "resolve"
    elif graph:
        stage = "graph"
    elif records or failed:
        stage = "extract"
    else:
        stage = "index"

    vector_counts = vectors.get("counts")
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "article_count": _as_int(index.get("article_count")),
        "chunks_total": total,
        "chunks_extracted": records,
        "chunks_failed": failed,
        "chunks_done": done,
        "percent": round(100.0 * done / total, 1) if total else 0.0,
        "triples": triples,
        "nodes": nodes,
        "edges": edges,
        "communities": community_count,
        "vector_backend": str(vectors.get("backend", "")) if vectors else "",
        "vectors": vector_counts if isinstance(vector_counts, dict) else {},
        "elapsed_seconds": int(time.time() - started_at) if started_at else 0,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        return (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
    except Exception:
        return ""


def resolve_repo_id(repo_name: str | None, token: str | None) -> str:
    ensure_module("huggingface_hub")
    from huggingface_hub import HfApi

    name = (repo_name or os.environ.get("HF_REPO_ID", "")).strip() or "medical-graphrag-data-test"
    if "/" in name:
        return name
    if not token:
        return ""
    username = HfApi(token=token).whoami()["name"]
    return f"{username}/{name}"


def _progress_state(data_dir: Path) -> dict:
    progress = collect_progress(data_dir)
    stored = _read_json(data_dir / "artifacts" / "progress.json")
    stored_stage = str(stored.get("stage") or "")
    if stored_stage == "complete" and progress["stage"] == "embed":
        progress["stage"] = "complete"
    elif stored_stage == "corpus" and progress["stage"] == "index":
        progress["stage"] = "corpus"
    return progress


def _progress_rank(progress: dict) -> tuple[int, int, int]:
    stage = str(progress.get("stage") or "index")
    return (
        STAGE_ORDER.get(stage, 0),
        _as_int(progress.get("chunks_done")),
        _as_int(progress.get("article_count")),
    )


def restore_dataset(
    repo_name: str | None = None,
    *,
    data_dir: Path | None = None,
    token: str | None = None,
) -> int:
    """Restore the newest HF mirror so a new session resumes instead of restarting."""
    data_dir = data_dir or DATA_TEST
    token = get_hf_token() if token is None else token
    repo_id = resolve_repo_id(repo_name, token)
    if not repo_id:
        print(
            "[restore] skipped: set --repo-id or HF_REPO_ID to locate the HF mirror.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    ensure_module("huggingface_hub")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    print(f"[restore] checking HF mirror: {repo_id}", flush=True)
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token or None,
            allow_patterns=RESTORE_PATTERNS,
        )
    except RepositoryNotFoundError:
        print("[restore] no mirror found; starting from local state.", flush=True)
        return 0
    except Exception as exc:
        print(
            f"[restore] download failed ({type(exc).__name__}: {exc}); "
            "continuing with local state.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    staging = Path(snapshot_path)
    if not (staging / "corpus").is_dir() and not (staging / "artifacts").is_dir():
        print("[restore] mirror has no corpus or artifacts yet; keeping local state.", flush=True)
        return 0

    remote = _progress_state(staging)
    local = _progress_state(data_dir)
    local_has_data = (data_dir / "corpus" / "manifest.json").is_file() or (
        data_dir / "artifacts" / "index.json"
    ).is_file()
    if _progress_rank(remote) <= _progress_rank(local) and local_has_data:
        print(
            f"[restore] local state is current ({local['stage']} "
            f"{local['chunks_done']}/{local['chunks_total']}); keeping it.",
            flush=True,
        )
        return 0

    restored = 0
    for name in ("corpus", "artifacts"):
        entry = staging / name
        if not entry.is_dir():
            continue
        shutil.copytree(entry, data_dir / name, dirs_exist_ok=True)
        restored += 1

    print(
        f"[restore] restored {restored} data folders from HF "
        f"({remote['stage']} {remote['chunks_done']}/{remote['chunks_total']}, "
        f"triples={remote['triples']}).",
        flush=True,
    )
    return 0


def upload_dataset(
    repo_name: str | None,
    *,
    data_dir: Path | None = None,
    token: str | None = None,
    commit_message: str | None = None,
) -> int:
    """Create/update a public HF dataset from a data bundle."""
    token = token or get_hf_token()
    if not token:
        print("Skipping upload: HF_TOKEN is not set (env or Kaggle secret).", file=sys.stderr)
        return 0

    ensure_module("huggingface_hub")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_id = resolve_repo_id(repo_name, token)

    print(f"Using Hugging Face dataset: {repo_id}", flush=True)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=str(data_dir or DATA_TEST),
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=UPLOAD_IGNORE,
            commit_message=commit_message,
        )
    except Exception as exc:
        print(f"Upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Uploaded: https://huggingface.co/datasets/{repo_id}", flush=True)
    return 0


def push_dataset(repo_name: str | None) -> int:
    """Create/update the public dataset with the data_test bundle."""
    return upload_dataset(repo_name, data_dir=DATA_TEST)


_CHECKPOINT_LOCK = threading.Lock()


def checkpoint_once(
    repo_name: str | None,
    data_dir: Path,
    started_at: float,
    token: str,
    *,
    stage: str | None = None,
) -> dict:
    with _CHECKPOINT_LOCK:
        progress = collect_progress(data_dir, started_at)
        if stage:
            progress["stage"] = stage
        write_json_atomic(data_dir / "artifacts" / "progress.json", progress)
        print(
            f"[checkpoint] {progress['stage']} "
            f"{progress['chunks_extracted']}/{progress['chunks_total']} extracted "
            f"({progress['percent']}% attempted), failed={progress['chunks_failed']}, "
            f"triples={progress['triples']}, elapsed={progress['elapsed_seconds']}s",
            flush=True,
        )
        rc = upload_dataset(
            repo_name,
            data_dir=data_dir,
            token=token,
            commit_message=f"checkpoint {progress['updated_at']}",
        )
        if rc != 0:
            print("[checkpoint] upload failed; will retry next interval.", file=sys.stderr, flush=True)
        return progress


def checkpoint_loop(
    stop_event: threading.Event,
    repo_name: str | None,
    interval: int,
    data_dir: Path,
    started_at: float,
    token: str,
) -> None:
    while not stop_event.is_set():
        try:
            checkpoint_once(repo_name, data_dir, started_at, token)
        except Exception as exc:
            print(f"[checkpoint] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if stop_event.wait(interval):
            break


def start_checkpoint_thread(
    repo_name: str | None,
    interval: int,
    data_dir: Path,
    started_at: float,
    token: str,
) -> tuple[threading.Event, threading.Thread] | None:
    if interval <= 0:
        return None
    if not token:
        print("[checkpoint] disabled: HF_TOKEN is not set.", file=sys.stderr, flush=True)
        return None
    stop_event = threading.Event()
    thread = threading.Thread(
        target=checkpoint_loop,
        args=(stop_event, repo_name, interval, data_dir, started_at, token),
        name="hf-checkpoint",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def stop_checkpoint_thread(handle: tuple[threading.Event, threading.Thread] | None) -> None:
    if handle is None:
        return
    stop_event, thread = handle
    stop_event.set()
    thread.join(timeout=120)
    if thread.is_alive():
        print("[checkpoint] upload thread is still running; continuing.", file=sys.stderr, flush=True)


MARKER = DATA_TEST / ".prepared"


def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def ensure_environment() -> None:
    os.environ.setdefault("USE_OLLAMA", "1")
    venv_python = ROOT / ".venv" / "bin" / "python"
    needs_prepare = not MARKER.is_file() or not venv_python.is_file()
    if os.environ.get("USE_OLLAMA", "").strip() == "1" and not shutil.which("ollama"):
        needs_prepare = True
    if needs_prepare:
        py = venv_python if venv_python.is_file() else Path(sys.executable)
        print("Preparing environment...", flush=True)
        subprocess.check_call([str(py), str(DATA_TEST / "prepare_environment.py")], env=clean_env())
    # Make sure the rest of the pipeline runs under the venv, even when the
    # script was started with the system interpreter.
    in_venv = Path(sys.prefix).resolve() == (ROOT / ".venv").resolve()
    if venv_python.is_file() and not in_venv:
        os.execv(
            str(venv_python),
            [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        )


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def main() -> int:
    load_env()
    ensure_environment()
    load_env()  # pick up keys that prepare_environment just wrote

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
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Do not restore the latest HF mirror before running.",
    )
    parser.add_argument("--repo-id", default=None, help="Hugging Face dataset repo, e.g. user/medical-graphrag-data-test")
    parser.add_argument(
        "--checkpoint-seconds",
        type=int,
        default=env_int("HF_CHECKPOINT_SECONDS", 3600),
        help="Upload an HF snapshot every N seconds while indexing (0 disables).",
    )
    parser.add_argument(
        "--checkpoint-repo",
        default=os.environ.get("HF_CHECKPOINT_REPO") or None,
        help="Dataset repo for hourly snapshots (default: same as --repo-id).",
    )
    args = parser.parse_args()

    token = get_hf_token()
    mirror_repo = args.checkpoint_repo or args.repo_id
    allow_push = not args.skip_push and bool(token)

    if not args.skip_restore:
        restore_dataset(mirror_repo, data_dir=DATA_TEST, token=token)

    started_at = time.time()

    # 1.5 Ollama (optional)
    setup_ollama()

    # 2. models
    if not args.skip_models and os.environ.get("USE_OLLAMA", "").strip() != "1":
        ensure_module("huggingface_hub")
        download_model()
        download_embed_model()

    # 3. corpus
    if not args.skip_corpus:
        ensure_corpus(args.min_articles)
        if allow_push:
            checkpoint_once(mirror_repo, DATA_TEST, started_at, token, stage="corpus")

    server_proc = start_local_servers()
    checkpoint = None

    def mirror_phase(phase: str, rc: int) -> None:
        if not allow_push:
            return
        checkpoint_once(
            mirror_repo,
            DATA_TEST,
            started_at,
            token,
            stage=phase if rc == 0 else None,
        )

    try:
        if not args.skip_push:
            checkpoint = start_checkpoint_thread(
                mirror_repo,
                args.checkpoint_seconds,
                DATA_TEST,
                started_at,
                token,
            )

        # 4. build
        if not args.skip_build:
            rc = build_index(
                args.retry_failed,
                args.workers,
                args.force,
                on_phase=mirror_phase,
            )
            if rc != 0:
                print("Build failed.", file=sys.stderr)
                return rc

        stop_checkpoint_thread(checkpoint)
        checkpoint = None

        if allow_push:
            checkpoint_once(mirror_repo, DATA_TEST, started_at, token)

        # 5. verify
        rc = verify()
        if rc != 0:
            print("Verification failed.", file=sys.stderr)
            return rc

        # 6. upload
        if not args.skip_push:
            with _CHECKPOINT_LOCK:
                progress = collect_progress(DATA_TEST, started_at)
                progress["stage"] = "complete"
                write_json_atomic(DATA_TEST / "artifacts" / "progress.json", progress)
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
        stop_checkpoint_thread(checkpoint)
        stop_local_servers(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
