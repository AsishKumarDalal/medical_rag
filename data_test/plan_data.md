# Plan: data_test/ — Offline Medical GraphRAG Data + Local LLM

Goal: make a reproducible, shareable data bundle so anyone can download it from
Hugging Face and run the Medical GraphRAG without rebuilding the expensive index.

This plan is documentation only. No code is added here; commands are provided for
manual execution.

---

## 0. What this bundle contains

- `data_test/models/` — local LLM weights (Unsloth GGUF).
- `data_test/embeddings/` — optional small embedding model (e.g. BGE small).
- `data_test/corpus/` — at least 50 medical articles + `manifest.json`.
- `data_test/artifacts/` — built GraphRAG index:
  - `index.json`
  - `triples.jsonl`
  - `graph.json`
  - `merge_log.json`, `merge_audit.jsonl`
  - `communities.json`
  - `vectors.json`
- `data_test/plan_data.md` — this file.

The Hugging Face dataset will be public and will store the same tree under its
repo root, so downloading it restores the same folder layout.

---

## 1. Prerequisites

- Linux machine with internet access.
- Python venv (`.venv`) with project dependencies installed.
- `HF_TOKEN` with write access to create/push the dataset.
- Kaggle account with internet enabled (for the heavy build).
- Enough disk space: ~6 GB for the model + ~1 GB for corpus/artifacts.

Install the Hugging Face CLI if missing:
```bash
pip install -U huggingface_hub
huggingface-cli login
```

---

## 2. Step A — Download the local LLM (Unsloth GGUF)

Repository: `unsloth/Qwen3-4B-Instruct-2507-GGUF`

Pick one quantization. Recommended for CPU/GPU balance:
- `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` (~2.5 GB)

Download into `data_test/models/`:
```bash
mkdir -p data_test/models
huggingface-cli download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  --include "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" \
  --local-dir data_test/models
```

Notes:
- If you want smaller/faster, use `Q4_K_S` or `Q3_K_M`.
- If you want higher quality, use `Q6_K` or `Q8_0`.
- Record the chosen filename in the dataset README so users know which file to load.

---

## 2.5 Step A2 — Optional small embedding model (recommended)

Instead of the deterministic hash embedder, use a small local embedding model for
better retrieval. Recommended small models:

- `BAAI/bge-small-en-v1.5` — 33.4M params, 384 dim, MIT, strong for its size.
- `sentence-transformers/all-MiniLM-L6-v2` — 22.7M params, 384 dim, Apache 2.0, fastest.
- `NeuML/pubmedbert-base-embeddings` — 110M params, 768 dim, medical-specific (larger).

For a small, fast bundle, use `bge-small-en-v1.5` or `all-MiniLM-L6-v2`.

### How to serve it (OpenAI-compatible)
The project only talks to OpenAI-compatible `/embeddings` endpoints. Run a small
server, for example with Text Embeddings Inference (TEI) or llama.cpp embedding
mode, then set:
```bash
export GRAPHRAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
export GRAPHRAG_EMBED_BASE_URL=http://127.0.0.1:8080/v1
export GRAPHRAG_EMBED_API_KEY=local
export GRAPHRAG_EMBED_DIM=384
```

If you do not serve an embedding model, the pipeline automatically falls back to
the deterministic hash embedder and records `backend="hash-ngram"` in
`vectors.json`. That is acceptable for demos, but a small real embedding model
gives better retrieval.

### Download the embedding model (optional, for local serving)
```bash
mkdir -p data_test/embeddings
huggingface-cli download BAAI/bge-small-en-v1.5 \
  --local-dir data_test/embeddings/bge-small-en-v1.5
```

---

## 3. Step B — Build a corpus of at least 50 medical articles

Use the existing fetch script, but point `--out` at `data_test/corpus`.

Option 1: extend the topic list to 50+ medical topics, then run:
```bash
mkdir -p data_test/corpus
.venv/bin/python test/data/test_data_fetch.py \
  --out data_test/corpus \
  --pause 2
```

Option 2: run multiple batches and merge:
```bash
.venv/bin/python test/data/test_data_fetch.py --out data_test/corpus --limit 24 --pause 2
# add more topics to a custom list, then:
.venv/bin/python test/data/test_data_fetch.py --out data_test/corpus --limit 26 --pause 2
```

Requirements:
- At least 50 `.txt` articles.
- `manifest.json` present with count, total chars, sha256 per article.
- Verify:
```bash
ls data_test/corpus/*.txt | wc -l
cat data_test/corpus/manifest.json | python -c "import json,sys; d=json.load(sys.stdin); print('count=',d['count'],'chars=',d['total_chars'])"
```

---

## 4. Step C — Use Kaggle to build the index (heavy stage)

Kaggle is used because extraction is slow and needs a stable GPU/CPU runtime.

### 4.1 Kaggle setup
- Create a Kaggle notebook.
- Add your `HF_TOKEN` as a Kaggle Secret named `HF_TOKEN`.
- Enable internet.
- Upload or clone this repository into the notebook.
- Set environment variables:
```bash
export GRAPHRAG_CORPUS=/kaggle/working/data_test/corpus
export GRAPHRAG_ARTIFACTS=/kaggle/working/data_test/artifacts
export LLM_BASE_URL=<your local or remote endpoint>
export LLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
export LLM_MAX_TOKENS=4096
# optional small embedding model:
export GRAPHRAG_EMBED_MODEL=BAAI/bge-small-en-v1.5
export GRAPHRAG_EMBED_BASE_URL=http://127.0.0.1:8080/v1
export GRAPHRAG_EMBED_API_KEY=local
export GRAPHRAG_EMBED_DIM=384
export HF_TOKEN=<from Kaggle secrets>
```

### 4.2 Resume / track progress
The pipeline is checkpointed. `auto_pipeline.py` restores the newest HF mirror's
`corpus/` and `artifacts/` before it fetches anything or starts the build, so a
restarted session resumes from the mirrored corpus and extraction checkpoint
instead of restarting from an empty local folder. Code files stay local, and the
local bundle is kept when it is ahead of the mirror. A snapshot is uploaded after
corpus, extract, graph, resolve, communities and embed, immediately when the
build finishes, and every `HF_CHECKPOINT_SECONDS` during long extraction, so a
run that finishes before the hourly interval is still uploaded. Use
`--skip-restore` only for a deliberate local-only run.

Track progress by:
- `wc -l data_test/corpus/*.txt` (corpus size)
- `wc -l data_test/artifacts/triples.jsonl` (extracted chunks)
- `cat data_test/artifacts/failed_chunks.json` (failed chunks)
- `index.json` chunk count vs triples count.

### 4.3 Build commands (run in order)
```bash
# extraction (slowest, resumable)
.venv/bin/python -m graphrag.build --stage extract --workers 1

# if endpoint died:
.venv/bin/python -m graphrag.build --stage extract --retry-failed

# graph
.venv/bin/python -m graphrag.build --stage graph

# resolve entities
.venv/bin/python -m graphrag.build --stage resolve

# communities
.venv/bin/python -m graphrag.build --stage communities

# embeddings
.venv/bin/python -m graphrag.build --stage embed
```

For a faster demo subset, you can use `--limit N` on extract, but for the final
dataset run the full corpus.

---

## 5. Step D — Push artifacts to a public Hugging Face dataset

Dataset repo name suggestion: `<your-username>/medical-graphrag-data-test`

Create the repo (if not exists) and upload:
```bash
huggingface-cli repo create <your-username>/medical-graphrag-data-test --type dataset --public

# upload corpus + artifacts + model metadata
huggingface-cli upload <your-username>/medical-graphrag-data-test \
  data_test/ . \
  --repo-type dataset
```

Important:
- Do NOT upload `.env`, secrets or the `.py` helper scripts; the mirror is data
  only (`corpus/` and `artifacts/`).
- Include a `README.md` in the dataset describing:
  - corpus size (articles, chars, chunks)
  - model file name
  - backend used for embeddings (hash vs openai-compatible)
  - how to download and run
- The dataset should allow “resume” by checking `manifest.json` version and
  `triples.jsonl` line count.

---

## 6. Step E — Download script for end users (manual command)

End users should be able to fetch the data into the correct folders with:
```bash
# download entire dataset into data_test/
huggingface-cli download <your-username>/medical-graphrag-data-test \
  --repo-type dataset \
  --local-dir data_test
```

Then run:
```bash
export GRAPHRAG_CORPUS=data_test/corpus
export GRAPHRAG_ARTIFACTS=data_test/artifacts
.venv/bin/python -m graphrag.ask --mode auto "What is asthma?"
.venv/bin/python -m harness --once "What is asthma?"
```

Optional: if the user wants to use the local GGUF model, point their inference
server at `data_test/models/<gguf-file>`.

---

## 7. Versioning and resume rules

- `manifest.json` is the source of truth for corpus identity.
- `triples.jsonl` is append-only; line count = extracted chunks.
- `failed_chunks.json` lists failed chunk IDs.
- `vectors.json` fingerprints ensure unchanged text reuses vectors.
- If a dataset version already exists, `auto_pipeline.py` restores it on startup.
  To resume by hand:
  1. Download it (`download_dataset.py`).
  2. Compare manifest count to desired corpus.
  3. If corpus changed, re-run fetch with merge.
  4. Re-run extract with `--retry-failed`.
  5. Rebuild graph/resolve/communities/embed.

---

## 8. Minimal success criteria

- `data_test/corpus/` has >= 50 articles + valid manifest.
- `data_test/artifacts/` has `index.json`, `graph.json`, `communities.json`,
  `vectors.json`, and `triples.jsonl`.
- `graphrag.ask` returns a cited answer using the downloaded data.
- Dataset is public on Hugging Face and downloadable with `huggingface-cli`.

---

## 9. Code in data_test/

The following helper scripts live in `data_test/`:

- `download_dataset.py` — download the public HF dataset bundle into `data_test/`.
- `build_index.py` — run the full indexing pipeline against `data_test/corpus` and `data_test/artifacts`.
- `verify_bundle.py` — check corpus count and required artifacts.
- `run_demo.py` — ask a question using the `data_test` bundle.
- `auto_pipeline.py` — one-command resumable pipeline: prerequisites, model downloads, corpus to >=50, index build, verify, demo.
- `prepare_environment.py` — create venv, install deps, write `.env` with data_test paths.
- `setup_ollama.py` — install/check Ollama and pull chat + embedding models.

## 10. Risks and notes:

- Extraction is the bottleneck; keep `workers=1` for tunnel stability.
- If the endpoint dies, use `--retry-failed`, not a full reset.
- Do not upload secrets or `.env`.
- Large GGUF files may exceed Kaggle/HF limits; consider uploading the model as
  a separate repo if needed.
- Hash embeddings are fine for demos; for better retrieval, configure a real
  embedding endpoint before the final push.