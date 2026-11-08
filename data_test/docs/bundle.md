# The data_test bundle

`data_test/` is a reproducible, resumable bundle for a larger Medical GraphRAG
demonstration. It keeps generated data out of the repository and can publish the
finished index as a public Hugging Face dataset.

## Layout

```
data_test/
  auto_pipeline.py        one-command orchestrator
  prepare_environment.py  venv, pip, Ollama, .env
  setup_ollama.py         Ollama serve + model pulls + GPU settings
  build_index.py          extract → graph → resolve → communities → embed
  verify_bundle.py        corpus and artifact checks
  run_demo.py             ask a question against the bundle
  download_dataset.py     restore a published HF dataset
  embedding_server.py     optional local sentence-transformers /v1/embeddings
  start_local_servers.py  optional GGUF + embedding servers instead of Ollama
  plan_data.md            design, commands, risks
  docs/                   this documentation
```

Generated data (git-ignored):

```
data_test/corpus/         *.txt articles + manifest.json
data_test/artifacts/      index.json, triples.jsonl, graph.json,
                          communities.json, vectors.json, audit files
data_test/models/         optional GGUF weights
data_test/embeddings/     optional small embedding model
```

## Flow

```text
prepare_environment
  → setup_ollama (qwen3-4b-instruct-2507 + nomic-embed-text)
  → corpus fetch/resume (>= 50 articles)
  → extract → graph → resolve → communities → embed
  → verify_bundle
  → upload medical-graphrag-data-test
  → run_demo
```

`extract` is the slow, paid stage. It is checkpointed to
`artifacts/triples.jsonl`; completed chunks are skipped on resume. `--retry-failed`
re-asks only chunks recorded in `failed_chunks.json`.

## Models

- Chat: `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`, aliased to
  `qwen3-4b-instruct-2507`.
- Embeddings: `nomic-embed-text` (768 dimensions) through Ollama.
- Optional: `BAAI/bge-small-en-v1.5` (384 dimensions) through
  `embedding_server.py`.

Ollama settings used by `setup_ollama.py`:

```
OLLAMA_SCHED_SPREAD=1
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_CONTEXT_LENGTH=32768
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=-1
```

A 4B Q4 model fits on one T4, so Ollama may keep it on a single GPU. That is
normal; forcing a split is not automatically faster.

## Verification

```bash
.venv/bin/python data_test/verify_bundle.py
```

Checks: at least 50 corpus articles, `manifest.json`, `index.json`, `graph.json`,
`communities.json`, `vectors.json`, `triples.jsonl`, and vector backend metadata.

## Publish and restore

The pipeline creates or updates the public dataset
`<hf-username>/medical-graphrag-data-test` when `HF_TOKEN` is available. An
existing dataset is reused.

Restore on another machine:

```bash
HF_REPO_ID=yourname/medical-graphrag-data-test \
HF_TOKEN=hf_xxx \
python data_test/download_dataset.py
```

Then verify and ask:

```bash
python data_test/prepare_environment.py
.venv/bin/python data_test/verify_bundle.py
.venv/bin/python data_test/run_demo.py "What is asthma?"
```

## Boundaries

The corpus is Wikipedia medicine, not a clinical reference. Answers must cite
chunks and abstain when evidence is missing.
