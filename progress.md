# Progress

Living status of this repo. Updated when a phase or meaningful feature completes.
Task protocol and detailed learnings live in `TASK.md` and `logs-and-fixes.md`.

- **Last updated:** 2026-11-08
- **Current phase:** G1 live indexing on Kaggle via `data_test/auto_pipeline.py`
  (52 articles, 725 chunks); extraction stopped at **69/725** to add crash-safe
  HF checkpoints.
- **Next:** monitor the resumed 725-chunk build and its hourly snapshots, then
  the final verify/push/demo; followed by live G2 queries, G3 tool round-trip and
  the G4 evaluation run.

## Data_test Kaggle run (2026-10-28)

- Stopped the full run at **69 chunks / 1,067 triples**; artifacts are in
  `data_test/artifacts/` and resume without `--force`.
- `auto_pipeline.py` now writes `artifacts/progress.json` and uploads a snapshot
  immediately and every `HF_CHECKPOINT_SECONDS` (default **3600**) while
  indexing. It stops the uploader before verify/final push and excludes
  `models/`, `embeddings/`, logs and `.prepared`.
- Checkpoint + upload path tested against a scratch HF repo (`progress.json`
  round-trip, SHA match), then the scratch repo was deleted. Offline suite:
  **140 passed**.
- PR #2 merged to `main` (`fd979d1`); resumed without `--force` at 69/725.
  First snapshot is live at
  `https://huggingface.co/datasets/sddfhy546y/medical-graphrag-data-test`
  (67 files, 69/725, 1,067 triples, 9.5%, public). Next upload is one hour after
  the 19:21 UTC start; the final push follows verify/demo.

## Documentation (2026-11-08)

- Added [`data_test/docs/`](data_test/docs/): bundle overview and Kaggle run guide.
- Updated `TASK.md` with the completed data_test automation block and refreshed
  `logs-and-fixes.md` with the Kaggle venv, `sitecustomize`, Ollama and HF upload
  fixes.

## Phases

| Phase | Scope | Status |
|---|---|---|
| **H1** | Agent loop: config, messages, LLM client, tool registry, loop | ✅ done |
| **H2** | Tool validation, result caps, RAG tool stub, tests | ✅ done |
| **H3** | Context budget: 45% tool-output stubbing, 75% LLM compression | ✅ done |
| **H4** | SQLite sessions: schema, store, resume | ✅ done |
| **H5** | Security: allowlist, budgets, injection wrapping, redaction | ✅ done |
| **H6** | Minimal CLI REPL + scripted run | ✅ done |
| **G0** | Corpus fetch script + 24 articles + offline tests | ✅ done |
| **G1** | Indexing pipeline (I1–I11) + live run | 🔄 code/tests done; live run via `data_test/` (52 articles, 725 chunks) in progress |
| **Data test** | Kaggle/Ollama bundle: corpus, index, models, HF upload | 🔄 automation done; live build/upload in progress |
| **G2** | Query pipeline (Q0–QG) + live smoke queries | 🔄 code/tests done; live pending |
| **G3** | Harness integration — RAG as the tool, end-to-end | 🔄 code/tests done; live transcript pending |
| **G4** | Evaluation set, run script, `docs/eval.md` | 🔄 questions/script done; run pending |

## What exists now

```
harness/            H1–H6 agent harness
  docs/             one record per phase: h1.md … h6.md
  config.py         env settings; enforces the 50k prompt+output window
  messages.py       typed messages/tool calls, provider conversion
  llm.py            single provider doorway: retries, normalization, redaction
  loop.py           tool-calling loop; never raises for model/tool failures
  context.py        H3 token budget, stubbing and compression
  session.py        H4 SQLite persistence and resume
  security.py       H5 allowlist, bounds, untrusted-text wrapping, redaction
  cli.py            H6 REPL; wires the real RAG backend when configured
  tools/            registry + echo + real search_medical_literature
tui/                minimal curses UI; background worker, transcript, input history
graphrag/           G0–G4 pipeline
  docs/             one record per phase: g0.md … g4.md
  corpus/chunk/     I1–I2 deterministic 1,000-char corpus chunks
  extract/          I3–I4 one-call-per-chunk extraction + JSONL checkpoint
  validate/normalize/graph  I5–I8 validation, identity, aggregation
  resolve/          I9 candidate nomination + LLM judge + audit
  communities/      I10 weighted label propagation + summaries
  embed/            I11 vectors (OpenAI-compatible or hash fallback)
  query/rerank/     Q0–QG local/global retrieval + citations
  backend/ask       harness adapter and `python -m graphrag.ask`
test/               pytest suite and evaluation
  docs/             live/offline run records
  data/             G0 fetch script + committed fixture + ignored corpus
  eval/             G4 questions, runner, results
```

## Data test bundle (`data_test/`)

A reproducible Kaggle/local bundle for a larger demonstration corpus:

```
data_test/
  auto_pipeline.py        one-command resumable pipeline
  prepare_environment.py  venv, pip, Ollama, .env
  setup_ollama.py         start Ollama, pull Qwen3-4B + embeddings
  build_index.py          extract → graph → resolve → communities → embed
  verify_bundle.py        corpus/artifact checks
  run_demo.py             ask a question against the bundle
  download_dataset.py     fetch the public HF dataset
  plan_data.md            detailed design and run plan
  docs/                   bundle and Kaggle run guides
```

Current live run: **52 articles, 725 chunks**, using local Ollama
`qwen3-4b-instruct-2507` (Unsloth Q4_K_M) and `nomic-embed-text`. The pipeline
auto-creates or updates the public dataset `medical-graphrag-data-test` on Hugging
Face when `HF_TOKEN` is available.

## Verification status

- **Offline suite:** see the latest `pytest -q` in `logs-and-fixes.md`; the
  GraphRAG tests use stub LLMs and deterministic embeddings (no network).
- **Live H1–H6:** loop smoke (`test/docs/smoke-h1.md`), context demo
  (`test/docs/h3-context.md`), CLI run (`test/docs/h6-cli.md`).
- **TUI:** `python -m tui` smoke-tested in a PTY (startup, `/help`, `/exit`);
  `test/test_tui.py` covers structured events and transcript layout.
- **G0 live:** 24/24 articles, 260,462 characters; manifest hash recorded in
  `graphrag/docs/g0.md`.
- **G1 live (data_test run):** the mini index remains at 40 chunks
  (`index.json` sha256 `d296e584…fd3cf809`) with 6/40 extracted from the earlier
  dev-tunnel attempt. The current Kaggle run uses the larger `data_test/` corpus:
  52 articles / 725 chunks, local Ollama, resumable extraction. Earlier dev-tunnel
  runs stopped on HTTP 530 rather than marking all pending chunks failed.

## Known issues / environment notes

- `python3 -m venv` is broken on this machine (no `ensurepip`); `.venv` was built
  with `uv`.
- `data_test/auto_pipeline.py` defaults to local Ollama (`qwen3-4b-instruct-2507`
  for chat, `nomic-embed-text` for embeddings). Kaggle's system `sitecustomize`
  imports `wrapt`, so the venv must have it before any venv Python call; this is
  bootstrapped by `prepare_environment.py`.
- The dev tunnel URL rotates; update `LLM_BASE_URL` in `.env` when it dies.
- The endpoint is chat-only (no `/embeddings`); the index uses the documented
  deterministic hash embedder unless `GRAPHRAG_EMBED_MODEL` is configured.
- The reasoning model spends output tokens before answering: extraction needs the
  4,096 cap and is slow; four concurrent calls did not improve throughput enough
  to justify the extra load.
- Wikipedia rate-limits batch extract requests; the fetch script backs off on 429
  and resumes, so interrupted runs are safe to re-run.
- Kaggle's `uv venv` does not install pip by default; `prepare_environment.py` uses
  `--seed` and bootstraps pip when a venv already exists without it.
- Kaggle injects a system `sitecustomize` via `PYTHONPATH`; the data_test scripts
  strip `PYTHONPATH` for venv calls to avoid `ModuleNotFoundError: wrapt`.
- The Ollama install script needs `zstd`; `setup_ollama.py` installs it and falls
  back to the standalone binary when the installer still fails.
- Qwen3-4B Q4 fits on one T4, so Ollama may keep it on a single GPU even with
  `OLLAMA_SCHED_SPREAD=1`; splitting a small model is not automatically faster.
