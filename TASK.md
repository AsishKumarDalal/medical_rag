# Active task

## Data test bundle (completed 2026-11-08)

- [x] `data_test/plan_data.md` — corpus, model, Kaggle, dataset, verification plan
- [x] `prepare_environment.py` — venv/pip, Ollama, `.env`, clean `PYTHONPATH`
- [x] `setup_ollama.py` — install Ollama, pull Qwen3-4B + nomic-embed-text,
      GPU/KV-cache settings, model alias
- [x] `auto_pipeline.py` — resumable corpus → index → verify → demo → HF upload
- [x] `build_index.py`, `verify_bundle.py`, `run_demo.py`, `download_dataset.py`
- [x] Auto-create/update `medical-graphrag-data-test` on Hugging Face
- [x] `data_test/docs/` — bundle and Kaggle run guides

## G1 — Indexing pipeline (`goal.md` Part B, phase G1)

- [x] I1/I2: `corpus.py` + `chunk.py` — load corpus, deterministic ~1000-char chunks
- [x] I3/I4: `extract.py` — per-chunk extraction, JSON salvage, per-chunk retry
      (`GRAPHRAG_EXTRACT_ATTEMPTS`, default 2), JSONL checkpoint, resume
- [x] I5/I6: `validate.py` + `normalize.py` — rejection gate, stable entity keys
- [x] I7/I8: `graph.py` — aggregate mentions/source chunks, majority type backfill
- [x] I9: `resolve.py` — blocking → fuzzy → union-find → LLM judge → canonical rebuild + audit
- [x] I10: `communities.py` — weighted label propagation + one summary per community
- [x] I11: `embed.py` — node/fact/chunk vectors, backend metadata, resumable
- [x] `build.py` stage CLI + `graphrag/llm.py` single model doorway
- [x] Offline tests for every stage with a stub LLM and a mini fixture corpus
- [ ] Live mini run on the three-article corpus (40 chunks)
      — paused at 6/40: the dev tunnel answers HTTP 530. Resume with
      `--stage extract --retry-failed` once `LLM_BASE_URL` is live, then
      `graph` → `resolve` → `communities` → `embed`
- [ ] `graphrag/docs/g1.md` with the live index numbers

Done when `python -m graphrag.build --stage all` rebuilds graph, communities and
vectors from the checkpoint, and the offline suite is green.

---

_Done: H1 (loop), H2 (tools), H3 (context), H4 (sessions), H5 (security),
H6 (CLI), G0 (corpus: 24 articles, 260,462 chars)._
