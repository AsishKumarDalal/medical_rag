# Medical Agentic RAG

An agent harness whose first tool is a GraphRAG over a small medical literature
corpus. Built in that order: harness first, retrieval second, integration third,
measurement before any upgrade.

- **Scope and phases:** [`goal.md`](goal.md)
- **How work is done here:** [`AGENT.md`](AGENT.md)
- **Current task:** [`TASK.md`](TASK.md)
- **Living status:** [`progress.md`](progress.md)
- **Durable learnings:** [`logs-and-fixes.md`](logs-and-fixes.md)
- **GraphRAG specification:** [`docs/rag_docs/`](docs/rag_docs/) (upstream, read-only)

## Layout

```
harness/    agent loop, tools, context/compression, sessions, security, CLI
  docs/     one completion record per phase: h1.md … h6.md
graphrag/   indexing pipeline (I1–I11) and query pipeline (Q0–QG)
  docs/     one completion record per phase: g0.md … g4.md
test/       tests, fixtures, corpus fetcher and G4 evaluation
docs/       design notes, the upstream spec, and docs/eval.md
```

## Status

The harness phases H1–H6 are complete and tested. The GraphRAG pipeline G0–G4 is
implemented; G0 fetched 24 medical articles and the G1–G4 live mini run is recorded
in [`progress.md`](progress.md) and the phase docs. The offline suite uses a stub
model and never needs the network.

## Running it

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env          # then set LLM_BASE_URL / LLM_MODEL
.venv/bin/python -m pytest -q # offline, stubbed model

# corpus + index (the live extraction is checkpointed and resumable)
.venv/bin/python test/data/test_data_fetch.py --pause 2
.venv/bin/python -m graphrag.build --stage extract --limit 40
.venv/bin/python -m graphrag.build --stage graph
.venv/bin/python -m graphrag.build --stage resolve
.venv/bin/python -m graphrag.build --stage communities
GRAPHRAG_CORPUS=test/data/corpus_mini .venv/bin/python -m graphrag.build --stage embed

# ask directly, or through the harness terminal
.venv/bin/python -m graphrag.ask --mode auto "What is asthma?"
.venv/bin/python -m harness --once "What is asthma?"

# G4 evaluation: deterministic baseline, then a small live smoke subset
.venv/bin/python test/eval/run_eval.py --offline --corpus test/data/corpus_mini
.venv/bin/python test/eval/run_eval.py --live --ids L01 L06 U01 \
  --corpus test/data/corpus_mini
```

The dev endpoint is chat-only: without an embeddings route the pipeline stores
deterministic hash vectors and records that backend in `vectors.json`.

## Requirements

Python 3.11+, an OpenAI-compatible chat endpoint, `pytest`, and outbound HTTP
access to the article sources for a fresh corpus.
