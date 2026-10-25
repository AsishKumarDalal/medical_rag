# Medical Agentic RAG

An agent harness whose first tool is a GraphRAG over a small medical literature
corpus. Built in that order: harness first, retrieval second, integration third,
measurement before any upgrade.

- **Scope and phases:** [`goal.md`](goal.md)
- **How work is done here:** [`AGENT.md`](AGENT.md)
- **Current task:** [`TASK.md`](TASK.md)
- **Durable learnings:** [`logs-and-fixes.md`](logs-and-fixes.md)
- **GraphRAG specification:** [`docs/rag_docs/`](docs/rag_docs/) (upstream, read-only)

## Layout

```
harness/    agent loop, tools, context/compression, sessions, security, CLI
graphrag/   indexing pipeline (I1–I11) and query pipeline (Q0–QG)
test/       test scripts; test/data/ holds the corpus fetch script
docs/       design notes and the upstream GraphRAG spec
```

## Status

Preparation only: documents and structure. No code yet — phase H1 is next in
`TASK.md`.

## Requirements (from H1 onward)

Python 3.11+, an OpenAI-compatible chat endpoint, `pytest`, and — after
`test/data/test_data_fetch.py` exists — outbound HTTP access to the article sources.
