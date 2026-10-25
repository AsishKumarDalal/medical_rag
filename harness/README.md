# harness/

The agent, not the retrieval. Intended modules (built in phases H1–H6 of `goal.md`):

| Module | Phase | Responsibility |
|---|---|---|
| `config.py` | H1 | env-based settings, thresholds, model choice |
| `llm.py` | H1 | **the only** model client: chat + tool calls |
| `messages.py` | H1 | message / tool-call types, provider serialization |
| `loop.py` | H1 | call → dispatch tools → append → repeat |
| `tools/` | H1–H2 | `base.py` contract, `registry.py`, `echo.py` (H1 proof), `rag.py` (real in G3) |
| `context.py` | H3 | token budget, 45% tool-output stubbing, 75% LLM compression |
| `session.py` | H4 | SQLite persistence, resume by id |
| `security.py` | H5 | allowlists, caps, untrusted-document handling, redaction |
| `cli.py` | H6 | minimal REPL |

Rules that hold across the package: no provider SDK outside `llm.py`; tool errors
come back as tool results, never tracebacks; every compression event is logged with
token counts.
