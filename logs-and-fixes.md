# Logs and fixes

Durable memory for this project. **Read the relevant section before debugging.**
Append a new entry at the top; never delete an old one.

Format:

```
## YYYY-MM-DD — short title
Context: what we were doing
Tried: approach → why it failed
Fixed: what actually worked
Do not: the trap to avoid next time
```

---

## 2026-10-25 — `python3 -m venv` fails on this machine

Context: creating `.venv` for `openai` + `pytest`.
Tried: `python3 -m venv .venv` → "ensurepip is not available", on Python 3.14.4
(Ubuntu, no `python3.14-venv` package).
Fixed: `uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`.
`uv` was already at `~/.local/bin/uv`.
Do not: `apt install python3.14-venv` (sudo, heavy) when `uv` is right there.

## 2026-10-25 — Scripts under `test/` cannot import `harness`

Context: `python test/smoke_h1.py` → `ModuleNotFoundError: No module named 'harness'`.
Tried: assuming the project root is on `sys.path`; running a file puts its own
folder (`test/`) there instead.
Fixed: scripts insert `Path(__file__).resolve().parents[1]` at the top; pytest
gets the root via `pythonpath = .` in `pytest.ini`, so `pytest -q` works from anywhere.
Do not: create a `conftest.py` that patches `sys.path` when the ini option exists.

## 2026-10-25 — The dev endpoint does tool calling, but returns SDK objects

Context: first live H1 run against the Ollama tunnel (`k2-horizon-mova-36b-a4b`).
Tried: reading `message.tool_calls[0]` as a dict → attribute errors are easy to hit.
Fixed: `openai` 3.x returns pydantic models; normalize with `model_dump()` before
`ToolCall.from_provider(...)`. The model does emit `tool_calls` with `tools=` set,
so the harness can rely on real tool calling in development. Keep `max_tokens` at
4096: prompt + output must stay under 50,000, and `Config` rejects a completion cap
that leaves no room.
Do not: send `temperature` by default — reasoning models can reject it. It stays
unset unless `LLM_TEMPERATURE` is defined.

## 2026-10-25 — The upstream "rag_docs" link 404s unauthenticated

Context: fetching `github.com/AsishKumarDalal/final-year-project/tree/main/docs/rag_docs`
to read the GraphRAG spec.
Tried: `read_url` and plain `curl` to the GitHub API → both 404, although the repo
is public and the path exists.
Fixed: use the authenticated CLI (`gh api repos/<owner>/<repo>/contents/docs/rag_docs`)
or a sparse clone; both work. The blob pages are fine, the HTML tree route is not.
Do not: conclude a repo is missing before checking with `gh`.

## 2026-10-25 — `medical_rag` was created empty

Context: bootstrapping the project repository.
Tried: treating `main` as existing.
Fixed: `git init -b main`, one bootstrap commit, push `main`, then branch + PR for
actual work. The GitHub default branch now exists.
Do not: open the first PR against a branch that has no commits — push a bootstrap
commit first.

## 2026-10-25 — The spec is 5,000 lines; read it by headings

Context: grounding the GraphRAG phases without burning context on five long docs.
Tried: reading `full_Architecture.md` whole (~1,170 lines of dense text).
Fixed: `grep -E '^#{1,3} '` over the docs to get the stage map (I1–I11, Q0–QG),
then read only the sections being implemented. `architecture.md` is the readable
"why"; `full_Architecture.md` is the contract.
Do not: paste spec prose into prompts. Cite stage ids instead.

## 2026-10-25 — Deferred overlays are traps for the unwary

Context: the spec ships two optional designs, `NER_model.md` (encoder extraction)
and `system-1-model.md` (System One / Jev typed decisions).
Tried: nothing yet — this entry exists to prevent the temptation.
Fixed: out of scope until G4 numbers exist. Reranking, merge judging and routing use
the LLM now, each behind a one-module interface (`rerank(...)`, judge, router).
Do not: implement both a heuristic and an LLM path "for later". One path, documented
seam.
