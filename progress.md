# Progress

Living status of this repo. Updated at the end of each phase. Task protocol and
the detailed work log live in `TASK.md` and `logs-and-fixes.md`.

- **Last updated:** 2026-10-25
- **Current phase:** G0 — test corpus (in progress)
- **Next:** finish the 24-article fetch, record counts in `test/docs/`, then G1

## Phases

| Phase | Scope | Status |
|---|---|---|
| **H1** | Agent loop: config, messages, LLM client, tool registry, loop | ✅ done |
| **H2** | Tool validation, result caps, RAG tool stub, tests | ✅ done |
| **H3** | Context budget: 45% tool-output stubbing, 75% LLM compression | ✅ done |
| **H4** | SQLite sessions: schema, store, resume | ✅ done |
| **H5** | Security: allowlist, budgets, injection wrapping, redaction | ✅ done |
| **H6** | Minimal CLI REPL + scripted run | ✅ done |
| **G0** | Corpus fetch script + 24 articles + offline tests | 🔄 in progress |
| **G1** | Indexing pipeline (I1–I11) + live mini run | ⬜ not started |
| **G2** | Query pipeline (Q0–QG) + live smoke queries | ⬜ not started |
| **G3** | Harness integration — RAG as the tool, end-to-end | ⬜ not started |
| **G4** | Evaluation set, run script, `docs/eval.md` | ⬜ not started |

## What exists now

```
harness/            the H1–H6 agent harness
  config.py         env-driven settings; enforces the 50k prompt+output window
  messages.py       typed messages/tool calls, provider-boundary conversion
  llm.py            single provider doorway: retries, normalization, max_tokens
  loop.py           the agent loop; copies caller conversation, returns transcript
  context.py        H3: token budget, stubbing + compression of old tool output
  session.py        H4: SQLite persistence and resume
  schema.sql        sessions/messages schema
  security.py       H5: tool allowlist, budgets, untrusted-text wrapping, redaction
  cli.py, __main__.py  H6: REPL + scripted run
  tools/            Tool contract, registry, validation, EchoTool, RAG stub
test/               pytest suite (offline, stub model — no network)
  docs/             per-phase run records: smoke-h1.md, h3-context.md, h6-cli.md
  data/             G0: fetch script + committed fixture
```

## Verification status

- **Offline suite:** green after every phase (unit tests with a stub model:
  loop behaviour, message round-trips, config validation, tool validation,
  context budget, session resume, security policy, CLI scripted mode, fetch
  parse/clean/slug). `compileall` clean.
- **Live (needs the dev endpoint):** H1 smoke run (`stop_reason=answer`,
  `test/docs/smoke-h1.md`), H3 token logging, H6 one-turn CLI run
  (`test/docs/h6-cli.md`). Live runs use `k2-horizon-mova-36b-a4b` via
  `LLM_BASE_URL` in `.env`.
- **G0 live:** fetch script runs against the Wikipedia extracts API; rate
  limits (HTTP 429) hit, backoff + resume handling added.

## G0 detail (current)

- [x] `test/data/test_data_fetch.py` written: 24 medical topics, one full
  extract per request (the API's `exlimit`/batch behaviour only reliably
  returns one page per call — documented in the script), deterministic
  cleaning, slug ids, per-article files + manifest with sha256, `--limit`,
  `--force`, pause, 429 backoff, resumable (skips articles already on disk,
  merges into the manifest).
- [x] `test/test_fetch.py` offline tests over the committed fixture.
- [ ] Fetch all 24 articles for real — **3 of 24 written** so far, fetch
  interrupted by rate limiting; re-run the script to continue.
- [ ] Record corpus counts in `test/docs/`.

## Known issues / environment notes

- `python3 -m venv` is broken on this machine (no `ensurepip`); `.venv` was
  built with `uv`.
- `.gitignore` needed `!.env.example` so the template isn't swallowed by
  `.env.*`.
- The dev tunnel URL rotates; update `LLM_BASE_URL` in `.env` when it dies.
- Wikipedia rate-limits batch extract requests; the fetch script backs off on
  429 and resumes, so interrupted runs are safe to re-run.
- All of the above are logged in `logs-and-fixes.md`.
