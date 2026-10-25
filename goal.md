# Goal — Medical Agentic RAG

**In one line:** a small, readable agent **harness** whose first and only tool is a
**GraphRAG over a medical literature corpus**.

The harness is built first. The RAG system is built second, then plugged into the
harness as a tool. Everything is measured on a small local corpus before it is
made bigger.

- **Part A — Harness** (phases H1–H6): agent loop → tools → context/compression →
  sessions → security → CLI.
- **Part B — GraphRAG** (phases G0–G4, deferred G5): corpus → indexing → querying →
  harness integration → evaluation.

Specification for Part B: `docs/rag_docs/full_Architecture.md` (stages **I1–I11**
indexing, **Q0–QG** querying). Where this file and that spec disagree, that spec
wins for GraphRAG internals; this file wins for scope and ordering.

---

## 0. Principles

1. **One LLM doorway.** All model calls go through a single client module. No
   provider SDK is imported anywhere else.
2. **Model output is a suggestion, not data.** Validate, cap, normalize, log —
   at every stage, in both parts.
3. **Cheap layers nominate, expensive layers decide.** Applies to entity merging,
   reranking, and context compression alike.
4. **Every answer is citable.** Retrieved evidence keeps its chunk ids back to the
   source text. No black-box answers.
5. **Fail loudly and partially.** A missing answer is reported as missing; a failed
   stage is skipped and logged, not silently defaulted.
6. **Small enough to read.** Prefer plain dicts, JSON artifacts and SQLite over
   frameworks, until a measurement says otherwise.

**Safety framing:** this is a medical *literature retrieval and summarization*
assistant. It is not a diagnostic tool, it does not give treatment advice about a
person, and it must say "the evidence does not contain this" when that is true.

---

## 1. Repo layout (target)

```
goal.md                 scope, phases, deliverables          ← this file
AGENT.md                how the agent works (task + logs protocol)
TASK.md                 the one active task, with subtasks
logs-and-fixes.md       durable learnings; read before debugging
docs/
  rag_docs/             upstream GraphRAG specification (read-only reference)
harness/                agent loop, tools, context, sessions, security, CLI
graphrag/               corpus → graph → query
test/                   all test scripts and fixtures
  data/                 fetch script + cached corpus (corpus git-ignored)
```

No `src/`, no framework scaffolding. Two importable packages: `harness`, `graphrag`.

---

# Part A — Harness

Target shape: a user message goes in, the model may call tools (today: the RAG
tool), results come back as messages, and the loop ends with a final answer that
is persisted and displayed. Each phase must leave the harness runnable.

## Phase H1 — Agentic loop

**Goal:** one turn of conversation, end to end: `messages → LLM → tool calls →
tool results → LLM → final answer`, with no hidden magic.

**Deliverables**
- `harness/llm.py` — single client: chat completion with tool-calling, retries with
  backoff, one place for model name / temperature / max tokens.
- `harness/messages.py` — message + tool-call types, serialization to/from the
  provider format, token-ish accounting hooks for H3.
- `harness/loop.py` — the loop: call model, dispatch tool calls, append results,
  repeat until a message with no tool calls; hard cap on iterations; surface model
  and tool errors as messages rather than exceptions.
- One trivial built-in tool (`echo`) used only to prove the loop.
- A smoke script under `test/` that prints a full transcript.

**Not deliverables**
- Any real tool, persistence, compression, security policy, or CLI (H2–H6).
- Streaming, multi-agent, parallel tool calls, MCP.
- Provider abstraction beyond "OpenAI-compatible endpoint".

## Phase H2 — Tools

**Goal:** a tool registry the loop can trust: declared schemas, validated
arguments, bounded results, and one documented contract for adding tools.

**Deliverables**
- `harness/tools/base.py` — `Tool` protocol: `name`, `description`, `json_schema`,
  `run(args) -> ToolResult`.
- `harness/tools/registry.py` — register/lookup, JSON-schema generation for the
  model, unknown-tool and bad-argument handling (returned to the model as errors,
  never raised through the loop).
- `ToolResult` contract: `ok | error`, content string, optional structured payload,
  byte/token cap with truncation marker.
- `harness/tools/rag.py` — `search_medical_literature(question, mode, k)` as a
  **stub** returning "graphrag not wired yet" until G3. The signature is fixed now
  so the prompt and loop are written against it once.
- Tests: registry discovery, bad args, error results, oversized result truncation.

**Not deliverables**
- Any tool other than the RAG stub (no shell, file, web, python-exec).
- Parallel or streaming tool execution, MCP, tool marketplaces.

## Phase H3 — Context management & compression

**Goal:** the conversation never overflows the window, and what gets dropped is
predictable and auditable.

**Rules (defaults live in config; the thresholds are part of the design)**
- **Token accounting** per message and per request; budget = model window − output
  reserve. Counted, not guessed, and recomputed after every append.
- **At 45% of budget — cut stale tool output.** Replace long, older tool results
  (all but the most recent 2, and only those above a size threshold) with a short
  stub: tool name, byte count, and a reference to the stored full output. The
  original stays in the session store, so it is recoverable but not re-sent.
- **At 75% of budget — LLM compression.** Summarize the middle of the conversation
  into one summary message. **First 3 and last 6 messages are never touched and
  never compressed** — they stay verbatim and byte-identical.
- Never drop: system prompt, the current task statement, the latest user message.
- Truncation and compression are **idempotent** — running the same pass twice
  changes nothing the second time — and every event is logged with the token
  counts before and after.

**Deliverables**
- `harness/context.py` — counter, budget, `maybe_compress()` with the 45% and 75%
  passes, event records.
- Tests: thresholds fire exactly once; first-3/last-6 preserved byte-for-byte;
  stub replacement keeps recoverability; idempotence; summary failure falls back to
  deterministic truncation.
- A scripted long conversation in `test/` that drives both triggers.

**Not deliverables**
- Embedding/vector memory, cross-session memory, RAG-over-history.
- Summarizer prompt tuning beyond correctness; per-model tokenizer zoo (one token
  estimate, documented).

## Phase H4 — Sessions (SQLite)

**Goal:** every conversation is durable and resumable by id, including a loop
interrupted mid-tool-call.

**Deliverables**
- `harness/session.py` + `harness/schema.sql` — tables for `sessions`, `messages`,
  `tool_calls`, `context_events`, `usage`; schema created on first run.
- Append-only writes during the loop (a crash keeps the turn up to the last
  completed step), resume by session id, list/inspect helpers.
- Tests: create → append → reload round-trip; resume after a simulated crash
  mid-loop; compression events survive reload; two sessions stay isolated.
- WAL mode on; one connection per process, no ORM.

**Not deliverables**
- Users, auth, sharing, Postgres, migration tooling, a web API.

## Phase H5 — Security sandboxing

**Goal:** bound what the model can do and neutralize what retrieved text can do.

**Scope**
- **Tool arguments:** validated against schema; unknown tools and oversized args
  rejected before execution.
- **Authorization:** per-tool allowlist in config; the loop only offers allowed
  tools to the model.
- **Prompt injection:** retrieved documents are data. They are delimited/labelled
  as untrusted, never executed as instructions, and any instruction-like content is
  passed through to the model *as quoted evidence only*; the system prompt states
  that document content cannot change the task.
- **Resource bounds:** per-session tool-call budget, per-call timeout, output caps,
  request size caps.
- **Secrets:** never written to logs or session rows; env-only configuration;
  redaction pass on error strings.
- **Paths (ready, unused):** a single rooted base dir with traversal rejection, for
  when file tools arrive.

**Deliverables**
- `harness/security.py` + config defaults, wired into the loop and tools.
- Tests: unknown tool, invalid args, budget exhaustion, injection payload in a
  retrieved chunk cannot trigger a tool call, secret-shaped string is redacted.

**Not deliverables**
- Containers, VMs, syscall/OS-level sandboxing, egress firewalling, red-team
  reports. Those are deployment concerns, not this phase.

## Phase H6 — Minimal CLI

**Goal:** the human can drive the harness by hand in a terminal — nothing more.

**Deliverables**
- `python -m harness` REPL: prompt, answer, tool calls shown compactly, token usage
  and compression notices, `/new`, `/resume <id>`, `/sessions`, `/usage`, `/exit`.
- Plain `input()` + `print()`; Ctrl-C cancels the current turn, not the process.
- One `test/` script that pipes a scripted conversation through the CLI in
  non-interactive mode and asserts the transcript.

**Not deliverables**
- TUI, streaming markdown, colors beyond default, web UI, voice, multi-user.

---

# Part B — GraphRAG

Corpus → chunk → extract → validate → normalize → aggregate → resolve → communities
→ embeddings → query. Follow `docs/rag_docs/full_Architecture.md`; read
`architecture.md` for the *why* and `graph_making.md` for the mechanics.

**Explicitly out of scope for now, and why:** `NER_model.md` (encoder extraction)
and `system-1-model.md` (System One / Jev typed decisions) are *deferred overlays*.
Every decision they would make — reranking, merge judging, routing, triage — is made
by the LLM for now behind a small interface, so swapping them in later touches one
module, not the pipeline.

## Phase G0 — Test corpus

**Goal:** 20–30 real medical articles, fetched reproducibly, cached on disk, small
enough to iterate on in minutes.

**Deliverables**
- `test/data/test_data_fetch.py` — URL manifest (20–30 articles from 2–3 plain-HTML
  sources, e.g. MedlinePlus, NCBI Bookshelf/StatPearls, Wikipedia medicine), fetch
  with polite rate limit + retries, strip boilerplate, write one file per article
  plus `manifest.json` (url, title, fetched-at, sha256, char count).
- `test/data/corpus/` — **git-ignored** (regenerated by the script).
- `test/test_fetch.py` — offline test over committed fixtures: cleaning removes
  nav/boilerplate, manifest is consistent, no empty documents.

**Not deliverables**
- A general crawler, PDF/JS-rendered sources, paywalled content, scheduled refresh,
  more than ~30 articles, licensing beyond personal research use.

## Phase G1 — Indexing (I1–I11)

**Goal:** a checkpointed, re-runnable pipeline that turns the corpus into a
validated, merged, embedded knowledge graph with community summaries.

**Deliverables, in stage order**
- **I1 acquisition** — load corpus files; deterministic article ids.
- **I2 chunking** — ~1000-char windows on paragraph boundaries with overlap; chunk
  ids as `article:index`; identical output on re-run.
- **I3 extraction** — one LLM call per chunk → `{entities, triples}`; per-chunk cap
  (~20 triples); strict JSON salvage; per-chunk retry then permanent-failure record.
- **I4 checkpointing** — append-only JSONL, resume by chunk id, failed chunks never
  re-billed blindly. This is the only expensive artifact.
- **I5 validation gate** — shape, self-loops, vague groups, compound "A and B"
  names, runaway name lengths; rejects logged with reasons.
- **I6 normalization** — stable identity (case, accents, punctuation, leading
  articles, short legal-suffix list); display name kept separately.
- **I7 aggregation** — one edge per (head, relation, tail) with mention count and
  sorted `source_chunks`; node mention counts.
- **I8 type backfill** — majority type per node from its mentions.
- **I9 entity resolution** — blocking → fuzzy scoring (nominate only) → clustering →
  one LLM judge per group shown names, types, counts and sample edges → deterministic
  rebuild with code-chosen canonical name → `merge_audit.jsonl` for every candidate
  and verdict. Bias is explicitly "do not merge".
- **I10 communities** — undirected log-weighted graph, label propagation with a
  fixed seed, one LLM summary per community, tiny communities skipped.
- **I11 embeddings** — chunk vectors + entity-name vectors, stored on disk with the
  model name and dimension; used by Q2/Q4.
- Stage CLI: `python -m graphrag.build --stage extract|graph|resolve|communities|embed
  [--resume]`, each stage rebuildable from the checkpoint.

**Not deliverables**
- NER encoder extraction, System-1/Jev, a graph database, incremental indexing,
  visualization UI, semantic (similarity-created) edges.
- Agentic/iterative extraction, multi-pass refinement of extracted facts.

## Phase G2 — Query (Q0–QG)

**Goal:** answer two shapes of question — "how are X and Y connected" and "what does
this corpus say about Z" — with citations, or abstain honestly.

**Deliverables**
- **Q0 probe / Q1 route** — cheap LLM call: local vs global vs abstain; failure
  falls back to local (cheaper, better tested).
- **Q2 seeds, two doors** — name linking (exact → normalized → containment, most
  mentioned wins) ∪ embedding similarity over entity names and chunks.
- **Q3 expand** — 2-hop walk in both directions ∪ vector fetch of top chunks; every
  fact keeps its source chunks.
- **Q4 rank** — seed proximity, mention count, chunk support; then an **LLM
  reranker** over the shortlist. Interface: `rerank(question, candidates, k)` so an
  embedding/cross-encoder implementation drops in later.
- **Q5 narrate** — answer strictly from supplied evidence, inline citations
  (`[article:chunk]`), explicit "evidence does not contain this" path.
- **Q6 verify** (light) — check every claim maps to a cited chunk; drop or flag
  unsupported sentences.
- **QG global** — map (score each community summary) → filter → reduce into one
  answer citing community ids.
- Entry point: `graphrag.answer(question) -> {answer, citations, evidence, mode}`
  and `python -m graphrag.ask "..."` for manual runs.
- Tests: router fallback, no-seed → vector fallback, abstain path, citation
  formatting, every returned citation resolves to a real chunk.

**Not deliverables**
- Cross-encoder reranking, fine-tuned models, >2-hop reasoning, query decomposition,
  semantic answer caching.

## Phase G3 — Harness integration

**Goal:** the RAG system is *the* harness tool, with a stable contract.

**Deliverables**
- `harness/tools/rag.py` real implementation: `search_medical_literature(question,
  mode="auto", k=8)` → answer + citations + evidence ids; `graphrag` import isolated
  in one module; errors become tool errors, never tracebacks in the loop.
- Argument/result caps tuned to H3's stubbing thresholds.
- End-to-end test with a stubbed LLM: question → tool call → cited answer.
- One recorded demo transcript in `test/`.

**Not deliverables**
- Additional tools (shell, web, calculator) — explicitly skipped for now.

## Phase G4 — Evaluation

**Goal:** numbers instead of impressions, before any upgrade.

**Deliverables**
- `test/eval/questions.json` — ~30 hand-written questions: local (named entities),
  global (themes), and unanswerable (must abstain).
- Scripts computing retrieval recall@k against hand-marked relevant chunks, citation
  coverage, abstain accuracy, and latency/call counts per question.
- `docs/eval.md` — results table, with the corpus snapshot hash it was measured on.

**Not deliverables**
- Automated LLM-judge faithfulness at scale (manual spot-checks only), clinical
  validation, load testing.

## Phase G5 — Deferred upgrades (do not start before G4 numbers exist)

Replace the LLM reranker with embeddings/cross-encoder · encoder-based extraction
(`NER_model.md`) · System One / Jev typed decisions (`system-1-model.md`) ·
incremental indexing · graph visualization · graph database.

---

## Non-goals (whole project)

- Diagnosis, treatment recommendations, or anything touching patient data.
- Web UI, mobile app, multi-tenant auth, billing.
- Training or fine-tuning models.
- Production deployment, horizontal scale, uptime guarantees.
- Chasing corpus size before the pipeline is measured on 20–30 articles.

## Definition of done (applies to every phase)

1. Code + tests committed; `pytest -q` green.
2. The behaviour was exercised the way a user meets it (CLI run, script run, or tool
   round-trip) — not just imported successfully.
3. Any non-obvious fix or dead end is written into `logs-and-fixes.md`.
4. The finished task is removed from `TASK.md`.

## Stack

Python 3.11+, an OpenAI-compatible chat endpoint (model choice stays configurable),
SQLite (stdlib), pytest. Add `numpy` only when embeddings arrive; add `networkx` only
if the hand-rolled graph code becomes the bottleneck. No agent framework.
