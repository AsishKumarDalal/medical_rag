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

## 2026-10-25 — Hash cosine alone ranks the wrong medical passage

Context: G4's offline retrieval baseline showed "What is asthma?" losing the
definition chunk, and "What triggers asthma symptoms?" returning unrelated asthma
chunks.
Tried: trusting the hash-vector cosine; trigram collisions and generic medical
words drowned the intent word (`trigger`).
Fixed: for the hash backend, require shared content words, rescore by overlap
(cosine as a tie-breaker), and require two matches when there is no graph seed.
This also made the out-of-scope questions abstain instead of returning a random
passage. Real embedding backends still use pure cosine.
Do not: tune a lexical fallback as if it were a semantic model; use the signal it
actually has, and say so in the artifact metadata.

## 2026-10-25 — A narrator refusal was being converted into a cited answer

Context: Q5 promised an honest "evidence does not contain this" path even when
retrieval had returned some chunks.
Tried: checking only whether the answer had citations; an LLM refusal has none,
so the code appended source ids automatically and returned a confident-looking
answer.
Fixed: detect the exact refusal phrase after local narration and global reduction
and return an abstention with no citations. Added a test with a refusing narrator.
Do not: treat "no citations" as "add citations"; a refusal is a result, not a
formatting miss.

## 2026-10-25 — Hash-vector seeds need a lexical guard

Context: testing Q3's "no name seed" fallback with `airway inflammation`.
Tried: trusting cosine over hash-vector node names with the normal threshold →
`diabetes` and `wheezing` were seeded at ~0.17 by character-n-gram collisions.
Fixed: when `vectors.backend` starts with `hash`, keep a vector seed only if the
question shares a meaningful content word with the node key. Real embedding
backends keep the pure cosine door.
Do not: evaluate a lexical fallback as if it were a semantic embedding model.

## 2026-10-25 — Four concurrent extraction calls are not four times faster

Context: measuring whether to parallelise the 333-chunk live extraction.
Tried: four concurrent `extract_chunk` calls → 684 s wall time for four chunks
(212–684 s each), versus roughly 55–190 s for one sequential chunk.
Fixed: kept `--workers` as an explicit option but use one worker for the live run;
the endpoint is the bottleneck and extra concurrency mostly adds queueing.
Do not: scale workers before measuring the provider, especially through a tunnel.

## 2026-10-25 — `llm or GraphLLM(...)` silently defeated offline tests

Context: G3 integration test passed `llm=None` to `GraphRagBackend` and took 17 s.
Tried: assuming `None` meant "no model"; the constructor used `llm or ...` and
created a live client, so the router/narrator hit the real endpoint.
Fixed: an explicit `_UNSET` sentinel distinguishes "argument omitted" (build the
default client) from `llm=None` (offline, deterministic path).
Do not: use truthiness to express an optional dependency when `None` is a valid
configured value.

## 2026-10-25 — ChunkIndex serialization lost directly constructed chunks

Context: G3 built a tiny `ChunkIndex(chunks=[...])` and saved it with
`write_index`; the reloaded engine had zero chunks and rejected every citation.
Tried: assuming `to_dict()` serialized the `chunks` field; it only serialized the
per-article records, and the test fixture's article records had no `chunks` list.
Fixed: `to_dict()` now fills each article's chunk list from the authoritative
top-level `self.chunks` when it is missing. The on-disk shape is unchanged.
Do not: test a loader with an object whose in-memory and serialized shapes differ.

## 2026-10-25 — The dev endpoint has chat only, no embeddings endpoint

Context: G1's I11 needs node/fact/chunk vectors.
Tried: `client.embeddings.create(model="nomic-embed-text", ...)` → HTTP 404
"model not found"; `/v1/models` lists only the GGUF chat model.
Fixed: `graphrag/embed.py` has two backends behind one interface. If
`EMBED_BASE_URL`/`EMBED_MODEL` are set it calls an OpenAI-compatible
`/embeddings`; otherwise it falls back to a deterministic, dependency-free
hashing embedder stored with `backend="hash"` in `vectors.json`. Query stages
never need to know which produced the vectors.
Do not: assume the chat endpoint also serves embeddings because it speaks the
OpenAI API. Check `/v1/models` first.

## 2026-10-25 — This reasoning model returns empty content when max_tokens is tiny

Context: probing the endpoint with `max_tokens=16`.
Tried: one-line "Reply with exactly OK" → `finish_reason=stop`, content `""`,
16 completion tokens consumed by reasoning.
Fixed: give extraction/narration at least ~800–1,200 output tokens; the model
spends tokens on reasoning before the answer. H1's 4,096 default is safe.
Do not: diagnose an empty answer as a broken loop before checking the
completion cap — reasoning models can think the whole budget away.

## 2026-10-25 — `.env.*` in `.gitignore` silently excluded `.env.example`

Context: committing `.env.example` alongside the ignored local `.env`.
Tried: `git add .env.example` → the file stayed untracked, no error.
Fixed: added `!.env.example` after the `.env.*` rule; `git check-ignore -v`
confirms the exception. `.env` itself stays ignored.
Do not: assume an ignored file is ignored for the right reason — check with
`git check-ignore -v <path>` before concluding a file is missing.

## 2026-10-25 — The dev tunnel URL rotates; `.env` goes stale silently
Context: `LLM_BASE_URL` points at a trycloudflare tunnel that dies and gets a new
URL whenever the machine restarts the tunnel.
Tried: nothing — recorded so the next live-run failure is not blamed on the code.
Fixed: when a live run fails with connection errors, update `LLM_BASE_URL` in
`.env` first; everything else (model name, `api_key=ollama`) stays the same.
Do not: debug `harness/llm.py` for a connection error — check the URL.

## 2026-10-25 — Wikipedia returns ONE full extract per request, whatever exlimit says

Context: fetching 24 articles with `titles=A|B|C` and `exlimit=20`.
Tried: batching titles → the API answers with a warning, "exlimit was too large
for a whole article extracts request, lowered to 1", and returns a single extract.
Three batches produced three articles, not 24, with no error.
Fixed: one request per title. `exintro=1` would allow batches again, but only the
lead section, which is too thin for a corpus that needs real paragraphs.
Do not: trust `exlimit` for full extracts; check `count` in the manifest after a run.

## 2026-10-25 — Sequential Wikipedia fetches hit HTTP 429

Context: switching the fetcher to one request per title.
Tried: 0.5 s pause and 3 attempts → HTTP 429 Too Many Requests after a few titles.
Fixed: 2 s pause, 4 attempts, honour `Retry-After` else back off 10 s × attempt, and
keep going when a single title fails. `write_corpus(merge=True)` re-reads the old
manifest, so a partial run resumes instead of refetching what already landed.
Do not: retry a 429 immediately — the block gets longer.

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
