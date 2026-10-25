# graphrag/

Retrieval, as a library the harness calls. Stages follow
[`../docs/rag_docs/full_Architecture.md`](../docs/rag_docs/full_Architecture.md).

**Indexing (G1), each stage re-runnable from the previous artifact:**

| Module | Stage | Output |
|---|---|---|
| `corpus.py` | I1 | article records, stable ids |
| `chunk.py` | I2 | ~1000-char chunks with overlap |
| `extract.py` | I3–I4 | entities + triples per chunk, checkpointed JSONL |
| `validate.py` | I5 | accepted facts + rejection log |
| `normalize.py` | I6 | stable identity per entity name |
| `graph.py` | I7–I8 | nodes, edges, mention counts, source chunks |
| `resolve.py` | I9 | merge funnel, judge, canonical rebuild, audit log |
| `communities.py` | I10 | communities + one summary each |
| `embed.py` | I11 | chunk and entity-name vectors |

**Query (G2):** `probe → route → seeds (names ∪ vectors) → expand (walk ∪ fetch) →
rerank → narrate → verify` for local questions, `map → filter → reduce` for global
ones. Public entry point: `graphrag.answer(question)`.

Deferred: encoder extraction, System One / Jev, graph database, incremental
indexing, visualization. The checkpoint is the precious artifact — everything
downstream must be rebuildable without new extraction.
