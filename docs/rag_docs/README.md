# GraphRAG Documentation — Index

A **domain-agnostic GraphRAG** over Wikipedia text: raw text → LLM-extracted
knowledge graph → LLM-resolved entities → communities → interactive
visualization → question answering with citations.

These docs are the design record for that pipeline. They are written as theory:
ideas first, diagrams second, code almost never.

> **Status:** the implementation is being rebuilt from scratch. This folder is
> the specification the new code is written against — the `.py` files that
> produced the numbers quoted here are not currently in the tree. The one
> exception is **[system-1-model.md](system-1-model.md)**, which is a forward
> proposal against the new pipeline, not a description of the old one.

## Read in this order

| Doc | What it covers |
|---|---|
| **[full_Architecture.md](full_Architecture.md)** | **The spec. Start here.** The consolidated final architecture: ingestion stages I1–I11 and query stages Q0–QG in order, the algorithm index, on-disk contracts, the cost model, failure modes and limits. This is what the new code is written against. |
| **[architecture.md](architecture.md)** | The whole system as theory: every stage, the idea behind it, and why each design decision exists. Readable without reading any code. |
| **[graph_making.md](graph_making.md)** | The build pipeline stage by stage: fetch → clean → chunk → extract → validate → aggregate, with real examples and diagrams. |
| **[solution.md](solution.md)** | The entity-merging problem ("Tesla, Inc." / "Tesla" / "Tesla Motors, Inc." became three nodes) and the layered, auditable, LLM-decided fix. Includes the live failures that shaped it. |
| **[embed_research.md](embed_research.md)** | Why a structural graph benefits from a geometric layer: geometry vs. topology, the five insertion points, evidence ranking, and the hybrid architecture (two doors + vector fallback). |
| **[NER_model.md](NER_model.md)** | Alternative extraction design — replace the LLM *writer* with a small encoder *labeler* (BIO + typed pair classification), silver labels from the graph, and where it beats prompting. |
| **[system-1-model.md](system-1-model.md)** | *Optional, not history.* TypeSafe AI's System One Models & Jev — typed probabilistic decisions at 70–500 ms — and eight insertion points in this RAG (type classification, relation direction, merge scoring, chunk triage, reranking, community pruning, routing, answer guardrail) with cost arithmetic, confidence gates, an evaluation ladder, and honest limits. |

## The three defining choices

1. **LLM extraction.** Every chunk is sent to a prompted model that returns
   `{"entities": [...], "triples": [...]}` — one call per chunk, resumable.
2. **Cheap layers nominate, expensive layers decide.** Blocking and fuzzy
   matching surface candidate duplicates for free; the LLM only ever judges
   those candidates, and every verdict is written to an audit log.
3. **The graph stores structure; text stays retrievable.** Every edge keeps its
   `source_chunks`, so answers are citable back to source text rather than
   black-box.

## Principles that survive any domain

- **LLM output is a suggestion, not data.** Every stage validates,
  normalizes, caps, and logs before trusting anything.
- **Mention counts are free confidence scores.** Real relationships get
  re-extracted across documents; junk appears once.
- **Checkpoint everything expensive.** Extraction is paid once; every
  downstream stage rebuilds in seconds for free.
- **Over-merging is worse than under-merging.** Two fragments of one entity
  cost edge weight; a single false merge poisons traversals.