# Evaluation

Numbers for the query pipeline on the three-article mini corpus. Regenerate with
`test/eval/run_eval.py`; the question set is `test/eval/questions.json` (30 questions:
15 local, 8 global, 7 unanswerable).

## Setup

- **Corpus:** Alzheimer's disease, asthma, breast cancer (40 chunks, 36,344 chars).
- **Corpus snapshot:** `test/data/corpus_mini/manifest.json` sha256
  `6fa7c77e71898d4d218a7a36c0a32eefc57c8afd5731dfc3f8457184805db89f`; the three
  article files concatenated hash to
  `2f1e5ba0b4cf0ab470a79a8c5cbaa5b6f9e84393cd09acc90c50c700b0bb3f03`.
- **Index snapshot:** `index.json` sha256 `d296e584253827206a3bccc2d46ce082ef077fc862079791c6a08c05fd3cf809`.
- **Vectors:** deterministic hashed n-gram backend (`vectors.json` records the
  backend and why); the dev endpoint serves chat only, not embeddings.
- **k:** 8 evidence items per question.
- **Metrics:** retrieval recall@8 against hand-marked chunks, abstain accuracy on the
  seven out-of-scope questions, citation coverage against real artifact ids, latency
  and model calls per question.

```bash
.venv/bin/python test/eval/run_eval.py --offline   # deterministic baseline
.venv/bin/python test/eval/run_eval.py --live --k 8 # router + reranker + narrator
```

## Results

Filled from `test/eval/results.json` when the live run finishes. The offline row is
the deterministic baseline (no router, no reranker, no narrator).
