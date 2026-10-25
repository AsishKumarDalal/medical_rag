# Active task

## H3 — Context management & compression (`goal.md` Part A, phase H3)

- [ ] `harness/context.py`: token estimate, prompt budget, `Budgeted` result
- [ ] 45% pass: stub stale long tool outputs for the request only
- [ ] 75% pass: summarize the middle, keep first 3 and last 6 verbatim
- [ ] `LLMSummarizer`, with deterministic fallback when it fails
- [ ] Wire into `run_turn` (context events returned, not lost)
- [ ] Tests: thresholds, preservation, idempotence, fallback, orphaned tool results

Done when: `pytest -q` is green and a scripted long conversation triggers both
passes exactly once.

---

_Done: H1 (agentic loop), H2 (tool validation, caps, RAG stub)._
