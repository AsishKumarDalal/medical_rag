# Active task

## H1 — Agentic loop (`goal.md` Part A, phase H1)

Goal: one turn of conversation end to end — messages → LLM → tool calls → results →
final answer — with one trivial `echo` tool to prove the loop.

- [ ] `harness/llm.py`: chat + tool-call client, retries with backoff, config for
      model/temperature/max tokens
- [ ] `harness/messages.py`: message and tool-call types + provider serialization
- [ ] `harness/loop.py`: dispatch loop with iteration cap; errors become messages
- [ ] `harness/tools/`: `Tool` protocol + registry + `echo` tool
- [ ] `harness/config.py`: env-based settings (`.env.example`)
- [ ] `test/test_loop.py`: stubbed LLM — tool round-trip, iteration cap, model error
- [ ] Smoke run: scripted transcript printed to stdout and pasted into the PR

Done when: `pytest -q` is green and the smoke run shows a tool call and a final answer.
