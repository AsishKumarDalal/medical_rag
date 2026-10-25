# Active task

## H4 — Sessions in SQLite (`goal.md` Part A, phase H4)

- [ ] `harness/schema.sql`: sessions, messages, tool_calls, context_events, usage
- [ ] `harness/session.py`: create / append / load / list / delete, WAL mode
- [ ] Recorder wired into `run_turn` so messages are durable mid-turn
- [ ] Resume a session by id; two sessions stay isolated
- [ ] Usage totals for `/usage`; context events preserved
- [ ] Tests: round-trip, crash mid-turn, reopen, cascade delete, redaction hook

Done when: `pytest -q` is green and a session can be closed and reloaded with its
tool calls intact.

---

_Done: H1 (loop), H2 (tools), H3 (context)._
