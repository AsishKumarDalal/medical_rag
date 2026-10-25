# Active task

## H6 — Minimal CLI (`goal.md` Part A, phase H6)

- [ ] `harness/cli.py`: Chat object (session + turn), REPL, command handling
- [ ] `/new`, `/resume <id>`, `/sessions`, `/usage`, `/help`, `/exit`
- [ ] Tool activity and context events printed compactly; `--once` and `--list`
- [ ] `python -m harness` entry point; Ctrl-C cancels the turn, not the process
- [ ] Tests with a stub model: ask, commands, --once, session switching
- [ ] Live CLI demo recorded in `test/docs/`

Done when: a human can hold a conversation from the terminal and resume it later.

---

_Done: H1 (loop), H2 (tools), H3 (context), H4 (sessions), H5 (security)._
