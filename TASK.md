# Active task

## H5 — Security sandboxing (`goal.md` Part A, phase H5)

- [ ] `harness/security.py`: policy defaults, allowlist, size caps, tool-call budget
- [ ] Untrusted-document wrapper + neutralization for retrieved text
- [ ] Secret redaction, wired into session writes and error strings
- [ ] Path containment helper (ready for future file tools)
- [ ] Wire policy into the registry and the loop; every refusal is a tool error
- [ ] Tests: unknown/disallowed tool, oversized args, budget exhaustion, injection
      payload stays data, redaction, traversal rejection

Done when: `pytest -q` is green and a document cannot turn into a tool call.

---

_Done: H1 (loop), H2 (tools), H3 (context), H4 (sessions)._
