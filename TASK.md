# Active task

## G0 — Test corpus (`goal.md` Part B, phase G0)

- [ ] `test/data/test_data_fetch.py`: 24 medical topics from the Wikipedia extracts API
- [ ] Deterministic cleaning, slug ids, per-article files + manifest with sha256
- [ ] Polite fetch: retries, pause, user agent, `--limit`, `--force`
- [ ] `test/test_fetch.py`: offline parse/clean/slug tests over a committed fixture
- [ ] Fetch the corpus for real and record counts in `test/docs/`

Done when: `python test/data/test_data_fetch.py` produces 20–30 articles and an
offline test proves the parsing without the network.

---

_Done: H1 (loop), H2 (tools), H3 (context), H4 (sessions), H5 (security),
H6 (CLI)._
