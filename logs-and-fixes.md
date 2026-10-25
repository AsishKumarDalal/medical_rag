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
