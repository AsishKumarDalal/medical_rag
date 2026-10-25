# Harness phase records

One short record per completed harness phase, in the style of the run notes under
`test/docs/`. They state what was built, how it was verified, and what is explicitly
out of scope.

| Phase | Record | Main proof |
|---|---|---|
| H1 | [h1.md](h1.md) | live tool round-trip (`test/docs/smoke-h1.md`) |
| H2 | [h2.md](h2.md) | tool validation/containment tests |
| H3 | [h3.md](h3.md) | both context passes (`test/docs/h3-context.md`) |
| H4 | [h4.md](h4.md) | create/reload/resume tests |
| H5 | [h5.md](h5.md) | injection, budget, redaction tests |
| H6 | [h6.md](h6.md) | scripted live CLI (`test/docs/h6-cli.md`) |
