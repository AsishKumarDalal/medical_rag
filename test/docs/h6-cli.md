# H6 demo — the terminal, live

Run against the dev endpoint with a script piped into stdin (one model call):

```bash
printf 'In one short sentence, what can you do?\n/usage\n/exit\n' \
  | .venv/bin/python -m harness --db /tmp/harness-demo.db
```

```text
medical harness — session 389e66761a96 — /help for commands
you>
I can search indexed medical literature and answer health questions with citations to the supporting evidence.
[usage] prompt=458 completion=126 total=584 calls=1
you> prompt=458 completion=126 total=584 calls=1
you> 
```

## What this shows

- The system prompt, the session store and the model client work together from a
  cold start: the session is created, the system message is written, the turn is
  persisted, usage is recorded and `/usage` reads it back.
- Ctrl-C / EOF ends cleanly (the second `you>` is the EOF exit).
- Only one model call was made; no tool was called because the question was not
  medical.

## Not shown here

A turn that calls `search_medical_literature` — the pipeline arrives in G1–G3.
The tool currently answers "the retrieval pipeline is not wired yet (phase G3)".
