# H1 smoke run — live tool round-trip

Recorded proof that the loop calls a real model, executes a tool and returns a final
answer. Not a unit test: it needs the dev endpoint.

- **Date:** 2026-10-25
- **Command:** `.venv/bin/python test/smoke_h1.py`
- **Model:** `k2-horizon-mova-36b-a4b`
- **Endpoint:** OpenAI-compatible dev tunnel (`LLM_BASE_URL` in `.env`)
- **Result:** `stop_reason=answer`, 2 model calls, exit code 0

## Transcript

```text
model=k2-horizon-mova-36b-a4b base_url=https://<dev-tunnel>/v1
[system] You are a tool-use test harness. When asked to echo a phrase, call the echo tool exactly once with that phrase, then reply with the exact text the tool returned and nothing else.
[user] Echo the phrase 'loop is alive', then tell me exactly what came back.
[assistant] tool_call echo({"text":"loop is alive"})
[tool:echo] loop is alive
[assistant] loop is alive
--- stop_reason=answer iterations=2
```

## What this proves

1. The endpoint accepts `tools=` and returns a structured `tool_calls` payload — the
   model is usable for the harness, not just plain chat.
2. Tool arguments arrive as JSON and are parsed into a dict (`text` → `"loop is alive"`).
3. The tool result is sent back as a `role: "tool"` message with its `tool_call_id`,
   and the model then answers.
4. The loop terminates on a message with no tool calls, reporting `iterations=2`.

## What it does not prove

Nothing about context compression (H3), persistence (H4), or security (H5). Token
counts from this run were not inspected; H3 will log them.
