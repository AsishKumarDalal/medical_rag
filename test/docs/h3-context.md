# H3 demo — both context passes

Offline and deterministic (`test/context_demo.py`); the summarizer is stubbed.

## Stub pass at 50% of budget

```text
conversation: 20 messages, ~5396 tokens
stub: 20 -> 20 messages, 5396 -> 2376 tokens | stubbed 5 earlier tool outputs (~3020 tokens)
request tokens after stubbing: ~2376
first 3 untouched: True
last 6 untouched: True
canonical keeps full text: 2579 chars
```

Five older, large tool outputs were replaced for the request only. The canonical
conversation still holds the full 2,579-character result, and the newest tool
outputs plus the first three and last six messages were not touched.

## Compress pass at 80% of budget

```text
compress: 20 -> 11 messages, 5396 -> 2082 tokens | summarized 10 messages into 29 characters
summary inserted at index 3: True
first 3 untouched: True
last 6 untouched: True
```

The middle 10 messages became one summary message; 20 messages became 11. The
stub pass did not fire afterwards because the conversation dropped to ~31% of the
budget.

## Notes

- Token counts are the harness's own estimate (4 characters ≈ 1 token), not the
  provider's tokenizer. The budget is deliberately conservative.
- The tail can be longer than six messages when the boundary would split an
  assistant tool call from its results.
