"""The only module in the GraphRAG package that talks to a model provider.

Two capabilities sit behind one object:

* ``complete_text`` / ``complete_json`` — chat completions for extraction,
  merging, summaries, routing and narration.
* ``embed`` — an OpenAI-compatible embeddings route when one is configured. The
  chat endpoint in development does not serve embeddings, so callers may fall
  back to the deterministic embedder in ``graphrag.embed``.

Every call is retried with backoff, every response is normalised, and token usage
is counted for the evaluation phase. Parsed JSON is still a *suggestion*: callers
validate it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from harness.config import Config
from harness.security import redact


class GraphRagLLMError(RuntimeError):
    """A model call failed after its retries, or returned unusable output."""


@dataclass(slots=True)
class Completion:
    text: str
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class CallStats:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0

    def record(self, completion: Completion, elapsed: float) -> None:
        self.calls += 1
        self.seconds += elapsed
        usage = completion.usage
        self.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.completion_tokens += int(usage.get("completion_tokens", 0))
        self.total_tokens += int(usage.get("total_tokens", 0))

    def to_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 3),
        }


class GraphLLM:
    """OpenAI-compatible chat client with retries, JSON salvage and stats."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        sdk: Any | None = None,
        embed_sdk: Any | None = None,
        embed_model: str | None = None,
        embed_base_url: str | None = None,
        embed_api_key: str | None = None,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> None:
        self.config = config or Config.from_env()
        self.retries = max(0, retries)
        self.backoff = max(0.0, backoff)
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key
        self.stats = CallStats()
        if sdk is None:
            from openai import OpenAI  # the one provider import in this package

            sdk = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.request_timeout,
            )
        self._sdk = sdk
        self._embed_sdk = embed_sdk

    # -- chat -------------------------------------------------------------

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        completion = self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = completion.text.strip()
        if not text:
            raise GraphRagLLMError(
                f"model returned an empty answer (finish_reason={completion.finish_reason!r})"
            )
        return text

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        attempts: int = 2,
        expect: type | tuple[type, ...] = (dict, list),
        retry_note: str | None = None,
    ) -> Any:
        """Ask for JSON and salvage what the model actually sent.

        A parse failure gets one corrective retry with the same request plus a
        blunt instruction. After that the error is raised so the caller can
        checkpoint a permanent failure instead of silently losing a chunk.
        """
        attempts = max(1, attempts)
        last_error: Exception | None = None
        cap = max_tokens or self.config.max_tokens
        for attempt in range(1, attempts + 1):
            prompt = user
            if attempt > 1:
                prompt = (
                    user
                    + "\n\nYour previous answer was not valid JSON (or was cut off). "
                    "Return a SHORTER, complete JSON object now, keeping the requested "
                    "schema: double quotes only, no commentary."
                    + (f" {retry_note}" if retry_note else "")
                )
            completion = self.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=cap,
                temperature=0.0,
            )
            try:
                parsed = parse_json(completion.text)
            except ValueError as exc:
                last_error = exc
                continue
            if isinstance(parsed, expect):
                return parsed
            last_error = GraphRagLLMError(
                f"expected {'/'.join(t.__name__ for t in expect)}, got {type(parsed).__name__}"
            )
        raise GraphRagLLMError(f"model did not return usable JSON: {last_error}")

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "max_tokens": max_tokens or self.config.max_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        elif self.config.temperature is not None:
            request["temperature"] = self.config.temperature

        attempts = self.retries + 1
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(1, attempts + 1):
            try:
                response = self._sdk.chat.completions.create(**request)
                completion = _to_completion(response)
                self.stats.record(completion, time.monotonic() - started)
                return completion
            except Exception as exc:  # provider/network; retried, then reported
                last_error = exc
                if attempt < attempts and self.backoff:
                    time.sleep(self.backoff * attempt)
        self.stats.failures += 1
        self.stats.seconds += time.monotonic() - started
        raise GraphRagLLMError(redact(f"{type(last_error).__name__}: {last_error}")) from last_error

    # -- embeddings -------------------------------------------------------

    @property
    def embeddings_configured(self) -> bool:
        return bool(self.embed_model)

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> list[list[float]]:
        if not self.embed_model:
            raise GraphRagLLMError("no embedding model configured (GRAPHRAG_EMBED_MODEL)")
        client = self._embed_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), max(1, batch_size)):
            batch = list(texts[start : start + max(1, batch_size)])
            started = time.monotonic()
            try:
                response = client.embeddings.create(model=self.embed_model, input=batch)
            except Exception as exc:
                self.stats.failures += 1
                raise GraphRagLLMError(redact(f"embedding request failed: {exc}")) from exc
            ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            for item in ordered:
                vectors.append([float(value) for value in item.embedding])
            self.stats.seconds += time.monotonic() - started
        if len(vectors) != len(texts):
            raise GraphRagLLMError(
                f"embedding endpoint returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    def _embed_client(self) -> Any:
        if self._embed_sdk is None:
            from openai import OpenAI

            self._embed_sdk = OpenAI(
                base_url=self.embed_base_url or self.config.base_url,
                api_key=self.embed_api_key or self.config.api_key,
                timeout=self.config.request_timeout,
            )
        return self._embed_sdk

    def reset_stats(self) -> None:
        self.stats = CallStats()


def parse_json(text: str) -> Any:
    """Parse model JSON, tolerating fences, prose and trailing commas."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    attempts = [cleaned]
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if starts:
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            attempts.append(cleaned[start : end + 1])
    for candidate in attempts:
        for variant in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                return json.loads(variant)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"not valid JSON: {cleaned[:200]!r}")


def _to_completion(response: Any) -> Completion:
    choices = _field(response, "choices") or []
    if not choices:
        raise GraphRagLLMError("model returned no choices")
    choice = choices[0]
    message = _field(choice, "message")
    if message is None:
        raise GraphRagLLMError("model returned a choice without a message")
    return Completion(
        text=_field(message, "content") or "",
        finish_reason=_field(choice, "finish_reason"),
        model=_field(response, "model"),
        usage=_usage_dict(_field(response, "usage")),
    )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(usage, key)
        if isinstance(value, int):
            result[key] = value
    return result
