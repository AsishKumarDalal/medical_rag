"""The only module in the harness that talks to a model provider.

Everything else depends on :class:`SupportsComplete`, so tests and later phases
can substitute a stub without touching the loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .config import Config
from .messages import Message, ToolCall
from .security import redact


class LLMError(RuntimeError):
    """A model call failed, after the client's retries."""


@dataclass(slots=True)
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    model: str | None = None


class SupportsComplete(Protocol):
    """What the loop needs from a model client."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class LLMClient:
    """OpenAI-compatible chat client with tool calling and bounded retries."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        sdk: Any | None = None,
        retries: int = 2,
        backoff: float = 2.0,
    ) -> None:
        self.config = config or Config.from_env()
        self.retries = max(0, retries)
        self.backoff = max(0.0, backoff)
        if sdk is None:
            from openai import OpenAI  # imported here so tests never need the SDK

            sdk = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.request_timeout,
            )
        self._sdk = sdk

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [message.to_provider() for message in messages],
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            request["temperature"] = self.config.temperature
        if tools:
            request["tools"] = list(tools)
            request["tool_choice"] = "auto"

        attempts = self.retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._sdk.chat.completions.create(**request)
                return self._to_llm_response(response)
            except Exception as exc:  # provider/network/parse; retried, then reported
                last_error = exc
                if attempt < attempts and self.backoff:
                    time.sleep(self.backoff * attempt)
        raise LLMError(redact(f"{type(last_error).__name__}: {last_error}")) from last_error

    def _to_llm_response(self, response: Any) -> LLMResponse:
        choices = _field(response, "choices") or []
        if not choices:
            raise LLMError("model returned no choices")
        choice = choices[0]
        message = _field(choice, "message")
        if message is None:
            raise LLMError("model returned a choice without a message")
        calls = [
            ToolCall.from_provider(_as_dict(call))
            for call in (_field(message, "tool_calls") or [])
        ]
        usage = _field(response, "usage")
        return LLMResponse(
            content=_field(message, "content") or "",
            tool_calls=calls,
            finish_reason=_field(choice, "finish_reason"),
            usage=_usage_dict(usage),
            model=_field(response, "model"),
        )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Normalize an SDK object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    return dict(vars(obj))


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = _field(usage, key)
        if isinstance(value, int):
            result[key] = value
    return result
