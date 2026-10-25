"""Environment-driven configuration for the harness.

Everything tunable lives here so no module reads ``os.environ`` directly:
the model endpoint, the token budget, the loop guard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

# The dev endpoint is a reasoning model whose prompt + output must stay under 50k.
DEFAULT_CONTEXT_WINDOW = 50_000
DEFAULT_MAX_TOKENS = 4_096
DEFAULT_REQUEST_TIMEOUT = 600.0
DEFAULT_MAX_ITERATIONS = 12
DEFAULT_DB_PATH = "sessions.db"
DEFAULT_MAX_TOOL_CALLS = 12


class ConfigError(RuntimeError):
    """A required setting is missing or nonsensical."""


def load_env_file(path: str | Path = ".env", *, override: bool = False) -> None:
    """Load simple ``KEY=VALUE`` lines into ``os.environ``.

    Deliberately small: comments, blank lines, an optional ``export`` prefix and
    matching surrounding quotes. Anything else is an error rather than a guess.
    Existing variables win unless ``override`` is set.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return
    for lineno, raw in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"{file_path}:{lineno}: expected KEY=VALUE")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if override or key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    context_window: int = DEFAULT_CONTEXT_WINDOW
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    temperature: float | None = None
    db_path: str = DEFAULT_DB_PATH
    allowed_tools: frozenset[str] | None = None
    max_tool_calls_per_turn: int = DEFAULT_MAX_TOOL_CALLS

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ConfigError("LLM_MAX_TOKENS must be positive")
        if self.max_tokens >= self.context_window:
            raise ConfigError(
                f"LLM_MAX_TOKENS ({self.max_tokens}) must leave room for the prompt "
                f"inside HARNESS_CONTEXT_WINDOW ({self.context_window})"
            )
        if self.max_iterations <= 0:
            raise ConfigError("HARNESS_MAX_ITERATIONS must be positive")
        if self.max_tool_calls_per_turn <= 0:
            raise ConfigError("HARNESS_MAX_TOOL_CALLS must be positive")

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        load_dotenv: bool = True,
    ) -> "Config":
        """Build a config from the environment (and ``.env``, if present)."""
        if env is None and load_dotenv:
            load_env_file()
        src: Mapping[str, str] = os.environ if env is None else env
        return cls(
            base_url=_required(src, "LLM_BASE_URL").rstrip("/"),
            api_key=_required(src, "LLM_API_KEY"),
            model=_required(src, "LLM_MODEL"),
            max_tokens=_integer(src, "LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            context_window=_integer(src, "HARNESS_CONTEXT_WINDOW", DEFAULT_CONTEXT_WINDOW),
            request_timeout=_number(src, "LLM_TIMEOUT", DEFAULT_REQUEST_TIMEOUT),
            max_iterations=_integer(src, "HARNESS_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS),
            temperature=_optional_number(src, "LLM_TEMPERATURE"),
            db_path=src.get("HARNESS_DB", "").strip() or DEFAULT_DB_PATH,
            allowed_tools=_tools(src),
            max_tool_calls_per_turn=_integer(src, "HARNESS_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS),
        )


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(f"{key} is not set (see .env.example)")
    return value


def _integer(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc


def _number(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc


def _tools(env: Mapping[str, str]) -> frozenset[str] | None:
    raw = env.get("HARNESS_ALLOWED_TOOLS", "").strip()
    if not raw:
        return None
    names = {name.strip() for name in raw.split(",") if name.strip()}
    return frozenset(names) or None


def _optional_number(env: Mapping[str, str], key: str) -> float | None:
    raw = env.get(key, "").strip()
    if not raw:
        return None
    return _number(env, key, 0.0)
