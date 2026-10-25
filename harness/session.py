"""Durable sessions.

Conversations, tool calls, context events and token usage live in one SQLite file.
Writes happen as the turn runs, so a crash costs at most the step in progress.

The store never imports the loop; the loop talks to a :class:`SessionRecorder`,
which is the small interface below.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .context import ContextEvent
from .messages import Message, ToolCall

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = "sessions.db"


class SessionError(RuntimeError):
    """The session store was used in a way it cannot honour."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(slots=True)
class SessionInfo:
    id: str
    title: str | None
    model: str | None
    created_at: str
    updated_at: str
    messages: int = 0


class SessionStore:
    """One SQLite file for every session."""

    def __init__(
        self,
        path: str | Path = DEFAULT_DB_PATH,
        *,
        sanitize: Callable[[str], str] | None = None,
    ) -> None:
        self.path = str(path)
        self.sanitize = sanitize or (lambda text: text)
        try:
            self._db = sqlite3.connect(self.path)
        except sqlite3.Error as exc:
            raise SessionError(f"cannot open {self.path}: {exc}") from exc
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self._db.commit()

    # -- sessions ---------------------------------------------------------

    def create(
        self,
        *,
        title: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
    ) -> str:
        identifier = session_id or uuid.uuid4().hex[:12]
        now = _now()
        try:
            self._db.execute(
                "INSERT INTO sessions (id, title, model, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (identifier, title, model, now, now),
            )
            self._db.commit()
        except sqlite3.IntegrityError as exc:
            raise SessionError(f"session {identifier!r} already exists") from exc
        return identifier

    def get(self, session_id: str) -> SessionInfo | None:
        row = self._db.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages "
            "FROM sessions s WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        return _to_info(row) if row else None

    def sessions(self, limit: int = 20) -> list[SessionInfo]:
        rows = self._db.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages "
            "FROM sessions s ORDER BY s.updated_at DESC, s.id LIMIT ?",
            (limit,),
        ).fetchall()
        return [_to_info(row) for row in rows]

    def delete(self, session_id: str) -> bool:
        cursor = self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._db.commit()
        return cursor.rowcount > 0

    def rename(self, session_id: str, title: str) -> None:
        self._db.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), session_id),
        )
        self._db.commit()

    # -- messages ---------------------------------------------------------

    def append_message(self, session_id: str, message: Message, *, turn: int = 1) -> int:
        """Append one message and return its position in the session."""
        self._require_session(session_id)
        position = self._next_position(session_id)
        content = self.sanitize(message.content)
        tool_calls = (
            json.dumps([call.to_provider() for call in message.tool_calls])
            if message.tool_calls
            else None
        )
        self._db.execute(
            "INSERT INTO messages "
            "(session_id, turn, position, role, content, tool_calls, tool_call_id, name, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                position,
                message.role,
                content,
                tool_calls,
                message.tool_call_id,
                message.name,
                _now(),
            ),
        )
        for call in message.tool_calls:
            self._db.execute(
                "INSERT INTO tool_calls (session_id, turn, call_id, name, arguments, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    turn,
                    call.id,
                    call.name,
                    self.sanitize(call.raw_arguments or json.dumps(call.arguments)),
                    _now(),
                ),
            )
        if message.role == "tool" and message.tool_call_id:
            self._db.execute(
                "UPDATE tool_calls SET result = ?, ok = ? "
                "WHERE session_id = ? AND call_id = ? AND result IS NULL",
                (content, 1, session_id, message.tool_call_id),
            )
        self._touch(session_id)
        self._db.commit()
        return position

    def append_messages(self, session_id: str, messages: Iterable[Message], *, turn: int = 1) -> None:
        for message in messages:
            self.append_message(session_id, message, turn=turn)

    def load_messages(self, session_id: str) -> list[Message]:
        rows = self._db.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY turn, position",
            (session_id,),
        ).fetchall()
        return [_to_message(row) for row in rows]

    def tool_log(self, session_id: str) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT call_id, name, arguments, result, ok FROM tool_calls "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    # -- events and usage -------------------------------------------------

    def record_context_event(self, session_id: str, event: ContextEvent, *, turn: int = 1) -> None:
        self._require_session(session_id)
        self._db.execute(
            "INSERT INTO context_events "
            "(session_id, turn, kind, before_tokens, after_tokens, messages_before, "
            " messages_after, detail, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                event.kind,
                event.before_tokens,
                event.after_tokens,
                event.messages_before,
                event.messages_after,
                event.detail,
                _now(),
            ),
        )
        self._touch(session_id)
        self._db.commit()

    def context_events(self, session_id: str) -> list[dict[str, object]]:
        rows = self._db.execute(
            "SELECT kind, before_tokens, after_tokens, messages_before, messages_after, detail "
            "FROM context_events WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def record_usage(self, session_id: str, usage: dict[str, int], *, turn: int = 1) -> None:
        if not usage:
            return
        self._require_session(session_id)
        self._db.execute(
            "INSERT INTO usage (session_id, turn, prompt_tokens, completion_tokens, total_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                turn,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                int(usage.get("total_tokens", 0)),
                _now(),
            ),
        )
        self._db.commit()

    def usage_totals(self, session_id: str) -> dict[str, int]:
        row = self._db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
            "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
            "COALESCE(SUM(total_tokens), 0) AS total_tokens, COUNT(*) AS calls "
            "FROM usage WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return {key: int(row[key]) for key in row.keys()}  # type: ignore[union-attr]

    # -- recorder ---------------------------------------------------------

    def recorder(self, session_id: str, *, turn: int) -> "SessionRecorder":
        return SessionRecorder(self, session_id, turn)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------

    def _require_session(self, session_id: str) -> None:
        row = self._db.execute(
            "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionError(f"unknown session {session_id!r}")

    def _next_position(self, session_id: str) -> int:
        row = self._db.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 AS next FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["next"])

    def _touch(self, session_id: str) -> None:
        row = self._db.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionError(f"unknown session {session_id!r}")
        self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?", (_now(), session_id)
        )


class SessionRecorder:
    """The loop's write path into one session turn."""

    def __init__(self, store: SessionStore, session_id: str, turn: int) -> None:
        self.store = store
        self.session_id = session_id
        self.turn = turn

    def message(self, message: Message) -> None:
        self.store.append_message(self.session_id, message, turn=self.turn)

    def context_event(self, event: ContextEvent) -> None:
        self.store.record_context_event(self.session_id, event, turn=self.turn)

    def usage(self, usage: dict[str, int]) -> None:
        self.store.record_usage(self.session_id, usage, turn=self.turn)


def _to_info(row: sqlite3.Row) -> SessionInfo:
    return SessionInfo(
        id=row["id"],
        title=row["title"],
        model=row["model"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=int(row["messages"]),
    )


def _to_message(row: sqlite3.Row) -> Message:
    calls: list[ToolCall] = []
    if row["tool_calls"]:
        calls = [ToolCall.from_provider(data) for data in json.loads(row["tool_calls"])]
    return Message(
        role=row["role"],
        content=row["content"],
        tool_calls=calls,
        tool_call_id=row["tool_call_id"],
        name=row["name"],
    )
