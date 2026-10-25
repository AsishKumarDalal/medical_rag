-- Harness session store. Applied on every open; all statements are idempotent.

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    model       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn         INTEGER NOT NULL,
    position     INTEGER NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content      TEXT NOT NULL,
    tool_calls   TEXT,
    tool_call_id TEXT,
    name         TEXT,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id, turn, position);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn        INTEGER NOT NULL,
    call_id     TEXT NOT NULL,
    name        TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    result      TEXT,
    ok          INTEGER,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_session
    ON tool_calls (session_id, turn);

CREATE TABLE IF NOT EXISTS context_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn            INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    before_tokens   INTEGER NOT NULL,
    after_tokens    INTEGER NOT NULL,
    messages_before INTEGER NOT NULL,
    messages_after  INTEGER NOT NULL,
    detail          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn              INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);
