from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import aiosqlite


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    telegram_user_id INTEGER PRIMARY KEY,
    username TEXT,
    display_name TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'normal',
    compliance_score INTEGER NOT NULL DEFAULT 0,
    conversation_count INTEGER NOT NULL DEFAULT 0,
    next_task_turn INTEGER NOT NULL DEFAULT 6,
    aftercare_until TEXT,
    paused_reason TEXT,
    dislikes TEXT NOT NULL DEFAULT '[]',
    hard_limits TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '[]',
    last_model_response_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    telegram_user_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    instructions TEXT NOT NULL,
    status TEXT NOT NULL,
    intensity TEXT NOT NULL DEFAULT 'normal',
    created_at TEXT NOT NULL,
    due_at TEXT,
    issued_at_turn INTEGER,
    completed_at TEXT,
    skipped_at TEXT,
    source TEXT NOT NULL DEFAULT 'model',
    metadata TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (telegram_user_id) REFERENCES users (telegram_user_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_status
ON tasks (telegram_user_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS conversation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    telegram_user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    message_kind TEXT NOT NULL DEFAULT 'text',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (telegram_user_id) REFERENCES users (telegram_user_id)
);

CREATE INDEX IF NOT EXISTS idx_conversation_events_user
ON conversation_events (telegram_user_id, created_at DESC);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self.path.as_posix())
        connection.row_factory = aiosqlite.Row
        try:
            yield connection
        finally:
            await connection.close()

    async def initialize(self) -> None:
        async with self.connect() as connection:
            await connection.executescript(SCHEMA)
            await connection.commit()

    async def execute(self, query: str, parameters: Iterable[Any] = ()) -> None:
        async with self.connect() as connection:
            await connection.execute(query, tuple(parameters))
            await connection.commit()

    async def fetchone(self, query: str, parameters: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with self.connect() as connection:
            cursor = await connection.execute(query, tuple(parameters))
            return await cursor.fetchone()

    async def fetchall(self, query: str, parameters: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with self.connect() as connection:
            cursor = await connection.execute(query, tuple(parameters))
            return await cursor.fetchall()

