from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryService:
    def __init__(self, *, database: Any) -> None:
        self.database = database

    async def store_message(
        self,
        telegram_user_id: int,
        role: str,
        content: str,
        *,
        message_kind: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO conversation_events (
                telegram_user_id, role, content, message_kind, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                telegram_user_id,
                role,
                content,
                message_kind,
                json.dumps(metadata or {}, ensure_ascii=False),
                utc_now_iso(),
            ),
        )

    async def recent_messages(self, telegram_user_id: int, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = await self.database.fetchall(
            """
            SELECT role, content, message_kind, metadata, created_at
            FROM conversation_events
            WHERE telegram_user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (telegram_user_id, limit),
        )
        events = [
            {
                "role": row["role"],
                "content": row["content"],
                "message_kind": row["message_kind"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return list(reversed(events))

