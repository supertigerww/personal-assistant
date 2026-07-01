from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.models import ConversationState, UserProfile


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class UserService:
    def __init__(self, *, database: Any, settings: Any) -> None:
        self.database = database
        self.settings = settings

    async def get_or_create(
        self,
        *,
        telegram_user_id: int,
        username: str | None,
        display_name: str,
    ) -> UserProfile:
        row = await self.database.fetchone(
            "SELECT * FROM users WHERE telegram_user_id = ?",
            (telegram_user_id,),
        )
        if row is not None:
            return self._row_to_profile(row)

        now = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO users (
                telegram_user_id, username, display_name, state, compliance_score,
                conversation_count, next_task_turn, aftercare_until, paused_reason,
                dislikes, hard_limits, notes, last_model_response_at, created_at, updated_at
            )
            VALUES (?, ?, ?, 'normal', 0, 0, ?, NULL, NULL, '[]', '[]', '[]', NULL, ?, ?)
            """,
            (
                telegram_user_id,
                username,
                display_name,
                self.settings.task_normal_min_turns,
                now,
                now,
            ),
        )
        return await self.get_profile(telegram_user_id)

    async def get_profile(self, telegram_user_id: int) -> UserProfile:
        row = await self.database.fetchone(
            "SELECT * FROM users WHERE telegram_user_id = ?",
            (telegram_user_id,),
        )
        if row is None:
            raise LookupError(f"Unknown user {telegram_user_id}")
        return self._row_to_profile(row)

    async def increment_conversation_count(self, telegram_user_id: int) -> UserProfile:
        await self.database.execute(
            """
            UPDATE users
            SET conversation_count = conversation_count + 1, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (utc_now_iso(), telegram_user_id),
        )
        return await self.get_profile(telegram_user_id)

    async def update_state(
        self,
        telegram_user_id: int,
        state: str,
        *,
        paused_reason: str | None,
        aftercare_until: str | None = None,
    ) -> None:
        await self.database.execute(
            """
            UPDATE users
            SET state = ?, paused_reason = ?, aftercare_until = ?, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (state, paused_reason, aftercare_until, utc_now_iso(), telegram_user_id),
        )

    async def update_next_task_turn(self, telegram_user_id: int, next_task_turn: int) -> None:
        await self.database.execute(
            """
            UPDATE users
            SET next_task_turn = ?, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (next_task_turn, utc_now_iso(), telegram_user_id),
        )

    async def append_dislikes(self, telegram_user_id: int, dislikes: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "dislikes", dislikes)

    async def append_hard_limits(self, telegram_user_id: int, hard_limits: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "hard_limits", hard_limits)

    async def append_notes(self, telegram_user_id: int, notes: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "notes", notes)

    async def _append_unique_json_list(self, telegram_user_id: int, column: str, values: list[str]) -> None:
        if not values:
            return

        profile = await self.get_profile(telegram_user_id)
        existing = list(getattr(profile, column))
        merged: list[str] = existing[:]
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)

        await self.database.execute(
            f"UPDATE users SET {column} = ?, updated_at = ? WHERE telegram_user_id = ?",
            (json.dumps(merged, ensure_ascii=False), utc_now_iso(), telegram_user_id),
        )

    @staticmethod
    def _row_to_profile(row: Any) -> UserProfile:
        return UserProfile(
            telegram_user_id=row["telegram_user_id"],
            username=row["username"],
            display_name=row["display_name"],
            state=ConversationState(row["state"]),
            compliance_score=row["compliance_score"],
            conversation_count=row["conversation_count"],
            next_task_turn=row["next_task_turn"],
            aftercare_until=row["aftercare_until"],
            paused_reason=row["paused_reason"],
            dislikes=json.loads(row["dislikes"] or "[]"),
            hard_limits=json.loads(row["hard_limits"] or "[]"),
            notes=json.loads(row["notes"] or "[]"),
            last_model_response_at=row["last_model_response_at"],
        )

