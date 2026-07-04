from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

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
                conversation_count, next_task_turn, next_photo_task_turn, next_video_turn, aftercare_until, paused_reason,
                dislikes, hard_limits, notes, onboarding_completed, last_model_response_at, created_at, updated_at
            )
            VALUES (?, ?, ?, 'normal', 0, 0, ?, ?, ?, NULL, NULL, '[]', '[]', '[]', 0, NULL, ?, ?)
            """,
            (
                telegram_user_id,
                username,
                display_name,
                self.settings.task_normal_min_turns,
                self.settings.photo_task_normal_min_turns,
                self.settings.video_normal_min_turns,
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

    async def update_next_photo_task_turn(self, telegram_user_id: int, next_photo_task_turn: int) -> None:
        await self.database.execute(
            """
            UPDATE users
            SET next_photo_task_turn = ?, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (next_photo_task_turn, utc_now_iso(), telegram_user_id),
        )

    async def update_next_video_turn(self, telegram_user_id: int, next_video_turn: int) -> None:
        await self.database.execute(
            """
            UPDATE users
            SET next_video_turn = ?, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (next_video_turn, utc_now_iso(), telegram_user_id),
        )

    async def append_dislikes(self, telegram_user_id: int, dislikes: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "dislikes", dislikes)

    async def append_hard_limits(self, telegram_user_id: int, hard_limits: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "hard_limits", hard_limits)

    async def append_notes(self, telegram_user_id: int, notes: list[str]) -> None:
        await self._append_unique_json_list(telegram_user_id, "notes", notes)

    async def mark_onboarding_completed(self, telegram_user_id: int) -> None:
        await self.database.execute(
            """
            UPDATE users
            SET onboarding_completed = 1, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (utc_now_iso(), telegram_user_id),
        )

    async def adjust_compliance_score(self, telegram_user_id: int, delta: int) -> UserProfile:
        profile = await self.get_profile(telegram_user_id)
        updated_score = max(0, min(100, profile.compliance_score + int(delta)))
        await self.database.execute(
            """
            UPDATE users
            SET compliance_score = ?, updated_at = ?
            WHERE telegram_user_id = ?
            """,
            (updated_score, utc_now_iso(), telegram_user_id),
        )
        profile = await self.get_profile(telegram_user_id)
        return await self.sync_runtime_state(profile)

    async def sync_runtime_state(self, profile: UserProfile) -> UserProfile:
        profile = await self._expire_aftercare_if_needed(profile)
        return await self._sync_intensity_from_compliance(profile)

    async def _expire_aftercare_if_needed(self, profile: UserProfile) -> UserProfile:
        if profile.state != ConversationState.AFTERCARE or not profile.aftercare_until:
            return profile

        until = datetime.fromisoformat(profile.aftercare_until)
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) < until:
            return profile

        logger.info("Aftercare expired for user %s; returning to normal.", profile.telegram_user_id)
        await self.update_state(
            profile.telegram_user_id,
            ConversationState.NORMAL,
            paused_reason=None,
            aftercare_until=None,
        )
        return await self.get_profile(profile.telegram_user_id)

    async def _sync_intensity_from_compliance(self, profile: UserProfile) -> UserProfile:
        if profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return profile

        enter_threshold = int(self.settings.intense_enter_compliance_score)
        exit_threshold = int(self.settings.intense_exit_compliance_score)

        if profile.state == ConversationState.NORMAL and profile.compliance_score >= enter_threshold:
            logger.info(
                "Promoting user %s to intense compliance_score=%s threshold=%s",
                profile.telegram_user_id,
                profile.compliance_score,
                enter_threshold,
            )
            await self.update_state(
                profile.telegram_user_id,
                ConversationState.INTENSE,
                paused_reason=None,
                aftercare_until=None,
            )
            return await self.get_profile(profile.telegram_user_id)

        if profile.state == ConversationState.INTENSE and profile.compliance_score <= exit_threshold:
            logger.info(
                "Demoting user %s to normal compliance_score=%s threshold=%s",
                profile.telegram_user_id,
                profile.compliance_score,
                exit_threshold,
            )
            await self.update_state(
                profile.telegram_user_id,
                ConversationState.NORMAL,
                paused_reason=None,
                aftercare_until=None,
            )
            return await self.get_profile(profile.telegram_user_id)

        return profile

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
            next_photo_task_turn=row["next_photo_task_turn"] if "next_photo_task_turn" in row.keys() else 0,
            next_video_turn=row["next_video_turn"] if "next_video_turn" in row.keys() else 18,
            aftercare_until=row["aftercare_until"],
            paused_reason=row["paused_reason"],
            dislikes=json.loads(row["dislikes"] or "[]"),
            hard_limits=json.loads(row["hard_limits"] or "[]"),
            notes=json.loads(row["notes"] or "[]"),
            onboarding_completed=bool(row["onboarding_completed"]) if "onboarding_completed" in row.keys() else True,
            last_model_response_at=row["last_model_response_at"],
        )

