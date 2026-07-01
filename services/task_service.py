from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from core.models import ConversationState, Task, TaskIntensity, TaskStatus, UserProfile

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskService:
    ACK_KEYWORDS = (
        "done",
        "completed",
        "finished",
        "got it",
        "roger",
        "ok",
        "okay",
        "收到",
        "完成",
        "做了",
        "做完",
        "马上",
        "好的",
        "嗯",
        "行",
        "可以",
    )

    def __init__(self, *, database: Any, settings: Any, user_service: Any) -> None:
        self.database = database
        self.settings = settings
        self.user_service = user_service

    async def ensure_schedule(self, profile: UserProfile) -> None:
        if profile.next_task_turn > 0:
            return
        await self.schedule_next_task(
            telegram_user_id=profile.telegram_user_id,
            state=str(profile.state),
            from_turn=profile.conversation_count,
        )

    async def schedule_next_task(self, *, telegram_user_id: int, state: str, from_turn: int) -> int:
        interval = self._pick_interval(state)
        next_turn = from_turn + interval
        await self.user_service.update_next_task_turn(telegram_user_id, next_turn)
        return next_turn

    def can_issue_now(self, profile: UserProfile) -> bool:
        if profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return False
        return profile.conversation_count >= profile.next_task_turn

    async def create_task(
        self,
        *,
        telegram_user_id: int,
        title: str,
        instructions: str,
        intensity: str = "normal",
        due_at: str | None = None,
        issued_at_turn: int | None = None,
        source: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        cleaned_title = title.strip()
        cleaned_instructions = instructions.strip()
        task_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        await self.database.execute(
            """
            INSERT INTO tasks (
                id, telegram_user_id, title, instructions, status, intensity,
                created_at, due_at, issued_at_turn, completed_at, skipped_at, source, metadata
            )
            VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                task_id,
                telegram_user_id,
                cleaned_title,
                cleaned_instructions,
                intensity,
                created_at,
                due_at,
                issued_at_turn,
                source,
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        logger.info(f"Task created for user {telegram_user_id}: {cleaned_title}")
        return await self.get_task(task_id)

    async def get_task(self, task_id: str) -> Task:
        row = await self.database.fetchone("SELECT * FROM tasks WHERE id = ?", (task_id,))
        if row is None:
            raise LookupError(f"Unknown task {task_id}")
        return self._row_to_task(row)

    async def get_open_task(self, telegram_user_id: int) -> Task | None:
        row = await self.database.fetchone(
            """
            SELECT * FROM tasks
            WHERE telegram_user_id = ? AND status = 'open'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (telegram_user_id,),
        )
        return self._row_to_task(row) if row is not None else None

    async def list_recent_tasks(self, telegram_user_id: int, limit: int = 10) -> list[Task]:
        rows = await self.database.fetchall(
            """
            SELECT * FROM tasks
            WHERE telegram_user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (telegram_user_id, limit),
        )
        return [self._row_to_task(row) for row in rows]

    async def complete_task(self, task_id: str) -> None:
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'completed', completed_at = ?
            WHERE id = ?
            """,
            (utc_now_iso(), task_id),
        )

    async def pause_all_tasks(self, telegram_user_id: int, *, reason: str) -> None:
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'skipped', skipped_at = ?
            WHERE telegram_user_id = ? AND status = 'open'
            """,
            (utc_now_iso(), telegram_user_id),
        )
        logger.info(f"Tasks paused for user {telegram_user_id}, reason: {reason}")

    async def skip_ignored_task_if_needed(
        self,
        *,
        telegram_user_id: int,
        current_turn: int,
        user_text: str,
    ) -> Task | None:
        task = await self.get_open_task(telegram_user_id)
        if task is None or task.issued_at_turn is None:
            return None
        if current_turn != task.issued_at_turn + 1:
            return None
        if self.looks_like_task_response(user_text):
            return None

        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'skipped', skipped_at = ?
            WHERE id = ?
            """,
            (utc_now_iso(), task.id),
        )
        logger.info(f"Task skipped for user {telegram_user_id}: {task.title}")
        return await self.get_task(task.id)

    @classmethod
    def looks_like_task_response(cls, user_text: str) -> bool:
        normalized = user_text.casefold()
        return any(keyword in normalized for keyword in cls.ACK_KEYWORDS)

    def _pick_interval(self, state: str) -> int:
        if state == ConversationState.INTENSE:
            lower_bound = max(5, int(self.settings.task_intense_min_turns))
            upper_bound = max(lower_bound, max(10, int(self.settings.task_intense_max_turns)))
            return random.randint(lower_bound, upper_bound)

        lower_bound = max(8, int(self.settings.task_normal_min_turns))
        upper_bound = max(lower_bound, max(15, int(self.settings.task_normal_max_turns)))
        return random.randint(lower_bound, upper_bound)

    @staticmethod
    def _row_to_task(row: Any) -> Task:
        return Task(
            id=row["id"],
            telegram_user_id=row["telegram_user_id"],
            title=row["title"],
            instructions=row["instructions"],
            status=TaskStatus(row["status"]),
            intensity=TaskIntensity(row["intensity"]),
            created_at=row["created_at"],
            due_at=row["due_at"],
            issued_at_turn=row["issued_at_turn"],
            completed_at=row["completed_at"],
            skipped_at=row["skipped_at"],
            source=row["source"],
            metadata=json.loads(row["metadata"] or "{}"),
        )
