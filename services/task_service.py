from __future__ import annotations

import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from core.models import (
    ConversationState,
    Task,
    TaskFollowupKind,
    TaskFollowupResult,
    TaskIntensity,
    TaskStatus,
    UserProfile,
)

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskService:
    COMPLETION_MARKERS: tuple[str, ...] = (
        "完成了",
        "做完了",
        "做好了",
        "已完成",
        "照做了",
        "搞定了",
        "做了",
        "finished",
        "completed",
        "done",
    )
    REFUSAL_MARKERS: tuple[str, ...] = (
        "不想做",
        "不愿意",
        "拒绝",
        "不做",
        "不肯",
        "算了",
        "refuse",
        "won't",
        "wont",
    )
    FAILURE_MARKERS: tuple[str, ...] = (
        "没做成",
        "失败了",
        "没完成",
        "做不到",
        "没能",
        "failed",
        "couldn't",
        "couldnt",
    )
    PHOTO_TASK_MARKERS: tuple[str, ...] = (
        "拍照",
        "照片",
        "发图",
        "上图",
        "拍张",
        "自拍",
        "验证照",
        "拍一张",
        "发一张",
        "photo",
        "picture",
        "selfie",
        "upload",
    )

    def __init__(self, *, database: Any, settings: Any, user_service: Any) -> None:
        self.database = database
        self.settings = settings
        self.user_service = user_service

    async def ensure_schedule(self, profile: UserProfile) -> None:
        if profile.next_task_turn <= 0:
            await self.schedule_next_task(
                telegram_user_id=profile.telegram_user_id,
                state=str(profile.state),
                from_turn=profile.conversation_count,
            )
        if profile.next_photo_task_turn <= 0:
            await self.schedule_next_photo_task(
                telegram_user_id=profile.telegram_user_id,
                state=str(profile.state),
                from_turn=profile.conversation_count,
            )

    async def schedule_next_task(self, *, telegram_user_id: int, state: str, from_turn: int) -> int:
        interval = self._pick_interval(state)
        next_turn = from_turn + interval
        await self.user_service.update_next_task_turn(telegram_user_id, next_turn)
        return next_turn

    async def schedule_next_photo_task(self, *, telegram_user_id: int, state: str, from_turn: int) -> int:
        interval = self._pick_photo_interval(state)
        next_turn = from_turn + interval
        await self.user_service.update_next_photo_task_turn(telegram_user_id, next_turn)
        return next_turn

    async def evaluate_task_window(
        self,
        *,
        profile: UserProfile,
        active_task: Task | None,
    ) -> tuple[UserProfile, bool]:
        if active_task is not None:
            return profile, False
        if not self.can_issue_now(profile):
            return profile, False

        chance = self._task_offer_probability(profile.state)
        roll = random.random()
        if roll < chance:
            logger.info(
                "Task window opened for user %s at turn=%s chance=%.2f roll=%.2f",
                profile.telegram_user_id,
                profile.conversation_count,
                chance,
                roll,
            )
            return profile, True

        retry_interval = self._pick_retry_interval(profile.state)
        next_turn = profile.conversation_count + retry_interval
        await self.user_service.update_next_task_turn(profile.telegram_user_id, next_turn)
        logger.info(
            "Deferred task window for user %s from turn=%s to next_task_turn=%s chance=%.2f roll=%.2f",
            profile.telegram_user_id,
            profile.conversation_count,
            next_turn,
            chance,
            roll,
        )
        return await self.user_service.get_profile(profile.telegram_user_id), False

    async def evaluate_photo_task_window(
        self,
        *,
        profile: UserProfile,
        active_task: Task | None,
    ) -> tuple[UserProfile, bool]:
        if active_task is not None:
            return profile, False
        if not self.can_issue_photo_now(profile):
            return profile, False

        chance = self._photo_task_offer_probability(profile.state)
        roll = random.random()
        if roll < chance:
            logger.info(
                "Photo task window opened for user %s at turn=%s chance=%.2f roll=%.2f",
                profile.telegram_user_id,
                profile.conversation_count,
                chance,
                roll,
            )
            return profile, True

        retry_interval = self._pick_photo_retry_interval(profile.state)
        next_turn = profile.conversation_count + retry_interval
        await self.user_service.update_next_photo_task_turn(profile.telegram_user_id, next_turn)
        logger.info(
            "Deferred photo task window for user %s from turn=%s to next_photo_task_turn=%s chance=%.2f roll=%.2f",
            profile.telegram_user_id,
            profile.conversation_count,
            next_turn,
            chance,
            roll,
        )
        return await self.user_service.get_profile(profile.telegram_user_id), False

    def can_issue_now(self, profile: UserProfile) -> bool:
        if profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return False
        return profile.conversation_count >= profile.next_task_turn

    def can_issue_photo_now(self, profile: UserProfile) -> bool:
        if profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return False
        return profile.conversation_count >= profile.next_photo_task_turn

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
        merged_metadata = dict(metadata or {})
        if self.is_photo_verification_task(cleaned_title, cleaned_instructions):
            merged_metadata["requires_photo"] = True
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
                json.dumps(merged_metadata, ensure_ascii=False),
            ),
        )
        logger.info(f"Task created for user {telegram_user_id}: {cleaned_title}")
        created = await self.get_task(task_id)
        if self.is_photo_verification_task(cleaned_title, cleaned_instructions):
            await self.schedule_next_photo_task(
                telegram_user_id=telegram_user_id,
                state=str((await self.user_service.get_profile(telegram_user_id)).state),
                from_turn=created.issued_at_turn or 0,
            )
        return created

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

    async def complete_task(self, task_id: str) -> Task:
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'completed', completed_at = ?
            WHERE id = ?
            """,
            (utc_now_iso(), task_id),
        )
        return await self.get_task(task_id)

    async def refuse_task(self, task_id: str) -> Task:
        now = utc_now_iso()
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'refused', skipped_at = ?
            WHERE id = ?
            """,
            (now, task_id),
        )
        return await self.get_task(task_id)

    async def fail_task(self, task_id: str) -> Task:
        now = utc_now_iso()
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'failed', skipped_at = ?
            WHERE id = ?
            """,
            (now, task_id),
        )
        return await self.get_task(task_id)

    async def skip_task(self, task_id: str) -> Task:
        await self.database.execute(
            """
            UPDATE tasks
            SET status = 'skipped', skipped_at = ?
            WHERE id = ?
            """,
            (utc_now_iso(), task_id),
        )
        return await self.get_task(task_id)

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

    async def resolve_photo_task_submission(self, telegram_user_id: int) -> TaskFollowupResult:
        task = await self.get_open_task(telegram_user_id)
        if task is None:
            return TaskFollowupResult(kind=TaskFollowupKind.NONE)
        if not self.task_requires_photo(task):
            logger.info(
                "Ignored photo submission for non-photo task user %s task_id=%s",
                telegram_user_id,
                task.id,
            )
            return TaskFollowupResult(kind=TaskFollowupKind.NONE, task=task)

        resolved = await self.complete_task(task.id)
        await self.user_service.adjust_compliance_score(
            telegram_user_id,
            int(self.settings.task_completion_score_delta),
        )
        logger.info("Photo submitted for open task user %s task_id=%s", telegram_user_id, task.id)
        return TaskFollowupResult(kind=TaskFollowupKind.PHOTO_SUBMITTED, task=resolved)

    async def resolve_open_task_followup(
        self,
        *,
        telegram_user_id: int,
        current_turn: int,
        user_text: str,
    ) -> TaskFollowupResult:
        task = await self.get_open_task(telegram_user_id)
        if task is None or task.issued_at_turn is None:
            return TaskFollowupResult(kind=TaskFollowupKind.NONE)
        if current_turn != task.issued_at_turn + 1:
            return TaskFollowupResult(kind=TaskFollowupKind.NONE, task=task)

        followup_kind = self.classify_task_followup(user_text)
        if followup_kind == TaskFollowupKind.COMPLETED:
            resolved = await self.complete_task(task.id)
            await self.user_service.adjust_compliance_score(
                telegram_user_id,
                int(self.settings.task_completion_score_delta),
            )
            logger.info("Task completed for user %s: %s", telegram_user_id, task.title)
            return TaskFollowupResult(kind=TaskFollowupKind.COMPLETED, task=resolved)

        if followup_kind == TaskFollowupKind.REFUSED:
            resolved = await self.refuse_task(task.id)
            await self.user_service.adjust_compliance_score(
                telegram_user_id,
                int(self.settings.task_refusal_score_delta),
            )
            logger.info("Task refused for user %s: %s", telegram_user_id, task.title)
            return TaskFollowupResult(kind=TaskFollowupKind.REFUSED, task=resolved)

        if followup_kind == TaskFollowupKind.FAILED:
            resolved = await self.fail_task(task.id)
            await self.user_service.adjust_compliance_score(
                telegram_user_id,
                int(self.settings.task_failure_score_delta),
            )
            logger.info("Task failed for user %s: %s", telegram_user_id, task.title)
            return TaskFollowupResult(kind=TaskFollowupKind.FAILED, task=resolved)

        resolved = await self.skip_task(task.id)
        await self.user_service.adjust_compliance_score(
            telegram_user_id,
            int(self.settings.task_ignore_score_delta),
        )
        logger.info("Task ignored for user %s: %s", telegram_user_id, task.title)
        return TaskFollowupResult(kind=TaskFollowupKind.IGNORED, task=resolved)

    async def skip_ignored_task_if_needed(
        self,
        *,
        telegram_user_id: int,
        current_turn: int,
        user_text: str,
    ) -> Task | None:
        result = await self.resolve_open_task_followup(
            telegram_user_id=telegram_user_id,
            current_turn=current_turn,
            user_text=user_text,
        )
        if result.kind == TaskFollowupKind.IGNORED:
            return result.task
        return None

    @classmethod
    def classify_task_followup(cls, user_text: str) -> TaskFollowupKind | None:
        normalized = user_text.casefold().strip()
        if not normalized:
            return None

        if cls._matches_any(normalized, cls.COMPLETION_MARKERS):
            return TaskFollowupKind.COMPLETED
        if cls._matches_any(normalized, cls.REFUSAL_MARKERS):
            return TaskFollowupKind.REFUSED
        if cls._matches_any(normalized, cls.FAILURE_MARKERS):
            return TaskFollowupKind.FAILED
        return None

    @classmethod
    def looks_like_task_response(cls, user_text: str) -> bool:
        return cls.classify_task_followup(user_text) is not None

    @classmethod
    def is_photo_verification_task(cls, title: str, instructions: str) -> bool:
        combined = f"{title} {instructions}".casefold()
        return any(marker in combined for marker in cls.PHOTO_TASK_MARKERS)

    @classmethod
    def task_requires_photo(cls, task: Task) -> bool:
        if task.metadata.get("requires_photo"):
            return True
        return cls.is_photo_verification_task(task.title, task.instructions)

    @staticmethod
    def _matches_any(normalized_text: str, markers: tuple[str, ...]) -> bool:
        ordered = sorted(markers, key=len, reverse=True)
        return any(marker in normalized_text for marker in ordered)

    def _pick_interval(self, state: str) -> int:
        if state == ConversationState.INTENSE:
            lower_bound = max(5, int(self.settings.task_intense_min_turns))
            upper_bound = max(lower_bound, max(10, int(self.settings.task_intense_max_turns)))
            return random.randint(lower_bound, upper_bound)

        lower_bound = max(8, int(self.settings.task_normal_min_turns))
        upper_bound = max(lower_bound, max(15, int(self.settings.task_normal_max_turns)))
        return random.randint(lower_bound, upper_bound)

    def _pick_retry_interval(self, state: str) -> int:
        if state == ConversationState.INTENSE:
            lower_bound = max(1, int(getattr(self.settings, "task_retry_min_turns_intense", 1)))
            upper_bound = max(lower_bound, int(getattr(self.settings, "task_retry_max_turns_intense", 2)))
            return random.randint(lower_bound, upper_bound)

        lower_bound = max(1, int(getattr(self.settings, "task_retry_min_turns_normal", 2)))
        upper_bound = max(lower_bound, int(getattr(self.settings, "task_retry_max_turns_normal", 4)))
        return random.randint(lower_bound, upper_bound)

    def _task_offer_probability(self, state: str) -> float:
        if state == ConversationState.INTENSE:
            raw_value = getattr(self.settings, "task_offer_probability_intense", 0.6)
        else:
            raw_value = getattr(self.settings, "task_offer_probability_normal", 0.4)
        return max(0.0, min(1.0, float(raw_value)))

    def _pick_photo_interval(self, state: str) -> int:
        if state == ConversationState.INTENSE:
            lower_bound = max(10, int(self.settings.photo_task_intense_min_turns))
            upper_bound = max(lower_bound, int(self.settings.photo_task_intense_max_turns))
            return random.randint(lower_bound, upper_bound)

        lower_bound = max(15, int(self.settings.photo_task_normal_min_turns))
        upper_bound = max(lower_bound, int(self.settings.photo_task_normal_max_turns))
        return random.randint(lower_bound, upper_bound)

    def _pick_photo_retry_interval(self, state: str) -> int:
        if state == ConversationState.INTENSE:
            lower_bound = max(2, int(getattr(self.settings, "photo_task_retry_min_turns_intense", 3)))
            upper_bound = max(lower_bound, int(getattr(self.settings, "photo_task_retry_max_turns_intense", 6)))
            return random.randint(lower_bound, upper_bound)

        lower_bound = max(3, int(getattr(self.settings, "photo_task_retry_min_turns_normal", 4)))
        upper_bound = max(lower_bound, int(getattr(self.settings, "photo_task_retry_max_turns_normal", 8)))
        return random.randint(lower_bound, upper_bound)

    def _photo_task_offer_probability(self, state: str) -> float:
        if state == ConversationState.INTENSE:
            raw_value = getattr(self.settings, "photo_task_offer_probability_intense", 0.2)
        else:
            raw_value = getattr(self.settings, "photo_task_offer_probability_normal", 0.12)
        return max(0.0, min(1.0, float(raw_value)))

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