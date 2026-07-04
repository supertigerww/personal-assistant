from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConversationState(StrEnum):
    NORMAL = "normal"
    INTENSE = "intense"
    AFTERCARE = "aftercare"
    PAUSED = "paused"


class SafewordLevel(StrEnum):
    RED = "red"
    YELLOW = "yellow"


class TaskStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    REFUSED = "refused"
    FAILED = "failed"


class TaskFollowupKind(StrEnum):
    NONE = "none"
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"
    IGNORED = "ignored"
    PHOTO_SUBMITTED = "photo_submitted"


@dataclass(slots=True)
class SavedUserPhoto:
    telegram_user_id: int
    path: str
    file_id: str
    caption: str | None = None


class TaskIntensity(StrEnum):
    NORMAL = "normal"
    INTENSE = "intense"


@dataclass(slots=True)
class UserProfile:
    telegram_user_id: int
    username: str | None
    display_name: str
    state: ConversationState
    compliance_score: int
    conversation_count: int
    next_task_turn: int
    next_photo_task_turn: int
    next_video_turn: int
    aftercare_until: str | None
    paused_reason: str | None
    dislikes: list[str] = field(default_factory=list)
    hard_limits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    onboarding_completed: bool = False
    last_model_response_at: str | None = None


@dataclass(slots=True)
class TaskFollowupResult:
    kind: TaskFollowupKind = TaskFollowupKind.NONE
    task: Task | None = None


@dataclass(slots=True)
class Task:
    id: str
    telegram_user_id: int
    title: str
    instructions: str
    status: TaskStatus
    intensity: TaskIntensity
    created_at: str
    due_at: str | None
    issued_at_turn: int | None
    completed_at: str | None
    skipped_at: str | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineResult:
    text: str
    state: ConversationState
    task: Task | None = None
    local_image_paths: list[str] = field(default_factory=list)
    local_video_paths: list[str] = field(default_factory=list)
    generated_image_urls: list[str] = field(default_factory=list)
    video_caption: str | None = None
    text_before_video: bool = False
    user_text_for_caption: str = ""
    show_quick_replies: bool = False
    has_open_task: bool = False
    suggested_video_foreshadow: str | None = None  # Rich creative foreshadow from media service for enhanced delivery


@dataclass(slots=True)
class SafetyDecision:
    triggered: bool
    reply: str
    state: ConversationState
    safeword_level: SafewordLevel | None = None
    recorded_limits: list[str] = field(default_factory=list)

