from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ConversationState(StrEnum):
    NORMAL = "normal"
    INTENSE = "intense"
    AFTERCARE = "aftercare"
    PAUSED = "paused"


class TaskStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    SKIPPED = "skipped"


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
    aftercare_until: str | None
    paused_reason: str | None
    dislikes: list[str] = field(default_factory=list)
    hard_limits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    last_model_response_at: str | None = None


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


@dataclass(slots=True)
class SafetyDecision:
    triggered: bool
    reply: str
    state: ConversationState
    recorded_limits: list[str] = field(default_factory=list)

