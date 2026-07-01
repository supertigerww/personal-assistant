from __future__ import annotations

from pathlib import Path
from typing import Any

from core.models import Task, UserProfile


class ContextBuilder:
    def __init__(self, settings: Any) -> None:
        self._prompt_path = Path(settings.prompt_path)
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8")

    def build_messages(
        self,
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
        active_task: Task | None,
        task_window_ready: bool,
        local_media_summary: dict[str, int],
    ) -> list[dict[str, str]]:
        runtime_context = self._build_runtime_context(
            profile=profile,
            active_task=active_task,
            task_window_ready=task_window_ready,
            local_media_summary=local_media_summary,
            recent_messages=recent_messages,
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": f"{self._system_prompt}\n\n{runtime_context}"}]
        for item in recent_messages:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _build_runtime_context(
        *,
        profile: UserProfile,
        active_task: Task | None,
        task_window_ready: bool,
        local_media_summary: dict[str, int],
        recent_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        task_summary = "none"
        if active_task is not None:
            task_summary = f"{active_task.title} | {active_task.instructions} | status={active_task.status}"

        dislikes = ", ".join(profile.dislikes) if profile.dislikes else "none"
        hard_limits = ", ".join(profile.hard_limits) if profile.hard_limits else "none"
        notes = ", ".join(profile.notes[-5:]) if profile.notes else "none"
        recent_mood = "neutral"
        if recent_messages:
            recent_mood = "positive" if any("好" in item.get("content", "") for item in recent_messages[-3:]) else "neutral"

        return (
            "Runtime context (important for decision making):\n"
            f"- user_display_name: {profile.display_name}\n"
            f"- current_state: {profile.state}\n"
            f"- conversation_count: {profile.conversation_count}\n"
            f"- task_window_ready: {str(task_window_ready).lower()}\n"
            f"- next_task_turn: {profile.next_task_turn}\n"
            f"- active_task: {task_summary}\n"
            f"- dislikes: {dislikes}\n"
            f"- hard_limits: {hard_limits}\n"
            f"- notes: {notes}\n"
            f"- recent_mood: {recent_mood}\n"
            f"- local_images_available: {local_media_summary.get('images', 0)}\n"
            f"- local_videos_available: {local_media_summary.get('videos', 0)}\n"
            "\nOperational reminders (must follow):\n"
            "- Task frequency must stay low: normal state every 8-15 turns, intense state every 5-10 turns. Never issue tasks in aftercare or paused state.\n"
            "- If user ignores a task in the next message, skip it completely and do not mention it again.\n"
            "- When user expresses dislike ('不喜欢', '讨厌', '不要'), immediately record it using update_user_profile.\n"
            "- Media usage: Prefer local assets. Use generate_scene_image only when the scene is very specific or local media is insufficient.\n"
            "- Safety first: Respect all hard limits and safewords strictly.\n"
            "- Tone: Commanding, stern, lewd and humiliating when appropriate.\n"
        )
