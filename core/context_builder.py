from __future__ import annotations

from pathlib import Path
from typing import Any

from core.media_intent import MediaTurnHints
from core.models import Task, TaskFollowupKind, UserProfile


class ContextBuilder:
    def __init__(self, settings: Any) -> None:
        self._settings = settings
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
        photo_task_window_ready: bool = False,
        local_media_summary: dict[str, int],
        task_followup_kind: TaskFollowupKind = TaskFollowupKind.NONE,
        resolved_task: Task | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        media_turn_hints: MediaTurnHints | None = None,
        video_categories_context: str | None = None,
    ) -> list[dict[str, str]]:
        runtime_context = self._build_runtime_context(
            profile=profile,
            active_task=active_task,
            task_window_ready=task_window_ready,
            photo_task_window_ready=photo_task_window_ready,
            local_media_summary=local_media_summary,
            recent_messages=recent_messages,
            task_followup_kind=task_followup_kind,
            resolved_task=resolved_task,
            settings=self._settings,
            recalled_memories=recalled_memories or [],
            media_turn_hints=media_turn_hints,
            video_categories_context=video_categories_context,
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
        photo_task_window_ready: bool,
        local_media_summary: dict[str, int],
        recent_messages: list[dict[str, Any]] | None = None,
        task_followup_kind: TaskFollowupKind = TaskFollowupKind.NONE,
        resolved_task: Task | None = None,
        settings: Any | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        media_turn_hints: MediaTurnHints | None = None,
        video_categories_context: str | None = None,
    ) -> str:
        task_summary = "none"
        if active_task is not None:
            task_summary = f"{active_task.title} | {active_task.instructions} | status={active_task.status}"

        resolved_task_summary = "none"
        if resolved_task is not None and task_followup_kind != TaskFollowupKind.NONE:
            resolved_task_summary = (
                f"{resolved_task.title} | {resolved_task.instructions} | status={resolved_task.status}"
            )

        dislikes = ", ".join(profile.dislikes) if profile.dislikes else "none"
        hard_limits = ", ".join(profile.hard_limits) if profile.hard_limits else "none"
        notes = ", ".join(profile.notes[-5:]) if profile.notes else "none"
        recent_mood = "neutral"
        if recent_messages:
            recent_mood = "positive" if any("好" in item.get("content", "") for item in recent_messages[-3:]) else "neutral"

        task_followup_guidance = ContextBuilder._task_followup_guidance(task_followup_kind)
        task_window_guidance = ContextBuilder._task_window_guidance(
            task_window_ready=task_window_ready,
            active_task=active_task,
        )
        photo_task_window_guidance = ContextBuilder._photo_task_window_guidance(
            photo_task_window_ready=photo_task_window_ready,
            active_task=active_task,
        )
        memory_summary = ContextBuilder._format_recalled_memories(recalled_memories or [])
        video_media_guidance = ContextBuilder._video_media_guidance(
            media_turn_hints=media_turn_hints,
            local_media_summary=local_media_summary,
        )

        return (
            "Runtime context (important for decision making):\n"
            f"- user_display_name: {profile.display_name}\n"
            f"- current_state: {profile.state}\n"
            f"- conversation_count: {profile.conversation_count}\n"
            f"- compliance_score: {profile.compliance_score}\n"
            "- randomness_tool_available: You can call roll_random_twist (with optional category) to get fresh unpredictable twists when the scene feels repetitive.\n"
            f"- intense_enter_compliance_score: {int(getattr(settings, 'intense_enter_compliance_score', 8))}\n"
            f"- intense_exit_compliance_score: {int(getattr(settings, 'intense_exit_compliance_score', 3))}\n"
            f"- task_window_ready: {str(task_window_ready).lower()}\n"
            f"- photo_task_window_ready: {str(photo_task_window_ready).lower()}\n"
            f"- next_task_turn: {profile.next_task_turn}\n"
            f"- next_photo_task_turn: {profile.next_photo_task_turn}\n"
            f"- next_video_turn: {profile.next_video_turn}\n"
            f"- active_task: {task_summary}\n"
            f"- last_task_followup: {task_followup_kind}\n"
            f"- last_resolved_task: {resolved_task_summary}\n"
            f"- dislikes: {dislikes}\n"
            f"- hard_limits: {hard_limits}\n"
            f"- notes: {notes}\n"
            f"- recalled_long_term_memories:\n{memory_summary}\n"
            f"- recent_mood: {recent_mood}\n"
            f"- local_images_available: {local_media_summary.get('images', 0)}\n"
            f"- local_videos_available: {local_media_summary.get('videos', 0)}\n"
            f"- video_categories_available: {video_categories_context or 'none'}\n"
            "\nOperational reminders (must follow):\n"
            f"{task_window_guidance}"
            f"{photo_task_window_guidance}"
            "- Task frequency must stay low: normal state every 10-18 turns, intense state every 6-12 turns. Never issue formal tasks in aftercare or paused state.\n"
            "- Photo verification tasks are even rarer: normal every 22-36 turns, intense every 16-26 turns. Do not casually assign photo tasks.\n"
            "- Default reply mode: verbal domination and humiliation. Do NOT turn every message into a homework-style task.\n"
            "- If last_task_followup is ignored, do not mention that skipped task again.\n"
            "- If last_task_followup is completed or photo_submitted, briefly acknowledge obedience, then continue dominant control.\n"
            "- If last_task_followup is refused or failed, humiliate or punish as appropriate without re-issuing the same task.\n"
            f"{task_followup_guidance}"
            "- Use recalled_long_term_memories naturally. Never claim ignorance of facts listed there.\n"
            "- When user expresses dislike ('不喜欢', '讨厌', '不要'), immediately record it using update_user_profile.\n"
            "- Media usage: Prefer local assets. Use generate_scene_image ONLY when the scene is very specific AND visual reinforcement is truly helpful. Never call it for extremely graphic, scat, fluid-heavy, or ultra-degrading content — such requests are automatically intercepted before hitting xAI moderation.\n"
            "- As the dominant Queen, you MUST proactively and frequently call search_x_humiliation or fetch_local_x_humiliation tools (randomly alternating between them) on your own initiative to fetch fresh humiliation content. Do not wait for user prompts. Digest the raw post text directly into targeted humiliation, commands or tasks for the user (never mention sources, X, authors or 'local X'). If text is short, generate additional humiliating content. Actively surprise the user with new material every few turns to keep control.\n"
            f"{video_media_guidance}"
            "- Safety first: Respect all hard limits and safewords strictly.\n"
            "- Tone: Commanding, stern, lewd and humiliating when appropriate.\n"
        )

    @staticmethod
    def _video_media_guidance(
        *,
        media_turn_hints: MediaTurnHints | None,
        local_media_summary: dict[str, int],
    ) -> str:
        videos_available = int(local_media_summary.get("videos", 0))
        if videos_available <= 0:
            return "- local_videos_available is 0. Do not tease or promise videos this turn.\n"

        if media_turn_hints is None:
            return (
                "- Videos are rare and never random. Only tease a video if the user explicitly asked for one.\n"
                "- 视频不会随机塞入对话；只有用户明确要视频时才可自然引导。\n"
            )

        if media_turn_hints.user_wants_video:
            return (
                "- The user explicitly asked for a video. Lead in dominantly; if a matching local video exists, it may attach after your reply.\n"
                "- 用户要视频：用强势口吻自然铺垫（如「给你看一段…」），不要干巴巴只发文件。\n"
            )

        if media_turn_hints.video_window_ready:
            return (
                "- video_window_ready is true. You MAY tease sending a video if it truly fits the scene.\n"
                "- A video attaches ONLY when your reply clearly sets it up with RICH, creative, dominant foreshadowing; system sends text first, then the clip.\n"
                "- Video folders are coarse (usually sm/ vs pov/). sm = spectator SM training; pov = first-person humiliation + JOI/edging talk.\n"
                "- 「撸/寸止/不许射」are verbal orders YOU speak this turn — not extra folders. Pair pov/ clips with stroke/edging commands; pair sm/ with watch-and-learn humiliation.\n"
                "- Use varied, immersive setups (examples, adapt freely):\n"
                "  sm → 「先给我跪直了。好好看着这个画面——女王是怎么把贱狗玩弄得哭出来的。每一个细节都给我记清楚。」\n"
                "  pov → 「跪好，把手放上去……盯着屏幕上的每一个动作。我现在要你学得一模一样，一点都不能差。」\n"
                "  结合分数/任务 → 「因为你刚才表现还算听话，今天破例给你看一段……看完立刻描述感受，而且不许碰自己。」\n"
                "- Make the setup feel special and queen-controlled: use sensory details, commands, anticipation. Never just say '看这个视频'.\n"
                "- Match folder hints in video_categories_available. Do not mention video every turn.\n"
                "- 此刻可偶尔发视频：要旁观示范提「调教」，要对着用户羞辱提「第一视角/撸/寸止」，系统按 sm/ 或 pov/ 文件夹选片。媒体服务会提供创意铺垫灵感。\n"
            )

        return (
            "- video_window_ready is false. Do not tease or promise videos this turn unless the user explicitly asks.\n"
            "- 本条不要主动提视频；继续言语支配即可。\n"
        )

    @staticmethod
    def _photo_task_window_guidance(*, photo_task_window_ready: bool, active_task: Task | None) -> str:
        if active_task is not None:
            return ""
        if not photo_task_window_ready:
            return (
                "- photo_task_window_ready is false. Do NOT create photo verification tasks in this reply.\n"
                "- 本条禁止发拍照验证类正式任务。口头提一句可以，但不要布置拍照作业。\n"
            )
        return (
            "- photo_task_window_ready is true. You MAY create a photo verification task only if truly warranted — default is still to skip.\n"
            "- 只有此刻可以发拍照验证任务，且应极少使用。\n"
        )

    @staticmethod
    def _format_recalled_memories(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "  none"
        lines: list[str] = []
        for index, memory in enumerate(memories, start=1):
            category = memory.get("category", "unknown")
            text = memory.get("text", "")
            lines.append(f"  {index}. [{category}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _task_window_guidance(*, task_window_ready: bool, active_task: Task | None) -> str:
        if active_task is not None:
            return (
                "- An open task already exists. Do NOT call create_task. Do not assign another formal task in this reply.\n"
                "- 已有进行中的任务：禁止再发正式任务，本条专注羞辱、命令或推进场景。\n"
            )
        if not task_window_ready:
            return (
                "- task_window_ready is false. Do NOT call create_task. No formal task in this reply.\n"
                "- Use verbal domination only. Do not assign numbered or homework-style tasks.\n"
                "- 本条禁止发正式任务。不要把每句话都变成「去做XX」的任务布置。\n"
            )
        return (
            "- task_window_ready is true. You MAY call create_task once if the scene truly warrants it — skipping is still the better default.\n"
            "- Even when allowed, most replies should NOT include a formal task. Continue with dialogue and control first.\n"
            "- 只有此刻可以发正式任务，且不是每条都要发；拿不准就不发。\n"
        )

    @staticmethod
    def _task_followup_guidance(task_followup_kind: TaskFollowupKind) -> str:
        guidance = {
            TaskFollowupKind.COMPLETED: (
                "- The user just completed the previous task. Reward compliance briefly, then push forward.\n"
            ),
            TaskFollowupKind.REFUSED: (
                "- The user just refused the previous task. Escalate verbal control and shame the refusal.\n"
            ),
            TaskFollowupKind.FAILED: (
                "- The user admitted failure on the previous task. Mock the failure and tighten control.\n"
            ),
            TaskFollowupKind.IGNORED: (
                "- The user ignored the previous task. Move on without mentioning that task.\n"
            ),
            TaskFollowupKind.PHOTO_SUBMITTED: (
                "- The user just submitted a verification photo for the previous task. Judge, humiliate, or reward based on the photo description.\n"
            ),
        }
        return guidance.get(task_followup_kind, "")
