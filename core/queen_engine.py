from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from typing import Any

from core.models import ConversationState, EngineResult, Task, UserProfile

logger = logging.getLogger(__name__)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "update_user_profile",
        "description": "Store new dislikes, hard limits, or profile notes that the user explicitly revealed.",
        "parameters": {
            "type": "object",
            "properties": {
                "dislikes": {"type": "array", "items": {"type": "string"}},
                "hard_limits": {"type": "array", "items": {"type": "string"}},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "name": "create_task",
        "description": "Create one concise, non-explicit task when the runtime context says task_window_ready is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "instructions": {"type": "string"},
                "intensity": {"type": "string", "enum": ["normal", "intense"]},
                "due_at": {"type": "string"},
            },
            "required": ["title", "instructions"],
        },
    },
    {
        "type": "function",
        "name": "pause_all_tasks",
        "description": "Pause or clear current tasks when safety, aftercare, or a user request requires it.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "target_state": {"type": "string", "enum": ["aftercare", "paused"]},
            },
            "required": ["reason", "target_state"],
        },
    },
    {
        "type": "function",
        "name": "generate_scene_image",
        "description": "Generate a supplemental image only when local media is clearly insufficient and a concrete visual would help.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 4},
            },
            "required": ["prompt"],
        },
    },
]


class QueenEngine:
    MEDIA_PLACEHOLDER_PATTERNS = (
        r"[（(]\s*(?:图片|图像)?\s*生成中[\s.…。.!！]*[)）]",
        r"[（(]\s*(?:generating(?:\s+image)?|image\s+generating|loading)[\s.….!！]*[)）]",
        r"(?:^|\n)\s*(?:正在生成(?:图片|图像)|图片生成中|图像生成中)[\s.…。!！]*(?=$|\n)",
    )

    def __init__(
        self,
        *,
        settings: Any,
        grok_client: Any,
        user_service: Any,
        task_service: Any,
        memory_service: Any,
        media_service: Any,
        safety_service: Any,
        context_builder: Any,
    ) -> None:
        self.settings = settings
        self.grok_client = grok_client
        self.user_service = user_service
        self.task_service = task_service
        self.memory_service = memory_service
        self.media_service = media_service
        self.safety_service = safety_service
        self.context_builder = context_builder

    async def handle_text_message(
        self,
        *,
        telegram_user_id: int,
        username: str | None,
        display_name: str,
        text: str,
    ) -> EngineResult:
        profile = await self.user_service.get_or_create(
            telegram_user_id=telegram_user_id,
            username=username,
            display_name=display_name,
        )
        logger.info(
            "Handling text message for user_id=%s state=%s turns=%s",
            telegram_user_id,
            profile.state,
            profile.conversation_count,
        )

        # 安全词检测必须最先执行
        if self.safety_service.detect_safeword(text):
            logger.warning("Safeword detected for user_id=%s", telegram_user_id)
            decision = await self.safety_service.handle_safeword(profile)
            await self._store_message(
                telegram_user_id,
                "user",
                text,
                metadata={"event": "safeword"},
            )
            await self._store_message(
                telegram_user_id,
                "assistant",
                decision.reply,
                metadata={"event": "aftercare"},
            )
            return EngineResult(text=decision.reply, state=decision.state)

        # 用户明确表达不喜欢的内容时，优先记入档案
        explicit_limits = self.safety_service.extract_limits(text)
        if explicit_limits:
            logger.info("Recording explicit dislikes for user_id=%s: %s", telegram_user_id, explicit_limits)
            await self.user_service.append_dislikes(telegram_user_id, explicit_limits)

        await self.task_service.ensure_schedule(profile)
        profile = await self.user_service.increment_conversation_count(telegram_user_id)
        logger.debug(
            "Conversation count incremented for user_id=%s to turn=%s",
            telegram_user_id,
            profile.conversation_count,
        )

        skipped_task = await self.task_service.skip_ignored_task_if_needed(
            telegram_user_id=telegram_user_id,
            current_turn=profile.conversation_count,
            user_text=text,
        )
        if skipped_task is not None:
            logger.info(
                "Skipped ignored task for user_id=%s task_id=%s at turn=%s",
                telegram_user_id,
                skipped_task.id,
                profile.conversation_count,
            )

        active_task = await self.task_service.get_open_task(telegram_user_id)
        profile, task_window_ready = await self.task_service.evaluate_task_window(
            profile=profile,
            active_task=active_task,
        )
        recent_messages = await self.memory_service.recent_messages(
            telegram_user_id,
            limit=self.settings.recent_message_limit,
        )
        await self._store_message(
            telegram_user_id,
            "user",
            text,
            metadata={
                "skipped_task_id": skipped_task.id if skipped_task else None,
                "explicit_limits": explicit_limits,
            },
        )
        media_summary = await self.media_service.asset_summary()
        logger.debug(
            "Context prepared for user_id=%s state=%s task_window_ready=%s active_task=%s media=%s",
            telegram_user_id,
            profile.state,
            task_window_ready,
            active_task.id if active_task else None,
            media_summary,
        )
        messages = self.context_builder.build_messages(
            profile=profile,
            user_text=text,
            recent_messages=recent_messages,
            active_task=active_task,
            task_window_ready=task_window_ready,
            local_media_summary=media_summary,
        )

        response_text = ""
        created_task: Task | None = None
        generated_urls: list[str] = []

        try:
            response = await self.grok_client.create_response(input_items=messages, tools=TOOL_SCHEMAS)
            logger.debug(
                "Initial model response received for user_id=%s response_id=%s",
                telegram_user_id,
                getattr(response, "id", None),
            )
            response_text, created_task, generated_urls = await self._resolve_tool_loop(
                response=response,
                profile=profile,
                task_window_ready=task_window_ready,
            )
        except Exception as exc:
            logger.exception("Model flow failed for user_id=%s: %s", telegram_user_id, exc)
            profile = await self._refresh_profile(profile)
            try:
                created_task = await self.task_service.get_open_task(telegram_user_id)
            except Exception as task_exc:
                logger.exception("Failed to recover active task for user_id=%s: %s", telegram_user_id, task_exc)
            response_text = self._fallback_reply(profile)

        if response_text:
            response_text = self._sanitize_response_text(response_text)
        else:
            logger.warning("Model returned empty text for user_id=%s before media resolution.", telegram_user_id)

        media_context = self._build_media_context(
            user_text=text,
            response_text=response_text,
            created_task=created_task,
        )
        if generated_urls:
            logger.debug(
                "Skipping autonomous media decision for user_id=%s because tool-generated images already exist.",
                telegram_user_id,
            )
            media_bundle = {"images": generated_urls[:], "videos": []}
        else:
            media_bundle = await self.media_service.get_or_generate_media(
                context=media_context,
                user_id=telegram_user_id,
            )

        local_images, local_videos, generated_urls = self._finalize_media_outputs(
            media_bundle=media_bundle,
            generated_urls=generated_urls,
            user_id=telegram_user_id,
        )

        if not response_text and not (local_images or local_videos or generated_urls):
            profile = await self._refresh_profile(profile)
            logger.warning("Model returned no usable text or media for user_id=%s; using fallback reply.", telegram_user_id)
            response_text = self._fallback_reply(profile)

        latest_profile = await self.user_service.get_profile(telegram_user_id)
        logger.info(
            "Reply ready for user_id=%s final_state=%s task_id=%s local_images=%s local_videos=%s generated_images=%s",
            telegram_user_id,
            latest_profile.state,
            created_task.id if created_task else None,
            len(local_images),
            len(local_videos),
            len(generated_urls),
        )
        await self._store_message(
            telegram_user_id,
            "assistant",
            response_text,
            metadata={
                "task_id": created_task.id if created_task else None,
                "generated_image_urls": generated_urls,
                "local_images": local_images,
                "local_videos": local_videos,
                "state": str(latest_profile.state),
            },
        )
        return EngineResult(
            text=response_text,
            state=latest_profile.state,
            task=created_task,
            local_image_paths=local_images,
            local_video_paths=local_videos,
            generated_image_urls=generated_urls,
        )

    async def _resolve_tool_loop(
        self,
        *,
        response: Any,
        profile: UserProfile,
        task_window_ready: bool,
    ) -> tuple[str, Task | None, list[str]]:
        created_task: Task | None = None
        generated_urls: list[str] = []

        for round_index in range(1, 5):
            function_calls = self.grok_client.extract_function_calls(response)
            if not function_calls:
                logger.debug(
                    "Tool loop finished for user_id=%s round=%s generated_urls=%s",
                    profile.telegram_user_id,
                    round_index,
                    len(generated_urls),
                )
                return self.grok_client.extract_text(response), created_task, self._dedupe_strings(generated_urls)

            logger.info(
                "Processing %s tool call(s) for user_id=%s in round=%s",
                len(function_calls),
                profile.telegram_user_id,
                round_index,
            )
            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                try:
                    result, maybe_task, maybe_urls = await self._execute_tool(
                        profile=profile,
                        function_call=function_call,
                        task_window_ready=task_window_ready,
                    )
                except Exception as exc:
                    logger.exception(
                        "Tool execution failed for user_id=%s tool=%s: %s",
                        profile.telegram_user_id,
                        function_call.get("name"),
                        exc,
                    )
                    result, maybe_task, maybe_urls = (
                        {
                            "ok": False,
                            "error": "tool_execution_failed",
                            "tool": function_call.get("name"),
                        },
                        None,
                        [],
                    )
                if maybe_task is not None:
                    created_task = maybe_task
                if maybe_urls:
                    generated_urls.extend(maybe_urls)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call["call_id"],
                        "output": json.dumps(result, ensure_ascii=False),
                    }
                )

            try:
                response = await self.grok_client.create_response(
                    input_items=tool_outputs,
                    tools=TOOL_SCHEMAS,
                    previous_response_id=response.id,
                )
            except Exception as exc:
                logger.exception(
                    "Follow-up model call failed for user_id=%s after round=%s: %s",
                    profile.telegram_user_id,
                    round_index,
                    exc,
                )
                raise

        logger.warning("Tool loop reached max rounds for user_id=%s", profile.telegram_user_id)
        return self.grok_client.extract_text(response), created_task, self._dedupe_strings(generated_urls)

    async def _execute_tool(
        self,
        *,
        profile: UserProfile,
        function_call: dict[str, str],
        task_window_ready: bool,
    ) -> tuple[dict[str, Any], Task | None, list[str]]:
        current_profile = await self.user_service.get_profile(profile.telegram_user_id)
        name = function_call["name"]
        try:
            arguments = json.loads(function_call["arguments"] or "{}")
        except JSONDecodeError as exc:
            logger.exception(
                "Invalid tool arguments for user_id=%s tool=%s: %s",
                current_profile.telegram_user_id,
                name,
                exc,
            )
            return {"ok": False, "error": "invalid_tool_arguments", "tool": name}, None, []

        logger.info(
            "Executing tool=%s for user_id=%s state=%s",
            name,
            current_profile.telegram_user_id,
            current_profile.state,
        )

        if name == "update_user_profile":
            dislikes = arguments.get("dislikes") or []
            hard_limits = arguments.get("hard_limits") or []
            notes = arguments.get("notes") or []
            if dislikes:
                await self.user_service.append_dislikes(profile.telegram_user_id, dislikes)
            if hard_limits:
                await self.user_service.append_hard_limits(profile.telegram_user_id, hard_limits)
            if notes:
                await self.user_service.append_notes(profile.telegram_user_id, notes)
            logger.info(
                "Profile updated via tool for user_id=%s dislikes=%s hard_limits=%s notes=%s",
                current_profile.telegram_user_id,
                len(dislikes),
                len(hard_limits),
                len(notes),
            )
            return {
                "ok": True,
                "dislikes_added": dislikes,
                "hard_limits_added": hard_limits,
                "notes_added": notes,
                "state": str(current_profile.state),
            }, None, []

        if name == "create_task":
            current_open_task = await self.task_service.get_open_task(current_profile.telegram_user_id)
            blocked_reason = self._get_task_block_reason(
                current_profile=current_profile,
                task_window_ready=task_window_ready,
                current_open_task=current_open_task,
            )
            if blocked_reason is not None:
                logger.info(
                    "Rejected create_task for user_id=%s reason=%s",
                    current_profile.telegram_user_id,
                    blocked_reason,
                )
                return {
                    "ok": False,
                    "error": blocked_reason,
                    "state": str(current_profile.state),
                }, None, []
            task = await self.task_service.create_task(
                telegram_user_id=current_profile.telegram_user_id,
                title=arguments["title"],
                instructions=arguments["instructions"],
                intensity=arguments.get("intensity", "normal"),
                due_at=arguments.get("due_at"),
                issued_at_turn=current_profile.conversation_count,
                source="model",
            )
            next_task_turn = await self.task_service.schedule_next_task(
                telegram_user_id=current_profile.telegram_user_id,
                state=str(current_profile.state),
                from_turn=current_profile.conversation_count,
            )
            logger.info(
                "Created task for user_id=%s task_id=%s next_task_turn=%s",
                current_profile.telegram_user_id,
                task.id,
                next_task_turn,
            )
            return {
                "ok": True,
                "task_id": task.id,
                "title": task.title,
                "next_task_turn": next_task_turn,
                "state": str(current_profile.state),
            }, task, []

        if name == "pause_all_tasks":
            target_state = arguments.get("target_state", "paused")
            reason = arguments.get("reason", "unspecified")
            await self.task_service.pause_all_tasks(current_profile.telegram_user_id, reason=reason)
            await self.user_service.update_state(current_profile.telegram_user_id, target_state, paused_reason=reason)
            updated_profile = await self._refresh_profile(current_profile)
            logger.warning(
                "Paused tasks for user_id=%s target_state=%s reason=%s",
                current_profile.telegram_user_id,
                target_state,
                reason,
            )
            return {
                "ok": True,
                "target_state": target_state,
                "reason": reason,
                "state": str(updated_profile.state),
            }, None, []

        if name == "generate_scene_image":
            prompt = str(arguments.get("prompt", "")).strip()
            if not prompt:
                logger.info("Rejected generate_scene_image for user_id=%s because prompt is empty.", current_profile.telegram_user_id)
                return {"ok": False, "error": "empty_prompt", "state": str(current_profile.state)}, None, []
            if not self._should_generate_image(current_profile):
                logger.info(
                    "Blocked generate_scene_image for user_id=%s because state=%s",
                    current_profile.telegram_user_id,
                    current_profile.state,
                )
                return {
                    "ok": False,
                    "error": "image_generation_not_allowed_in_current_state",
                    "state": str(current_profile.state),
                }, None, []
            count = max(1, min(int(arguments.get("count", 1)), 4))
            urls = await self.media_service.generate_scene_image(prompt=prompt, count=count)
            logger.info(
                "Generated scene image(s) for user_id=%s count=%s",
                current_profile.telegram_user_id,
                len(urls),
            )
            return {
                "ok": bool(urls),
                "urls": urls,
                "count": len(urls),
                "state": str(current_profile.state),
            }, None, urls

        logger.warning("Unknown tool requested for user_id=%s tool=%s", current_profile.telegram_user_id, name)
        return {"ok": False, "error": f"unknown_tool:{name}"}, None, []

    @staticmethod
    def _fallback_reply(profile: UserProfile) -> str:
        if profile.state == ConversationState.AFTERCARE:
            return "先深呼吸，慢慢来。我在这里陪着你。"
        if profile.state == ConversationState.PAUSED:
            return "会话已暂停。准备好了就告诉我。"
        return "系统刚刚有点卡顿，现在继续。"

    @staticmethod
    def _is_remote_asset(value: str) -> bool:
        normalized = value.casefold()
        return normalized.startswith("http://") or normalized.startswith("https://")

    @staticmethod
    def _build_media_context(
        *,
        user_text: str,
        response_text: str,
        created_task: Task | None,
    ) -> str:
        primary_text = user_text.strip()
        if primary_text:
            return primary_text

        fallback_parts: list[str] = []
        trimmed_response = response_text.strip()
        if trimmed_response:
            fallback_parts.append(trimmed_response)

        task_title = created_task.title.strip() if created_task and created_task.title else ""
        if task_title:
            fallback_parts.append(task_title)

        return " ".join(fallback_parts).strip()

    @classmethod
    def _sanitize_response_text(cls, text: str) -> str:
        cleaned = text
        for pattern in cls.MEDIA_PLACEHOLDER_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = cleaned.strip()

        if cleaned != text.strip():
            logger.info("Removed placeholder text from model reply.")

        return cleaned

    def _finalize_media_outputs(
        self,
        *,
        media_bundle: dict[str, list[str]],
        generated_urls: list[str],
        user_id: int,
    ) -> tuple[list[str], list[str], list[str]]:
        local_images = [item for item in media_bundle.get("images", []) if not self._is_remote_asset(item)]
        local_videos = media_bundle.get("videos", [])[:]
        merged_generated_urls = self._dedupe_strings(
            generated_urls + [item for item in media_bundle.get("images", []) if self._is_remote_asset(item)]
        )

        if merged_generated_urls and (local_images or local_videos):
            logger.info(
                "Dropping local media for user_id=%s because generated images are present. local_images=%s local_videos=%s generated_images=%s",
                user_id,
                len(local_images),
                len(local_videos),
                len(merged_generated_urls),
            )
            return [], [], merged_generated_urls

        return local_images, local_videos, merged_generated_urls

    @staticmethod
    def _should_generate_image(profile: UserProfile) -> bool:
        return profile.state not in {ConversationState.AFTERCARE, ConversationState.PAUSED}

    async def _store_message(
        self,
        telegram_user_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            await self.memory_service.store_message(
                telegram_user_id,
                role,
                content,
                metadata=metadata,
            )
        except Exception as exc:
            logger.exception(
                "Failed to store message for user_id=%s role=%s: %s",
                telegram_user_id,
                role,
                exc,
            )

    async def _refresh_profile(self, profile: UserProfile) -> UserProfile:
        try:
            return await self.user_service.get_profile(profile.telegram_user_id)
        except Exception as exc:
            logger.exception("Failed to refresh profile for user_id=%s: %s", profile.telegram_user_id, exc)
            return profile

    def _get_task_block_reason(
        self,
        *,
        current_profile: UserProfile,
        task_window_ready: bool,
        current_open_task: Task | None,
    ) -> str | None:
        if current_profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return f"task_blocked_by_state:{current_profile.state}"
        if current_open_task is not None:
            return "existing_open_task"
        if not task_window_ready or not self.task_service.can_issue_now(current_profile):
            return "task_window_not_available"
        return None

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
