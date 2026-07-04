from __future__ import annotations

import json
import logging
import random
import re
from json import JSONDecodeError
from typing import Any

from core.models import ConversationState, EngineResult, Task, TaskFollowupKind, UserProfile
from core.reply_utils import should_show_quick_replies
from services.media_service import MediaBundle

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
        "description": (
            "Create ONE formal task ONLY when task_window_ready is true and no open task exists. "
            "Photo verification tasks additionally require photo_task_window_ready=true. "
            "Photo tasks must stay rare. Skip if unsure — most replies should not include a task. "
            "Do not use for casual verbal commands or humiliation that belongs in normal dialogue."
        ),
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
    {
        "type": "function",
        "name": "roll_random_twist",
        "description": "Roll a dice for a random domination twist, punishment style, scene variation, or sudden command to increase unpredictability and replayability. Use sparingly when the scene feels repetitive or to inject fresh energy. Return a short creative description the Queen can adapt into dialogue.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional hint: punishment / humiliation / task_modifier / mood_shift / media_focus / surprise_order. Leave empty for fully random.",
                }
            },
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
        onboarding_service: Any | None = None,
    ) -> None:
        self.settings = settings
        self.grok_client = grok_client
        self.user_service = user_service
        self.task_service = task_service
        self.memory_service = memory_service
        self.media_service = media_service
        self.safety_service = safety_service
        self.context_builder = context_builder
        self.onboarding_service = onboarding_service

    async def handle_text_message(
        self,
        *,
        telegram_user_id: int,
        username: str | None,
        display_name: str,
        text: str,
    ) -> EngineResult:
        return await self._handle_user_message(
            telegram_user_id=telegram_user_id,
            username=username,
            display_name=display_name,
            user_text=text,
            message_kind="text",
            message_metadata={},
            photo_task_resolution=False,
        )

    async def handle_photo_message(
        self,
        *,
        telegram_user_id: int,
        username: str | None,
        display_name: str,
        photo_path: str,
        caption: str | None = None,
    ) -> EngineResult:
        photo_description = ""
        try:
            photo_description = await self.grok_client.describe_user_photo(image_path=photo_path)
        except Exception as exc:
            logger.exception("Photo vision failed for user_id=%s: %s", telegram_user_id, exc)

        user_text = self._build_photo_submission_text(
            caption=caption,
            photo_description=photo_description,
        )
        return await self._handle_user_message(
            telegram_user_id=telegram_user_id,
            username=username,
            display_name=display_name,
            user_text=user_text,
            message_kind="photo",
            message_metadata={
                "photo_path": photo_path,
                "photo_description": photo_description,
                "caption": caption,
            },
            photo_task_resolution=True,
        )

    async def _handle_user_message(
        self,
        *,
        telegram_user_id: int,
        username: str | None,
        display_name: str,
        user_text: str,
        message_kind: str,
        message_metadata: dict[str, Any],
        photo_task_resolution: bool,
    ) -> EngineResult:
        profile = await self.user_service.get_or_create(
            telegram_user_id=telegram_user_id,
            username=username,
            display_name=display_name,
        )
        profile = await self.user_service.sync_runtime_state(profile)
        logger.info(
            "Handling %s message for user_id=%s state=%s turns=%s",
            message_kind,
            telegram_user_id,
            profile.state,
            profile.conversation_count,
        )

        if (
            message_kind == "text"
            and not profile.onboarding_completed
            and self.onboarding_service is not None
        ):
            profile = await self.onboarding_service.complete_from_user_text(
                telegram_user_id,
                user_text,
            )

        safeword_source = message_metadata.get("caption") or user_text
        safeword_level = self.safety_service.classify_safeword(str(safeword_source))
        if safeword_level is not None:
            logger.warning("Safeword detected for user_id=%s level=%s", telegram_user_id, safeword_level)
            decision = await self.safety_service.handle_safeword(profile, level=safeword_level)
            assistant_event = "aftercare" if safeword_level.value == "red" else "safeword_yellow"
            await self._store_message(
                telegram_user_id,
                "user",
                user_text,
                message_kind=message_kind,
                metadata={"event": "safeword", "safeword_level": safeword_level.value, **message_metadata},
            )
            await self._store_message(
                telegram_user_id,
                "assistant",
                decision.reply,
                metadata={"event": assistant_event, "safeword_level": safeword_level.value},
            )
            return EngineResult(text=decision.reply, state=decision.state)

        explicit_limits = self.safety_service.extract_limits(user_text)
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

        if photo_task_resolution:
            task_followup = await self.task_service.resolve_photo_task_submission(telegram_user_id)
            caption = str(message_metadata.get("caption") or "").strip()
            if task_followup.kind == TaskFollowupKind.NONE and caption:
                task_followup = await self.task_service.resolve_open_task_followup(
                    telegram_user_id=telegram_user_id,
                    current_turn=profile.conversation_count,
                    user_text=caption,
                )
        else:
            task_followup = await self.task_service.resolve_open_task_followup(
                telegram_user_id=telegram_user_id,
                current_turn=profile.conversation_count,
                user_text=user_text,
            )
        if task_followup.kind != TaskFollowupKind.NONE:
            logger.info(
                "Resolved task followup for user_id=%s kind=%s task_id=%s at turn=%s",
                telegram_user_id,
                task_followup.kind,
                task_followup.task.id if task_followup.task else None,
                profile.conversation_count,
            )
            profile = await self.user_service.sync_runtime_state(
                await self.user_service.get_profile(telegram_user_id)
            )

        active_task = await self.task_service.get_open_task(telegram_user_id)
        profile, task_window_ready = await self.task_service.evaluate_task_window(
            profile=profile,
            active_task=active_task,
        )
        profile, photo_task_window_ready = await self.task_service.evaluate_photo_task_window(
            profile=profile,
            active_task=active_task,
        )
        profile, video_window_ready = await self.media_service.evaluate_video_window(profile=profile)
        media_turn_hints = self.media_service.analyze_turn_hints(
            user_text=user_text,
            video_window_ready=video_window_ready,
        )
        await self.memory_service.ingest_user_turn(
            telegram_user_id,
            user_text,
            explicit_limits=explicit_limits,
        )

        recent_messages = await self.memory_service.recent_messages(
            telegram_user_id,
            limit=self.settings.recent_message_limit,
        )
        recalled_memories = await self.memory_service.recall_relevant(
            telegram_user_id,
            query=user_text,
            profile=profile,
        )
        await self._store_message(
            telegram_user_id,
            "user",
            user_text,
            message_kind=message_kind,
            metadata={
                "task_followup_kind": str(task_followup.kind),
                "task_followup_id": task_followup.task.id if task_followup.task else None,
                "explicit_limits": explicit_limits,
                **message_metadata,
            },
        )
        media_summary = await self.media_service.asset_summary()
        video_categories_context = self.media_service.video_categories_context(media_summary=media_summary)
        logger.debug(
            "Context prepared for user_id=%s state=%s task_window_ready=%s photo_task_window_ready=%s video_window_ready=%s active_task=%s media=%s",
            telegram_user_id,
            profile.state,
            task_window_ready,
            photo_task_window_ready,
            video_window_ready,
            active_task.id if active_task else None,
            media_summary,
        )
        messages = self.context_builder.build_messages(
            profile=profile,
            user_text=user_text,
            recent_messages=recent_messages,
            active_task=active_task,
            task_window_ready=task_window_ready,
            photo_task_window_ready=photo_task_window_ready,
            local_media_summary=media_summary,
            task_followup_kind=task_followup.kind,
            resolved_task=task_followup.task,
            recalled_memories=recalled_memories,
            media_turn_hints=media_turn_hints,
            video_categories_context=video_categories_context,
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
                photo_task_window_ready=photo_task_window_ready,
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

        if generated_urls:
            logger.debug(
                "Skipping autonomous media decision for user_id=%s because tool-generated images already exist.",
                telegram_user_id,
            )
            resolved_media = MediaBundle(images=generated_urls[:], videos=[])
        else:
            resolved_media = await self.media_service.get_or_generate_media(
                user_text=user_text,
                response_text=response_text,
                user_id=telegram_user_id,
                video_window_ready=video_window_ready,
            )

        local_images, local_videos, generated_urls, video_caption, text_before_video, suggested_foreshadow = self._finalize_media_outputs(
            media_bundle=resolved_media,
            generated_urls=generated_urls,
            user_id=telegram_user_id,
        )

        if local_images or local_videos:
            try:
                await self.media_service.record_deliveries(
                    telegram_user_id,
                    [*local_images, *local_videos],
                )
            except Exception as exc:
                logger.exception(
                    "Failed to record local media deliveries for user_id=%s: %s",
                    telegram_user_id,
                    exc,
                )

        if local_videos:
            try:
                video_profile = await self.user_service.get_profile(telegram_user_id)
                await self.media_service.schedule_next_video_turn(
                    telegram_user_id,
                    video_profile.state,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to schedule next video window for user_id=%s: %s",
                    telegram_user_id,
                    exc,
                )

        if not response_text and not (local_images or local_videos or generated_urls):
            profile = await self._refresh_profile(profile)
            logger.warning("Model returned no usable text or media for user_id=%s; using fallback reply.", telegram_user_id)
            response_text = self._fallback_reply(profile)

        latest_profile = await self.user_service.get_profile(telegram_user_id)
        final_open_task = await self.task_service.get_open_task(telegram_user_id)
        show_quick_replies = should_show_quick_replies(
            state=latest_profile.state,
            has_open_task=final_open_task is not None,
        )
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
                "video_caption": video_caption,
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
            video_caption=video_caption,
            text_before_video=text_before_video,
            user_text_for_caption=user_text,
            show_quick_replies=show_quick_replies,
            has_open_task=final_open_task is not None,
            suggested_video_foreshadow=suggested_foreshadow,
        )

    async def _resolve_tool_loop(
        self,
        *,
        response: Any,
        profile: UserProfile,
        task_window_ready: bool,
        photo_task_window_ready: bool,
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
                        photo_task_window_ready=photo_task_window_ready,
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
        photo_task_window_ready: bool,
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
            await self.memory_service.ingest_profile_updates(
                profile.telegram_user_id,
                dislikes=dislikes,
                hard_limits=hard_limits,
                notes=notes,
            )
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
                photo_task_window_ready=photo_task_window_ready,
                current_open_task=current_open_task,
                title=str(arguments.get("title", "")),
                instructions=str(arguments.get("instructions", "")),
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

        if name == "roll_random_twist":
            category = str(arguments.get("category", "")).strip().lower() or None
            twist = self._roll_random_twist(category=category, profile=current_profile)
            logger.info(
                "Rolled random twist for user_id=%s category=%s: %s",
                current_profile.telegram_user_id,
                category,
                twist[:60],
            )
            return {
                "ok": True,
                "twist": twist,
                "category": category or "fully_random",
                "state": str(current_profile.state),
            }, None, []

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
        media_bundle: MediaBundle,
        generated_urls: list[str],
        user_id: int,
    ) -> tuple[list[str], list[str], list[str], str | None, bool, str | None]:
        local_images = [item for item in media_bundle.images if not self._is_remote_asset(item)]
        local_videos = media_bundle.videos[:]
        merged_generated_urls = self._dedupe_strings(
            generated_urls + [item for item in media_bundle.images if self._is_remote_asset(item)]
        )
        video_caption = media_bundle.video_caption
        text_before_video = media_bundle.text_before_video
        suggested = getattr(media_bundle, "suggested_foreshadow", None)

        if merged_generated_urls and (local_images or local_videos):
            logger.info(
                "Dropping local media for user_id=%s because generated images are present. local_images=%s local_videos=%s generated_images=%s",
                user_id,
                len(local_images),
                len(local_videos),
                len(merged_generated_urls),
            )
            return [], [], merged_generated_urls, None, False, None

        return local_images, local_videos, merged_generated_urls, video_caption, text_before_video, suggested

    @staticmethod
    def _should_generate_image(profile: UserProfile) -> bool:
        return profile.state not in {ConversationState.AFTERCARE, ConversationState.PAUSED}

    @staticmethod
    def _build_photo_submission_text(*, caption: str | None, photo_description: str) -> str:
        lines = ["[用户发送了一张照片]"]
        if photo_description.strip():
            lines.append(f"照片内容：{photo_description.strip()}")
        else:
            lines.append("照片内容：尚未生成视觉描述。")
        if caption and caption.strip():
            lines.append(f"用户配文：{caption.strip()}")
        return "\n".join(lines)

    async def _store_message(
        self,
        telegram_user_id: int,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        message_kind: str = "text",
    ) -> None:
        try:
            await self.memory_service.store_message(
                telegram_user_id,
                role,
                content,
                message_kind=message_kind,
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
        photo_task_window_ready: bool,
        current_open_task: Task | None,
        title: str = "",
        instructions: str = "",
    ) -> str | None:
        if current_profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return f"task_blocked_by_state:{current_profile.state}"
        if current_open_task is not None:
            return "existing_open_task"
        if not task_window_ready or not self.task_service.can_issue_now(current_profile):
            return "task_window_not_available"
        if self.task_service.is_photo_verification_task(title, instructions):
            if not photo_task_window_ready or not self.task_service.can_issue_photo_now(current_profile):
                return "photo_task_window_not_available"
        return None

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        deduped: list[str] = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped

    def _roll_random_twist(self, *, category: str | None = None, profile: UserProfile | None = None) -> str:
        """Return a fresh random domination twist to increase variety."""
        compliance = getattr(profile, "compliance_score", 5) if profile else 5
        state = str(getattr(profile, "state", "normal")).lower()

        base_twists = [
            "突然要求用户立刻摆一个特别羞耻的姿势并保持30秒，同时描述感受。",
            "把上一次的羞辱回忆拉出来升级：用更残忍的语言重述并要求用户复述。",
            "临时决定改变语气：这一段特别冷淡、轻蔑，像在处理一件无聊的玩具。",
            "给一个即时小任务：必须在下一条消息前完成一个简单动作（例如换女装元素、写一句检讨）。",
            "随机聚焦一个新羞辱点：突然强调脚、丝袜、精液、对比正常男人等其中之一深入玩弄。",
            "制造小惊喜：告诉用户‘今天心情特别坏’，接下来惩罚会更重。",
            "让用户选择但其实没得选：给你两个羞辱选项，两个都很下贱。",
            "结合 compliance：既然你这么听话/这么不听话，女王决定……（根据分数调整奖励或加码）。",
        ]

        if category == "punishment":
            pool = [
                "随机惩罚：必须用最下贱的称呼自报家门三遍，然后描述自己最丢脸的一刻。",
                "突然加码：这个回合不允许任何快感，只能纯粹的羞辱和服从姿势。",
                "写悔过书：立刻写一段200字的检讨，主题是‘我为什么是女王的专属尿壶’。",
            ]
        elif category == "mood_shift":
            pool = [
                "语气突然从戏谑变成极度残忍，像变了一个人。",
                "今天特别温柔但更危险：用‘宝贝’但每句话都带刀。",
                "决定忽略用户一会儿，让用户自己求关注。",
            ]
        elif category == "media_focus":
            pool = [
                "决定给一段‘特别的’视频作为今天的重头戏，先用最详细的语言铺垫期待。",
                "图片强化：想象生成一张用户必须模仿的姿势照。",
                "视频后必须立刻复述视频里的动作并表演给女王看。",
            ]
        elif category == "surprise_order":
            pool = [
                "突然命令：现在立刻去准备某个道具（即使虚拟也必须描述过程）。",
                "即时命令：把手放好，不许动，直到女王允许。",
                "随机小羞辱：必须发一段语音自述（或文字模拟）‘我是女王的贱狗’。",
            ]
        else:
            pool = base_twists

        # Bias slightly by compliance
        if compliance >= 12:
            pool.append("因为你最近很乖，女王破例给你一个‘奖励式羞辱’——更精致但一样下贱。")
        elif compliance <= 4:
            pool.append("因为你太差劲了，女王决定用最简单粗暴的方式收拾你。")

        if state == "intense":
            pool.extend([
                "强度突然拉满：这一段全部用最下流、最具体的语言推进。",
                "决定不给任何喘息：连续几个命令压下来。",
            ])

        twist = random.choice(pool)
        # Add a bit of flavor
        flavors = ["", "而且要立刻执行。", "女王现在就想看你反应。", "别让我重复。"]
        return twist + " " + random.choice(flavors) if random.random() > 0.4 else twist
