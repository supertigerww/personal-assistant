from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from aiogram import F, Router
from aiogram.exceptions import TelegramEntityTooLarge
from aiogram.types import FSInputFile, Message
import random

from bot.handlers.chat_actions import keep_typing
from bot.handlers.keyboards import build_quick_reply_keyboard
from core.config import Settings
from core.models import EngineResult
from core.queen_engine import QueenEngine
from core.reply_utils import split_reply_text
from services.media_service import MediaService
from services.memory_service import MemoryService
from services.processing_gate import ProcessingGate
from services.user_photo_service import UserPhotoService
from services.user_service import UserService

logger = logging.getLogger(__name__)
router = Router(name="messages")

MediaKind = Literal["photo", "video", "animation"]
MediaItem = tuple[MediaKind, str, bool]


@router.message(F.text)
async def text_message_handler(
    message: Message,
    engine: QueenEngine,
    processing_gate: ProcessingGate,
    media_service: MediaService,
    user_service: UserService,
    memory_service: MemoryService,
    settings: Settings,
) -> None:
    try:
        result = await _run_engine_turn(
            message=message,
            processing_gate=processing_gate,
            handler=lambda: engine.handle_text_message(
                telegram_user_id=message.from_user.id,
                username=message.from_user.username,
                display_name=message.from_user.full_name,
                text=message.text,
            ),
        )
        await _reply_with_engine_result(
            message,
            result,
            media_service=media_service,
            user_service=user_service,
            memory_service=memory_service,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(
            "Error in text_message_handler for user_id=%s: %s",
            getattr(message.from_user, "id", None),
            exc,
        )
        await message.answer("系统暂时出现问题，请稍后再试。")


@router.message(F.photo)
async def photo_message_handler(
    message: Message,
    engine: QueenEngine,
    user_photo_service: UserPhotoService,
    processing_gate: ProcessingGate,
    media_service: MediaService,
    user_service: UserService,
    memory_service: MemoryService,
    settings: Settings,
) -> None:
    try:
        async with processing_gate.acquire(message.from_user.id):
            async with keep_typing(message.bot, message.chat.id):
                saved_photo = await user_photo_service.save_telegram_photo(bot=message.bot, message=message)
                result = await engine.handle_photo_message(
                    telegram_user_id=message.from_user.id,
                    username=message.from_user.username,
                    display_name=message.from_user.full_name,
                    photo_path=saved_photo.path,
                    caption=saved_photo.caption,
                )
        await _reply_with_engine_result(
            message,
            result,
            media_service=media_service,
            user_service=user_service,
            memory_service=memory_service,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(
            "Error in photo_message_handler for user_id=%s: %s",
            getattr(message.from_user, "id", None),
            exc,
        )
        await message.answer("照片接收失败，请稍后再试。")


async def _run_engine_turn(
    *,
    message: Message,
    processing_gate: ProcessingGate,
    handler,
) -> EngineResult:
    async with processing_gate.acquire(message.from_user.id):
        async with keep_typing(message.bot, message.chat.id):
            return await handler()


async def _resolve_video_caption_if_needed(
    message: Message,
    result: EngineResult,
    *,
    media_service: MediaService,
    user_service: UserService,
    memory_service: MemoryService,
    settings: Settings,
) -> str | None:
    if not (result.text_before_video and result.local_video_paths):
        return result.video_caption
    if result.video_caption:
        return result.video_caption

    telegram_user_id = message.from_user.id
    recent_captions = await memory_service.recent_video_captions(
        telegram_user_id,
        limit=settings.video_caption_history_limit,
    )
    profile = await user_service.get_profile(telegram_user_id)

    async with keep_typing(message.bot, message.chat.id):
        caption = await media_service.generate_video_caption(
            video_path=result.local_video_paths[0],
            user_text=result.user_text_for_caption,
            response_text=result.text,
            profile=profile,
            recent_captions=recent_captions,
        )

    if caption:
        await memory_service.patch_last_assistant_metadata(
            telegram_user_id,
            {"video_caption": caption},
        )
    return caption


async def _reply_with_engine_result(
    message: Message,
    result: EngineResult,
    *,
    media_service: MediaService,
    user_service: UserService,
    memory_service: MemoryService,
    settings: Settings,
) -> None:
    keyboard = build_quick_reply_keyboard(result)
    video_caption = await _resolve_video_caption_if_needed(
        message,
        result,
        media_service=media_service,
        user_service=user_service,
        memory_service=memory_service,
        settings=settings,
    )

    if result.text_before_video and result.local_video_paths:
        text_to_use = result.text
        # Enhance foreshadowing with suggested creative setup from media service if the response is light on setup cues
        if result.suggested_video_foreshadow:
            cue_markers_lower = ["给你看", "盯着", "跪", "看着", "示范", "学着", "手放", "跟着", "看完", "现在"]
            has_strong_cue = any(m in result.text.lower() for m in cue_markers_lower)
            if not has_strong_cue or len(result.text) < 40:
                text_to_use = result.suggested_video_foreshadow + "\n\n" + result.text

        chunks = split_reply_text(text_to_use)
        for index, chunk in enumerate(chunks):
            has_followup_media = bool(
                result.local_image_paths or result.generated_image_urls or result.local_video_paths or getattr(result, "x_humiliation_posts", None)
            )
            markup = keyboard if index == len(chunks) - 1 and not has_followup_media else None
            await message.answer(chunk, reply_markup=markup)

        for image_path in result.local_image_paths:
            await _send_single_media(
                message,
                (_detect_media_kind(image_path, default="photo"), image_path, True),
            )
        for image_source in result.generated_image_urls:
            await _send_single_media(
                message,
                (
                    _detect_media_kind(image_source, default="photo"),
                    image_source,
                    not _is_remote_source(image_source),
                ),
            )
        for video_path in result.local_video_paths:
            await _send_single_media(
                message,
                (_detect_media_kind(video_path, default="video"), video_path, True),
                caption=video_caption,
                reply_markup=keyboard if video_path == result.local_video_paths[-1] else None,
            )

        # Send media from X posts (local DB or search) to enhance humiliation.
        # Text is already digested into the main response by the Queen (no source mentions).
        if getattr(result, "x_humiliation_posts", None):
            x_posts = list(result.x_humiliation_posts)
            random.shuffle(x_posts)  # ensure random order, not first folder/file
            for post in x_posts:
                for mpath in post.get("media_paths", []):
                    # Normalize path in case DB stored with extra prefix
                    cleaned = mpath
                    for bad in ["app/images/", "images/", "app/assets/images/", "assets/images/"]:
                        if cleaned.startswith(bad):
                            cleaned = cleaned[len(bad):]
                            break
                    if not cleaned.startswith("/"):
                        cleaned = str(Path("/app/assets/x_assets") / cleaned.lstrip("/"))
                    kind = _detect_media_kind(cleaned, default="photo")
                    try:
                        await _send_single_media(
                            message,
                            (kind, cleaned, True),
                        )
                    except Exception as e:
                        logger.warning("Failed to send X media %s: %s", cleaned, e)
        return

    # Also send X media in the general media sequence case
    if getattr(result, "x_humiliation_posts", None):
        x_posts = list(result.x_humiliation_posts)
        random.shuffle(x_posts)  # randomize to not start from first subfolder/file
        for post in x_posts:
            for mpath in post.get("media_paths", []):
                cleaned = mpath
                for bad in ["app/images/", "images/", "app/assets/images/", "assets/images/"]:
                    if cleaned.startswith(bad):
                        cleaned = cleaned[len(bad):]
                        break
                if not cleaned.startswith("/"):
                    cleaned = str(Path("/app/assets/x_assets") / cleaned.lstrip("/"))
                kind = _detect_media_kind(cleaned, default="photo")
                try:
                    await _send_single_media(
                        message,
                        (kind, cleaned, True),
                    )
                except Exception as e:
                    logger.warning("Failed to send X media %s: %s", cleaned, e)

    media_items = _build_media_items(result)
    if media_items:
        await _send_media_sequence(message, media_items, caption=result.text, reply_markup=keyboard)
        return

    chunks = split_reply_text(result.text)
    for index, chunk in enumerate(chunks):
        markup = keyboard if index == len(chunks) - 1 else None
        await message.answer(chunk, reply_markup=markup)

    # Fallback: send any X media if not sent in other branches
    if getattr(result, "x_humiliation_posts", None):
        x_posts = list(result.x_humiliation_posts)
        random.shuffle(x_posts)  # ensure random order, not first subfolder/file
        for post in x_posts:
            for mpath in post.get("media_paths", []):
                cleaned = mpath
                for bad in ["app/images/", "images/", "app/assets/images/", "assets/images/"]:
                    if cleaned.startswith(bad):
                        cleaned = cleaned[len(bad):]
                        break
                if not cleaned.startswith("/"):
                    cleaned = str(Path("/app/assets/x_assets") / cleaned.lstrip("/"))
                kind = _detect_media_kind(cleaned, default="photo")
                try:
                    await _send_single_media(
                        message,
                        (kind, cleaned, True),
                    )
                except Exception as e:
                    logger.warning("Failed to send X media %s: %s", cleaned, e)


def _build_media_items(result: EngineResult) -> list[MediaItem]:
    items: list[MediaItem] = []

    for image_path in result.local_image_paths:
        items.append((_detect_media_kind(image_path, default="photo"), image_path, True))

    for video_path in result.local_video_paths:
        items.append((_detect_media_kind(video_path, default="video"), video_path, True))

    for image_source in result.generated_image_urls:
        items.append(
            (
                _detect_media_kind(image_source, default="photo"),
                image_source,
                not _is_remote_source(image_source),
            )
        )

    return items


async def _send_media_sequence(
    message: Message,
    media_items: list[MediaItem],
    *,
    caption: str,
    reply_markup=None,
) -> None:
    first_item, *remaining_items = media_items
    first_sent = await _send_single_media(
        message,
        first_item,
        caption=caption,
        reply_markup=reply_markup if not remaining_items else None,
    )
    if not first_sent:
        await message.answer(caption, reply_markup=reply_markup)
        return

    for item in remaining_items:
        await _send_single_media(message, item)


async def _send_single_media(
    message: Message,
    item: MediaItem,
    *,
    caption: str | None = None,
    reply_markup=None,
) -> bool:
    kind, source, is_local = item
    payload = FSInputFile(source) if is_local else source

    try:
        if kind == "animation":
            await message.answer_animation(payload, caption=caption, reply_markup=reply_markup)
        elif kind == "video":
            await message.answer_video(payload, caption=caption, supports_streaming=True, reply_markup=reply_markup)
        else:
            await message.answer_photo(payload, caption=caption, reply_markup=reply_markup)
        return True
    except TelegramEntityTooLarge as exc:
        logger.warning(
            "Skipped oversized %s for chat_id=%s source=%s size_mb=%s: %s",
            kind,
            message.chat.id,
            source,
            _local_file_size_mb(source) if is_local else None,
            exc,
        )
        return False
    except Exception as exc:
        logger.exception(
            "Failed to send %s for chat_id=%s source=%s: %s",
            kind,
            message.chat.id,
            source,
            exc,
        )

        if not is_local and kind in {"photo", "animation"}:
            try:
                await message.answer(f"Generated image: {source}")
            except Exception as fallback_exc:
                logger.exception(
                    "Failed to send fallback generated image URL for chat_id=%s: %s",
                    message.chat.id,
                    fallback_exc,
                )
        return False


def _detect_media_kind(source: str, *, default: MediaKind) -> MediaKind:
    clean_source = source.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(clean_source).suffix.casefold()
    if suffix == ".gif":
        return "animation"
    if suffix in {".mp4", ".mov", ".mkv", ".webm"}:
        return "video"
    return default


def _is_remote_source(source: str) -> bool:
    normalized = source.casefold()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _local_file_size_mb(source: str) -> float | None:
    try:
        return round(Path(source).stat().st_size / (1024 * 1024), 2)
    except OSError:
        return None