from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from aiogram import F, Router
from aiogram.types import FSInputFile, Message

from core.models import EngineResult
from core.queen_engine import QueenEngine

logger = logging.getLogger(__name__)
router = Router(name="messages")

MediaKind = Literal["photo", "video", "animation"]
MediaItem = tuple[MediaKind, str, bool]


@router.message(F.text)
async def text_message_handler(
    message: Message,
    engine: QueenEngine,
) -> None:
    try:
        result = await engine.handle_text_message(
            telegram_user_id=message.from_user.id,
            username=message.from_user.username,
            display_name=message.from_user.full_name,
            text=message.text,
        )

        media_items = _build_media_items(result)
        if media_items:
            await _send_media_sequence(message, media_items, caption=result.text)
        else:
            await message.answer(result.text)
    except Exception as exc:
        logger.exception(
            "Error in text_message_handler for user_id=%s: %s",
            getattr(message.from_user, "id", None),
            exc,
        )
        await message.answer("系统暂时出现问题，请稍后再试。")


def _build_media_items(result: EngineResult) -> list[MediaItem]:
    items: list[MediaItem] = []

    for image_path in result.local_image_paths:
        items.append((_detect_media_kind(image_path, default="photo"), image_path, True))

    for video_path in result.local_video_paths:
        items.append((_detect_media_kind(video_path, default="video"), video_path, True))

    for image_url in result.generated_image_urls:
        items.append((_detect_media_kind(image_url, default="photo"), image_url, False))

    return items


async def _send_media_sequence(
    message: Message,
    media_items: list[MediaItem],
    *,
    caption: str,
) -> None:
    first_item, *remaining_items = media_items
    first_sent = await _send_single_media(message, first_item, caption=caption)
    if not first_sent:
        await message.answer(caption)

    for item in remaining_items:
        await _send_single_media(message, item)


async def _send_single_media(
    message: Message,
    item: MediaItem,
    *,
    caption: str | None = None,
) -> bool:
    kind, source, is_local = item
    payload = FSInputFile(source) if is_local else source

    try:
        if kind == "animation":
            await message.answer_animation(payload, caption=caption)
        elif kind == "video":
            await message.answer_video(payload, caption=caption, supports_streaming=True)
        else:
            await message.answer_photo(payload, caption=caption)
        return True
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
