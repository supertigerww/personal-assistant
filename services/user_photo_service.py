from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from core.models import SavedUserPhoto

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message

logger = logging.getLogger(__name__)


class UserPhotoService:
    SUFFIX_BY_MIME = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    def __init__(self, *, settings: Any) -> None:
        self.settings = settings
        self.uploads_path = Path(settings.user_uploads_path)
        self.uploads_path.mkdir(parents=True, exist_ok=True)

    async def save_telegram_photo(self, *, bot: Bot, message: Message) -> SavedUserPhoto:  # type: ignore[name-defined]
        if not message.photo:
            raise ValueError("Message does not contain a photo.")

        photo = message.photo[-1]
        telegram_file = await bot.get_file(photo.file_id)
        if telegram_file.file_path is None:
            raise RuntimeError(f"Telegram file path missing for file_id={photo.file_id}")

        suffix = Path(telegram_file.file_path).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        user_dir = self.uploads_path / str(message.from_user.id)
        user_dir.mkdir(parents=True, exist_ok=True)
        destination = user_dir / f"{uuid4().hex}{suffix}"

        await bot.download_file(telegram_file.file_path, destination=destination)
        logger.info(
            "Saved user photo user_id=%s path=%s file_id=%s",
            message.from_user.id,
            destination,
            photo.file_id,
        )
        return SavedUserPhoto(
            telegram_user_id=message.from_user.id,
            path=str(destination),
            file_id=photo.file_id,
            caption=message.caption,
        )