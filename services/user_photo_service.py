from __future__ import annotations

import logging
import shutil
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

        file_path = str(telegram_file.file_path)
        suffix = Path(file_path).suffix.lower() or ".jpg"
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"

        user_dir = self.uploads_path / str(message.from_user.id)
        user_dir.mkdir(parents=True, exist_ok=True)
        destination = user_dir / f"{uuid4().hex}{suffix}"

        await self._download_telegram_file(bot=bot, file_path=file_path, destination=destination)
        logger.info(
            "Saved user photo user_id=%s path=%s file_id=%s source=%s",
            message.from_user.id,
            destination,
            photo.file_id,
            file_path,
        )
        return SavedUserPhoto(
            telegram_user_id=message.from_user.id,
            path=str(destination),
            file_id=photo.file_id,
            caption=message.caption,
        )

    async def _download_telegram_file(self, *, bot: Bot, file_path: str, destination: Path) -> None:
        """Download from local Bot API path or HTTP, with filesystem fallback.

        Local telegram-bot-api with --local returns absolute paths like:
        /var/lib/telegram-bot-api/<token>/photos/file_1.jpg
        Those must be read from disk (is_local) or copied if the volume is mounted.
        """
        source = Path(file_path)
        if source.is_absolute() and source.is_file():
            shutil.copyfile(source, destination)
            logger.debug("Copied local Bot API file from %s to %s", source, destination)
            return

        try:
            await bot.download_file(file_path, destination=destination)
            return
        except Exception as exc:
            logger.warning(
                "bot.download_file failed for %s (%s); trying path rewrites.",
                file_path,
                exc,
            )

        # Sometimes getFile returns absolute path but the bot container mounts data elsewhere.
        candidates = self._candidate_local_paths(file_path)
        for candidate in candidates:
            if candidate.is_file():
                shutil.copyfile(candidate, destination)
                logger.info("Recovered photo via mounted path %s -> %s", candidate, destination)
                return

        # Last resort: strip leading slash / known prefixes and retry HTTP-style download
        relative = file_path.lstrip("/")
        if relative != file_path:
            try:
                await bot.download_file(relative, destination=destination)
                return
            except Exception as exc:
                logger.warning("Relative download_file also failed for %s: %s", relative, exc)

        raise FileNotFoundError(
            "Could not download Telegram photo. "
            f"getFile path={file_path!r}. "
            "If you use local telegram-bot-api with --local, set BOT_API_IS_LOCAL=true and mount "
            "the API data volume into this container at the same path (usually /var/lib/telegram-bot-api)."
        )

    def _candidate_local_paths(self, file_path: str) -> list[Path]:
        raw = Path(file_path)
        candidates: list[Path] = [raw]
        data_root = getattr(self.settings, "telegram_bot_api_data_path", None)
        if data_root:
            root = Path(str(data_root))
            # Absolute API path: /var/lib/telegram-bot-api/<token>/photos/...
            # If we only mounted the data root differently, try joining tail after telegram-bot-api.
            parts = raw.parts
            if "telegram-bot-api" in parts:
                idx = parts.index("telegram-bot-api")
                tail = Path(*parts[idx + 1 :]) if idx + 1 < len(parts) else Path()
                candidates.append(root / tail)
            candidates.append(root / raw.name)
            if not raw.is_absolute():
                candidates.append(root / raw)
        # Dedupe while preserving order
        seen: set[str] = set()
        unique: list[Path] = []
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique
