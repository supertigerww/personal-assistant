from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MediaDeliveryService:
    def __init__(self, *, database: Any, settings: Any) -> None:
        self.database = database
        self.settings = settings

    async def record_deliveries(self, telegram_user_id: int, asset_paths: list[str]) -> None:
        if self.database is None:
            return

        cleaned_paths = [path.strip() for path in asset_paths if path and path.strip()]
        if not cleaned_paths:
            return

        delivered_at = utc_now_iso()
        for asset_path in cleaned_paths:
            await self.database.execute(
                """
                INSERT INTO media_deliveries (telegram_user_id, asset_path, delivered_at)
                VALUES (?, ?, ?)
                """,
                (telegram_user_id, asset_path, delivered_at),
            )
        logger.info(
            "Recorded %s media delivery(ies) for user_id=%s",
            len(cleaned_paths),
            telegram_user_id,
        )

    async def recently_delivered_paths(self, telegram_user_id: int) -> set[str]:
        if self.database is None:
            return set()

        hours = max(1, int(getattr(self.settings, "media_repeat_cooldown_hours", 48)))
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = await self.database.fetchall(
            """
            SELECT DISTINCT asset_path
            FROM media_deliveries
            WHERE telegram_user_id = ? AND delivered_at >= ?
            """,
            (telegram_user_id, cutoff),
        )
        return {row["asset_path"] for row in rows}

    async def get_all_delivered_folders(self, telegram_user_id: int) -> set[str]:
        """Return set of top-level folders (e.g. 'Linmistresssh') that have ever been delivered to this user."""
        if self.database is None:
            return set()

        rows = await self.database.fetchall(
            """
            SELECT DISTINCT asset_path
            FROM media_deliveries
            WHERE telegram_user_id = ?
            """,
            (telegram_user_id,),
        )
        folders: set[str] = set()
        for row in rows:
            path = row["asset_path"] or ""
            try:
                parts = Path(path).parts
                if parts:
                    folders.add(parts[0])
            except Exception:
                continue
        return folders