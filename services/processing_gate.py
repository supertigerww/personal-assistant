from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class ProcessingGate:
    """Serialize message handling per Telegram user to prevent overlapping Grok calls."""

    def __init__(self) -> None:
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    @asynccontextmanager
    async def acquire(self, telegram_user_id: int) -> AsyncIterator[None]:
        lock = self._locks[telegram_user_id]
        if lock.locked():
            logger.info("User_id=%s message queued behind an in-flight request.", telegram_user_id)
        await lock.acquire()
        try:
            yield
        finally:
            lock.release()