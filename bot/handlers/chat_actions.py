from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from aiogram import Bot
from aiogram.enums import ChatAction


@asynccontextmanager
async def keep_typing(bot: Bot, chat_id: int, *, interval_seconds: float = 4.0) -> AsyncIterator[None]:
    await bot.send_chat_action(chat_id, ChatAction.TYPING)

    async def _refresh_typing() -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                await bot.send_chat_action(chat_id, ChatAction.TYPING)
        except asyncio.CancelledError:
            raise

    refresh_task = asyncio.create_task(_refresh_typing())
    try:
        yield
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass