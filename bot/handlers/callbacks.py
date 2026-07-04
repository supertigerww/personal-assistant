from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.types import CallbackQuery

from bot.handlers.keyboards import QUICK_ACTION_TEXT
from bot.handlers.messages import _reply_with_engine_result, _run_engine_turn
from core.config import Settings
from core.queen_engine import QueenEngine
from services.media_service import MediaService
from services.memory_service import MemoryService
from services.processing_gate import ProcessingGate
from services.user_service import UserService

logger = logging.getLogger(__name__)
router = Router(name="callbacks")


@router.callback_query(F.data.in_(set(QUICK_ACTION_TEXT)))
async def quick_action_callback(
    callback: CallbackQuery,
    engine: QueenEngine,
    processing_gate: ProcessingGate,
    media_service: MediaService,
    user_service: UserService,
    memory_service: MemoryService,
    settings: Settings,
) -> None:
    if callback.from_user is None or callback.message is None:
        await callback.answer("无法处理该操作。", show_alert=True)
        return

    action_text = QUICK_ACTION_TEXT[callback.data]
    await callback.answer()

    try:
        result = await _run_engine_turn(
            message=callback.message,
            processing_gate=processing_gate,
            handler=lambda: engine.handle_text_message(
                telegram_user_id=callback.from_user.id,
                username=callback.from_user.username,
                display_name=callback.from_user.full_name,
                text=action_text,
            ),
        )
        await _reply_with_engine_result(
            callback.message,
            result,
            media_service=media_service,
            user_service=user_service,
            memory_service=memory_service,
            settings=settings,
        )
    except Exception as exc:
        logger.exception(
            "Error in quick_action_callback for user_id=%s data=%s: %s",
            callback.from_user.id,
            callback.data,
            exc,
        )
        await callback.message.answer("系统暂时出现问题，请稍后再试。")