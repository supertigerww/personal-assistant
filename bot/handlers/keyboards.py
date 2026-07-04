from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from core.models import EngineResult
from core.reply_utils import should_show_quick_replies

QUICK_DONE = "quick:done"
QUICK_REFUSE = "quick:refuse"
QUICK_SLOW = "quick:slow"

QUICK_ACTION_TEXT = {
    QUICK_DONE: "完成了",
    QUICK_REFUSE: "做不到",
    QUICK_SLOW: "黄色",
}


def build_quick_reply_keyboard(result: EngineResult) -> InlineKeyboardMarkup | None:
    if not result.show_quick_replies:
        return None

    buttons: list[InlineKeyboardButton] = []
    if result.has_open_task:
        buttons.extend(
            [
                InlineKeyboardButton(text="完成了", callback_data=QUICK_DONE),
                InlineKeyboardButton(text="做不到", callback_data=QUICK_REFUSE),
            ]
        )
    buttons.append(InlineKeyboardButton(text="慢一点", callback_data=QUICK_SLOW))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


__all__ = [
    "QUICK_ACTION_TEXT",
    "QUICK_DONE",
    "QUICK_REFUSE",
    "QUICK_SLOW",
    "build_quick_reply_keyboard",
    "should_show_quick_replies",
]