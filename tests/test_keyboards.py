from __future__ import annotations

from core.models import ConversationState
from core.reply_utils import should_show_quick_replies, split_reply_text


def test_should_show_quick_replies_for_open_task():
    assert should_show_quick_replies(state=ConversationState.NORMAL, has_open_task=True) is True


def test_should_hide_quick_replies_in_aftercare():
    assert should_show_quick_replies(state=ConversationState.AFTERCARE, has_open_task=True) is False


def test_split_reply_text_breaks_long_message():
    long_text = "第一段内容。\n\n" + ("很长" * 120)
    chunks = split_reply_text(long_text, max_len=100)

    assert len(chunks) >= 2