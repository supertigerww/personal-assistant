from __future__ import annotations

import pytest

from core.models import ConversationState, TaskStatus


@pytest.mark.asyncio
async def test_schedule_next_task_uses_current_turn(settings, user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=2, username="tester", display_name="Tester")

    next_turn = await task_service.schedule_next_task(
        telegram_user_id=profile.telegram_user_id,
        state=str(ConversationState.NORMAL),
        from_turn=3,
    )

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert updated.next_task_turn == next_turn
    assert 11 <= next_turn <= 18


@pytest.mark.asyncio
async def test_skip_ignored_task_on_next_turn(user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=3, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Check in",
        instructions="Send a status update in one sentence.",
        issued_at_turn=4,
    )

    skipped = await task_service.skip_ignored_task_if_needed(
        telegram_user_id=profile.telegram_user_id,
        current_turn=5,
        user_text="Changing the subject completely",
    )

    assert skipped is not None
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.SKIPPED


def test_task_ack_detection(task_service):
    assert task_service.looks_like_task_response("done, handled") is True
    assert task_service.looks_like_task_response("收到，马上") is True
    assert task_service.looks_like_task_response("嗯，可以") is True
    assert task_service.looks_like_task_response("let us talk about something else") is False
