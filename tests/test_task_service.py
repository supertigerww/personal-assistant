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


@pytest.mark.asyncio
async def test_evaluate_task_window_can_defer_eligible_turn(settings, user_service, task_service, monkeypatch):
    profile = await user_service.get_or_create(telegram_user_id=4, username="tester", display_name="Tester")
    await user_service.update_next_task_turn(profile.telegram_user_id, 3)

    profile = await user_service.increment_conversation_count(profile.telegram_user_id)
    profile = await user_service.increment_conversation_count(profile.telegram_user_id)
    profile = await user_service.increment_conversation_count(profile.telegram_user_id)

    monkeypatch.setattr("services.task_service.random.random", lambda: 0.95)
    monkeypatch.setattr("services.task_service.random.randint", lambda lower, upper: upper)

    updated_profile, ready = await task_service.evaluate_task_window(profile=profile, active_task=None)

    assert ready is False
    assert updated_profile.next_task_turn == profile.conversation_count + settings.task_retry_max_turns_normal


@pytest.mark.asyncio
async def test_evaluate_task_window_can_open_eligible_turn(user_service, task_service, monkeypatch):
    profile = await user_service.get_or_create(telegram_user_id=5, username="tester", display_name="Tester")
    await user_service.update_next_task_turn(profile.telegram_user_id, 2)

    profile = await user_service.increment_conversation_count(profile.telegram_user_id)
    profile = await user_service.increment_conversation_count(profile.telegram_user_id)

    monkeypatch.setattr("services.task_service.random.random", lambda: 0.0)

    updated_profile, ready = await task_service.evaluate_task_window(profile=profile, active_task=None)

    assert ready is True
    assert updated_profile.next_task_turn == profile.next_task_turn
