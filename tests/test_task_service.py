from __future__ import annotations

import pytest

from core.models import ConversationState, TaskFollowupKind, TaskStatus


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
    assert 13 <= next_turn <= 21


@pytest.mark.asyncio
async def test_skip_ignored_task_on_next_turn(user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=3, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Check in",
        instructions="Send a status update in one sentence.",
        issued_at_turn=4,
    )

    result = await task_service.resolve_open_task_followup(
        telegram_user_id=profile.telegram_user_id,
        current_turn=5,
        user_text="Changing the subject completely",
    )

    assert result.kind == TaskFollowupKind.IGNORED
    assert result.task is not None
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.SKIPPED


@pytest.mark.asyncio
async def test_complete_task_on_followup_turn(settings, user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=31, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Kneel",
        instructions="Kneel for one minute.",
        issued_at_turn=2,
    )

    result = await task_service.resolve_open_task_followup(
        telegram_user_id=profile.telegram_user_id,
        current_turn=3,
        user_text="做完了",
    )

    assert result.kind == TaskFollowupKind.COMPLETED
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.COMPLETED
    assert refreshed.completed_at is not None

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert updated.compliance_score == settings.task_completion_score_delta


@pytest.mark.asyncio
async def test_refuse_task_on_followup_turn(settings, user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=32, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Write lines",
        instructions="Write the apology three times.",
        issued_at_turn=6,
    )

    result = await task_service.resolve_open_task_followup(
        telegram_user_id=profile.telegram_user_id,
        current_turn=7,
        user_text="我不想做",
    )

    assert result.kind == TaskFollowupKind.REFUSED
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.REFUSED

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert updated.compliance_score == max(0, settings.task_refusal_score_delta)


@pytest.mark.asyncio
async def test_fail_task_on_followup_turn(settings, user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=33, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Hold position",
        instructions="Hold the pose until told to stop.",
        issued_at_turn=9,
    )

    result = await task_service.resolve_open_task_followup(
        telegram_user_id=profile.telegram_user_id,
        current_turn=10,
        user_text="没做成，失败了",
    )

    assert result.kind == TaskFollowupKind.FAILED
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.FAILED

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert updated.compliance_score == max(0, settings.task_failure_score_delta)


def test_task_followup_classification(task_service):
    assert task_service.classify_task_followup("做完了") == TaskFollowupKind.COMPLETED
    assert task_service.classify_task_followup("done, handled") == TaskFollowupKind.COMPLETED
    assert task_service.classify_task_followup("我不想做") == TaskFollowupKind.REFUSED
    assert task_service.classify_task_followup("没做成") == TaskFollowupKind.FAILED
    assert task_service.classify_task_followup("嗯，可以") is None
    assert task_service.classify_task_followup("let us talk about something else") is None


@pytest.mark.asyncio
async def test_photo_submission_completes_photo_task(settings, user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=35, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Photo proof",
        instructions="拍一张验证照发给我。",
        issued_at_turn=3,
    )

    result = await task_service.resolve_photo_task_submission(profile.telegram_user_id)

    assert result.kind == TaskFollowupKind.PHOTO_SUBMITTED
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.COMPLETED
    assert refreshed.metadata.get("requires_photo") is True

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert updated.compliance_score == settings.task_completion_score_delta


@pytest.mark.asyncio
async def test_photo_submission_ignored_for_non_photo_task(user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=36, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Repeat mantra",
        instructions="Repeat the mantra three times.",
        issued_at_turn=2,
    )

    result = await task_service.resolve_photo_task_submission(profile.telegram_user_id)

    assert result.kind == TaskFollowupKind.NONE
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.OPEN


def test_is_photo_verification_task(task_service):
    assert task_service.is_photo_verification_task("验证", "拍一张丝袜照发过来") is True
    assert task_service.is_photo_verification_task("Mantra", "Repeat three lines") is False


@pytest.mark.asyncio
async def test_evaluate_photo_task_window_can_open(user_service, task_service, monkeypatch):
    profile = await user_service.get_or_create(telegram_user_id=37, username="tester", display_name="Tester")
    await user_service.update_next_photo_task_turn(profile.telegram_user_id, 1)
    profile = await user_service.increment_conversation_count(profile.telegram_user_id)
    profile = await user_service.increment_conversation_count(profile.telegram_user_id)

    monkeypatch.setattr("services.task_service.random.random", lambda: 0.0)

    updated_profile, ready = await task_service.evaluate_photo_task_window(profile=profile, active_task=None)

    assert ready is True
    assert updated_profile.next_photo_task_turn == profile.next_photo_task_turn


@pytest.mark.asyncio
async def test_weak_acknowledgement_is_treated_as_ignored(user_service, task_service):
    profile = await user_service.get_or_create(telegram_user_id=34, username="tester", display_name="Tester")
    task = await task_service.create_task(
        telegram_user_id=profile.telegram_user_id,
        title="Repeat mantra",
        instructions="Repeat the mantra once.",
        issued_at_turn=1,
    )

    result = await task_service.resolve_open_task_followup(
        telegram_user_id=profile.telegram_user_id,
        current_turn=2,
        user_text="嗯，可以",
    )

    assert result.kind == TaskFollowupKind.IGNORED
    refreshed = await task_service.get_task(task.id)
    assert refreshed.status == TaskStatus.SKIPPED


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