from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.models import ConversationState, SafewordLevel
from services.safety_service import SafetyService


@pytest.mark.asyncio
async def test_red_safeword_switches_to_aftercare(settings, user_service, task_service, caplog):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    profile = await user_service.get_or_create(telegram_user_id=1, username="tester", display_name="Tester")

    assert service.classify_safeword("red") == SafewordLevel.RED
    assert service.classify_safeword("红色") == SafewordLevel.RED
    assert service.classify_safeword("stop") == SafewordLevel.RED
    assert service.classify_safeword("停") is None
    assert service.classify_safeword("结束") is None

    with caplog.at_level("INFO"):
        decision = await service.handle_safeword(profile, level=SafewordLevel.RED)

    updated = await user_service.get_profile(1)
    aftercare_until = datetime.fromisoformat(updated.aftercare_until)
    delta_minutes = (aftercare_until - datetime.now(timezone.utc)).total_seconds() / 60

    assert decision.triggered is True
    assert decision.state == ConversationState.AFTERCARE
    assert decision.safeword_level == SafewordLevel.RED
    assert updated.state == ConversationState.AFTERCARE
    assert updated.paused_reason == "safeword_red"
    assert 40 <= delta_minutes <= 45


@pytest.mark.asyncio
async def test_yellow_safeword_demotes_intense_to_normal(settings, user_service, task_service):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    profile = await user_service.get_or_create(telegram_user_id=11, username="tester", display_name="Tester")
    await user_service.update_state(profile.telegram_user_id, ConversationState.INTENSE, paused_reason=None)

    assert service.classify_safeword("yellow") == SafewordLevel.YELLOW
    assert service.classify_safeword("黄色") == SafewordLevel.YELLOW

    decision = await service.handle_safeword(
        await user_service.get_profile(profile.telegram_user_id),
        level=SafewordLevel.YELLOW,
    )

    updated = await user_service.get_profile(profile.telegram_user_id)
    assert decision.state == ConversationState.NORMAL
    assert decision.safeword_level == SafewordLevel.YELLOW
    assert updated.state == ConversationState.NORMAL
    assert updated.paused_reason == "safeword_yellow"


@pytest.mark.asyncio
async def test_yellow_safeword_keeps_normal_state(settings, user_service, task_service):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    profile = await user_service.get_or_create(telegram_user_id=12, username="tester", display_name="Tester")

    decision = await service.handle_safeword(profile, level=SafewordLevel.YELLOW)
    updated = await user_service.get_profile(profile.telegram_user_id)

    assert decision.state == ConversationState.NORMAL
    assert updated.state == ConversationState.NORMAL
    assert updated.paused_reason is None


def test_extract_limits_finds_explicit_boundary(settings, user_service, task_service):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    assert service.extract_limits("I don't like public photos") == ["public photos"]
    assert service.extract_limits("不喜欢喊名字") == ["喊名字"]