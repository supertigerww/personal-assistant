from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.models import ConversationState
from services.safety_service import SafetyService


@pytest.mark.asyncio
async def test_safeword_switches_to_aftercare(settings, user_service, task_service, caplog):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    profile = await user_service.get_or_create(telegram_user_id=1, username="tester", display_name="Tester")

    assert service.detect_safeword("red") is True
    assert service.detect_safeword("停") is True
    assert service.detect_safeword("结束") is True
    assert service.detect_safeword("暂停") is True

    with caplog.at_level("INFO"):
        decision = await service.handle_safeword(profile)

    updated = await user_service.get_profile(1)
    aftercare_until = datetime.fromisoformat(updated.aftercare_until)
    delta_minutes = (aftercare_until - datetime.now(timezone.utc)).total_seconds() / 60

    assert decision.triggered is True
    assert decision.state == ConversationState.AFTERCARE
    assert updated.state == ConversationState.AFTERCARE
    assert updated.paused_reason == "safeword"
    assert decision.reply == (
        "安全词已接收。一切停止。\n"
        "现在深呼吸，慢慢来。\n"
        "告诉我你现在感觉如何？我在这里陪着你。"
    )
    assert 40 <= delta_minutes <= 45
    assert "Safeword triggered for user 1, reason: safeword" in caplog.text


def test_extract_limits_finds_explicit_boundary(settings, user_service, task_service):
    service = SafetyService(settings=settings, user_service=user_service, task_service=task_service)
    assert service.extract_limits("I don't like public photos") == ["public photos"]
    assert service.extract_limits("不喜欢喊名字") == ["喊名字"]
