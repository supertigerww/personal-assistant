from __future__ import annotations

import pytest

from services.onboarding_service import OnboardingService


def test_parse_setup_text_extracts_preferences():
    parsed = OnboardingService.parse_setup_text(
        "叫我贱狗。硬限：公开羞辱。强度：重"
    )

    assert parsed["nickname"] == "贱狗"
    assert parsed["hard_limits"] == ["公开羞辱"]
    assert parsed["intensity"] == "重"


@pytest.mark.asyncio
async def test_complete_from_user_text_marks_onboarding_done(database, settings, user_service):
    memory_service = None
    service = OnboardingService(user_service=user_service, memory_service=memory_service)
    profile = await user_service.get_or_create(telegram_user_id=51, username="t", display_name="Tester")
    assert profile.onboarding_completed is False

    updated = await service.complete_from_user_text(
        profile.telegram_user_id,
        "叫我小狗。硬限：实拍",
    )

    assert updated.onboarding_completed is True
    refreshed = await user_service.get_profile(profile.telegram_user_id)
    assert "小狗" in " ".join(refreshed.notes)
    assert "实拍" in refreshed.hard_limits