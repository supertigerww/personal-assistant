from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.models import ConversationState


@pytest.mark.asyncio
async def test_aftercare_expires_back_to_normal(user_service):
    profile = await user_service.get_or_create(telegram_user_id=21, username="tester", display_name="Tester")
    expired_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    await user_service.update_state(
        profile.telegram_user_id,
        ConversationState.AFTERCARE,
        paused_reason="safeword_red",
        aftercare_until=expired_at,
    )

    synced = await user_service.sync_runtime_state(await user_service.get_profile(profile.telegram_user_id))

    assert synced.state == ConversationState.NORMAL
    assert synced.aftercare_until is None
    assert synced.paused_reason is None


@pytest.mark.asyncio
async def test_compliance_promotes_user_to_intense(user_service):
    profile = await user_service.get_or_create(telegram_user_id=22, username="tester", display_name="Tester")
    await user_service.database.execute(
        "UPDATE users SET compliance_score = 10 WHERE telegram_user_id = ?",
        (profile.telegram_user_id,),
    )

    synced = await user_service.sync_runtime_state(await user_service.get_profile(profile.telegram_user_id))

    assert synced.state == ConversationState.INTENSE


@pytest.mark.asyncio
async def test_compliance_demotes_user_from_intense(user_service):
    profile = await user_service.get_or_create(telegram_user_id=23, username="tester", display_name="Tester")
    await user_service.update_state(profile.telegram_user_id, ConversationState.INTENSE, paused_reason=None)
    await user_service.database.execute(
        "UPDATE users SET compliance_score = 2 WHERE telegram_user_id = ?",
        (profile.telegram_user_id,),
    )

    synced = await user_service.sync_runtime_state(await user_service.get_profile(profile.telegram_user_id))

    assert synced.state == ConversationState.NORMAL