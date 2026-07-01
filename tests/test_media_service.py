from __future__ import annotations

import random

import pytest

from core.models import ConversationState, UserProfile
from services import media_service as media_service_module
from services.media_service import MediaService


class StubGrokClient:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, int]] = []

    async def generate_image(self, *, prompt: str, count: int = 1) -> list[str]:
        self.prompts.append((prompt, count))
        return [f"https://example.com/generated-{index}.png" for index in range(count)]


class StubUserService:
    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile

    async def get_profile(self, telegram_user_id: int) -> UserProfile:
        assert telegram_user_id == self.profile.telegram_user_id
        return self.profile


def build_profile(*, telegram_user_id: int, state: ConversationState) -> UserProfile:
    return UserProfile(
        telegram_user_id=telegram_user_id,
        username="tester",
        display_name="Tester",
        state=state,
        compliance_score=0,
        conversation_count=0,
        next_task_turn=6,
        aftercare_until=None,
        paused_reason=None,
    )


def test_extract_keywords_supports_chinese_and_english(settings):
    keywords = MediaService._extract_keywords("红色 heels_2026 close-up look!")

    assert "红色" in keywords
    assert "heels" in keywords
    assert "2026" in keywords
    assert "close" in keywords or "closeup" in keywords


@pytest.mark.asyncio
async def test_pick_relevant_assets_prioritizes_matching_subfolder(settings, tmp_path):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    folder_match = tmp_path / "images" / "heels" / "frame01.jpg"
    filename_match = tmp_path / "images" / "general" / "heels.jpg"
    folder_match.parent.mkdir(parents=True, exist_ok=True)
    filename_match.parent.mkdir(parents=True, exist_ok=True)
    folder_match.write_text("x", encoding="utf-8")
    filename_match.write_text("x", encoding="utf-8")

    service = MediaService(settings=settings, grok_client=StubGrokClient())
    result = await service.pick_relevant_assets(text="heels")

    assert result["images"] == [str(folder_match)]
    assert result["videos"] == []


@pytest.mark.asyncio
async def test_get_or_generate_media_can_skip_media_when_probability_gate_blocks(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    image = tmp_path / "images" / "heels" / "frame01.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("x", encoding="utf-8")

    profile = build_profile(telegram_user_id=7, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.99)

    result = await service.get_or_generate_media(context="heels", user_id=profile.telegram_user_id)

    assert result == {"images": [], "videos": []}


@pytest.mark.asyncio
async def test_get_or_generate_media_returns_single_best_asset(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    best_image = tmp_path / "images" / "heels" / "frame01.jpg"
    weaker_video = tmp_path / "videos" / "general" / "heels.mp4"
    best_image.parent.mkdir(parents=True, exist_ok=True)
    weaker_video.parent.mkdir(parents=True, exist_ok=True)
    best_image.write_text("x", encoding="utf-8")
    weaker_video.write_text("x", encoding="utf-8")

    profile = build_profile(telegram_user_id=8, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(context="heels", user_id=profile.telegram_user_id)

    assert result["images"] == [str(best_image)]
    assert result["videos"] == []


@pytest.mark.asyncio
async def test_get_or_generate_media_generates_when_local_match_is_weak(settings, tmp_path, monkeypatch):
    settings.enable_image_generation = True
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    profile = build_profile(telegram_user_id=9, state=ConversationState.NORMAL)
    grok_client = StubGrokClient()
    service = MediaService(
        settings=settings,
        grok_client=grok_client,
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(
        context="特殊场景 红色灯光 close-up outfit",
        user_id=profile.telegram_user_id,
    )

    assert result["images"] == ["https://example.com/generated-0.png"]
    assert result["videos"] == []
    assert grok_client.prompts == [("特殊场景 红色灯光 close-up outfit", 1)]


@pytest.mark.asyncio
async def test_should_not_generate_during_aftercare(settings, tmp_path, monkeypatch):
    settings.enable_image_generation = True
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    profile = build_profile(telegram_user_id=11, state=ConversationState.AFTERCARE)
    grok_client = StubGrokClient()
    service = MediaService(
        settings=settings,
        grok_client=grok_client,
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(
        context="具体场景 细节 灯光 镜头",
        user_id=profile.telegram_user_id,
    )

    assert result == {"images": [], "videos": []}
    assert grok_client.prompts == []


@pytest.mark.asyncio
async def test_oversized_videos_are_filtered_out(settings, tmp_path):
    settings.assets_videos_path = str(tmp_path / "videos")
    settings.max_local_video_size_mb = 1

    allowed = tmp_path / "videos" / "nested" / "small.mp4"
    oversized = tmp_path / "videos" / "nested" / "huge.mp4"
    allowed.parent.mkdir(parents=True, exist_ok=True)
    allowed.write_bytes(b"x" * 1024)
    oversized.write_bytes(b"x" * (1024 * 1024 + 1))

    service = MediaService(settings=settings, grok_client=StubGrokClient())

    summary = await service.asset_summary()
    fallback = await service.get_random_assets(video_count=2)

    assert summary["videos"] == 1
    assert fallback["videos"] == [str(allowed)]
    assert fallback["images"] == []
