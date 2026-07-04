import json

import pytest

from core.luna_visual import build_scene_image_prompt, load_visual_anchor
from core.models import ConversationState, UserProfile
from services import media_service as media_service_module
from services.media_service import POV_VIDEO_CAPTION_FALLBACKS, MediaService


class StubGrokClient:
    def __init__(self) -> None:
        self.prompts: list[tuple[str, int]] = []
        self.video_caption_calls: list[dict[str, str]] = []
        self.video_caption_response = "手别停，盯紧每一帧。"

    async def generate_image(self, *, prompt: str, count: int = 1) -> list[str]:
        self.prompts.append((prompt, count))
        return [f"https://example.com/generated-{index}.png" for index in range(count)]

    async def generate_video_caption(
        self,
        *,
        video_category: str | None,
        response_text: str,
        user_text: str,
        state: str,
        recent_captions: list[str] | None = None,
    ) -> str:
        self.video_caption_calls.append(
            {
                "video_category": video_category or "",
                "response_text": response_text,
                "user_text": user_text,
                "state": state,
                "recent_captions": list(recent_captions or []),
            }
        )
        return self.video_caption_response


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
        next_photo_task_turn=99,
        next_video_turn=99,
        aftercare_until=None,
        paused_reason=None,
        onboarding_completed=True,
    )


def media_call_kwargs(*, context: str, user_id: int) -> dict:
    return {
        "user_text": context,
        "response_text": "",
        "user_id": user_id,
        "video_window_ready": False,
    }


def expected_scene_prompt(*, settings, scene_prompt: str) -> str:
    return build_scene_image_prompt(
        scene_prompt=scene_prompt,
        visual_anchor=load_visual_anchor(settings),
    )


def test_extract_keywords_supports_chinese_and_english(settings):
    keywords = MediaService._extract_keywords("\u7ea2\u8272 heels_2026 close-up look!")

    assert "\u7ea2\u8272" in keywords
    assert "heels" in keywords
    assert "2026" in keywords
    assert "close" in keywords


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

    result = await service.get_or_generate_media(**media_call_kwargs(context="heels", user_id=profile.telegram_user_id))

    assert result.images == []
    assert result.videos == []


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

    result = await service.get_or_generate_media(**media_call_kwargs(context="heels", user_id=profile.telegram_user_id))

    assert result.images == [str(best_image)]
    assert result.videos == []


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

    prompt = "\u7279\u5199 \u7ea2\u8272 \u706f\u5149 close-up outfit"
    result = await service.get_or_generate_media(
        **media_call_kwargs(context=prompt, user_id=profile.telegram_user_id),
    )

    assert result.images == ["https://example.com/generated-0.png"]
    assert result.videos == []
    assert grok_client.prompts == [(expected_scene_prompt(settings=settings, scene_prompt=prompt), 1)]


@pytest.mark.asyncio
async def test_get_or_generate_media_bypasses_probability_gate_for_generation(settings, tmp_path, monkeypatch):
    settings.enable_image_generation = True
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    profile = build_profile(telegram_user_id=12, state=ConversationState.NORMAL)
    grok_client = StubGrokClient()
    service = MediaService(
        settings=settings,
        grok_client=grok_client,
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.99)

    prompt = "\u7279\u5199 \u9ed1\u8272 heels lighting angle"
    result = await service.get_or_generate_media(
        **media_call_kwargs(context=prompt, user_id=profile.telegram_user_id),
    )

    assert result.images == ["https://example.com/generated-0.png"]
    assert result.videos == []
    assert grok_client.prompts == [(expected_scene_prompt(settings=settings, scene_prompt=prompt), 1)]


@pytest.mark.asyncio
async def test_get_or_generate_media_skips_irrelevant_random_fallback(settings, tmp_path, monkeypatch):
    settings.enable_image_generation = True
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    unrelated_video = tmp_path / "videos" / "general" / "sample.mp4"
    unrelated_video.parent.mkdir(parents=True, exist_ok=True)
    unrelated_video.write_bytes(b"x")

    profile = build_profile(telegram_user_id=10, state=ConversationState.NORMAL)
    grok_client = StubGrokClient()
    service = MediaService(
        settings=settings,
        grok_client=grok_client,
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(
        **media_call_kwargs(context="\u7ee7\u7eed\u8bf4\u8bdd\uff0c\u5feb\u4e00\u70b9", user_id=profile.telegram_user_id),
    )

    assert result.images == []
    assert result.videos == []
    assert grok_client.prompts == []


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
        **media_call_kwargs(
            context="\u5177\u4f53 \u573a\u666f \u7ec6\u8282 \u706f\u5149 \u955c\u5934",
            user_id=profile.telegram_user_id,
        ),
    )

    assert result.images == []
    assert result.videos == []
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


@pytest.mark.asyncio
async def test_meta_tags_boost_asset_selection(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    tagged_image = tmp_path / "images" / "general" / "frame01.jpg"
    filename_match = tmp_path / "images" / "heels" / "heels.jpg"
    tagged_image.parent.mkdir(parents=True, exist_ok=True)
    filename_match.parent.mkdir(parents=True, exist_ok=True)
    tagged_image.write_text("x", encoding="utf-8")
    filename_match.write_text("x", encoding="utf-8")
    tagged_image.with_name("frame01.meta.json").write_text(
        json.dumps({"tags": ["heels", "studio"]}),
        encoding="utf-8",
    )

    profile = build_profile(telegram_user_id=13, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(**media_call_kwargs(context="studio", user_id=profile.telegram_user_id))

    assert result.images == [str(tagged_image)]
    assert result.videos == []


@pytest.mark.asyncio
async def test_repeat_penalty_deprioritizes_recent_asset(settings, tmp_path, database, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")
    settings.media_repeat_penalty_score = 24

    recent_image = tmp_path / "images" / "heels" / "recent.jpg"
    fresh_image = tmp_path / "images" / "heels" / "fresh.jpg"
    recent_image.parent.mkdir(parents=True, exist_ok=True)
    recent_image.write_text("x", encoding="utf-8")
    fresh_image.write_text("x", encoding="utf-8")

    profile = build_profile(telegram_user_id=14, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
        database=database,
    )
    await service.record_deliveries(profile.telegram_user_id, [str(recent_image)])
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(**media_call_kwargs(context="heels", user_id=profile.telegram_user_id))

    assert result.images == [str(fresh_image)]
    assert result.videos == []


@pytest.mark.asyncio
async def test_random_fallback_never_attaches_video(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    image = tmp_path / "images" / "sample.jpg"
    video = tmp_path / "videos" / "sample.mp4"
    image.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    image.write_text("x", encoding="utf-8")
    video.write_bytes(b"x")

    profile = build_profile(telegram_user_id=15, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    result = await service.get_or_generate_media(**media_call_kwargs(context="嗯", user_id=profile.telegram_user_id))

    assert result.videos == []
    if result.images:
        assert result.images == [str(image)]


@pytest.mark.asyncio
async def test_video_attaches_only_with_setup_cue(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    clip = tmp_path / "videos" / "heels" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")
    clip.with_name("clip.meta.json").write_text(
        json.dumps({"tags": ["heels"], "caption": "这段是专门给你准备的。"}),
        encoding="utf-8",
    )

    profile = build_profile(telegram_user_id=16, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    without_cue = await service.get_or_generate_media(
        user_text="heels",
        response_text="继续跪着，别说话。",
        user_id=profile.telegram_user_id,
        video_window_ready=True,
    )
    assert without_cue.videos == []

    with_cue = await service.get_or_generate_media(
        user_text="heels",
        response_text="我给你看一段，跪好，好好看着。",
        user_id=profile.telegram_user_id,
        video_window_ready=True,
    )
    assert with_cue.videos == [str(clip)]
    assert with_cue.text_before_video is True
    assert with_cue.video_caption is None


@pytest.mark.asyncio
async def test_video_folder_category_matches_chinese_alias(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")
    settings.video_folder_aliases_csv = "sm=调教,SM;pov=第一视角,撸,寸止"

    sm_clip = tmp_path / "videos" / "sm" / "clip.mp4"
    pov_clip = tmp_path / "videos" / "pov" / "clip.mp4"
    sm_clip.parent.mkdir(parents=True, exist_ok=True)
    pov_clip.parent.mkdir(parents=True, exist_ok=True)
    sm_clip.write_bytes(b"x")
    pov_clip.write_bytes(b"x")

    profile = build_profile(telegram_user_id=17, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    pov_result = await service.get_or_generate_media(
        user_text="继续",
        response_text="盯着屏幕，第一视角给我使劲撸，寸止，我说停才准停。",
        user_id=profile.telegram_user_id,
        video_window_ready=True,
    )
    assert pov_result.videos == [str(pov_clip)]

    sm_result = await service.get_or_generate_media(
        user_text="继续",
        response_text="跪好，我给你看一段调教示范，好好学着。",
        user_id=profile.telegram_user_id,
        video_window_ready=True,
    )
    assert sm_result.videos == [str(sm_clip)]


@pytest.mark.asyncio
async def test_asset_summary_lists_video_categories(settings, tmp_path):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")

    (tmp_path / "videos" / "sm").mkdir(parents=True)
    (tmp_path / "videos" / "pov").mkdir(parents=True)
    (tmp_path / "videos" / "sm" / "a.mp4").write_bytes(b"x")
    (tmp_path / "videos" / "pov" / "b.mp4").write_bytes(b"x")
    (tmp_path / "videos" / "pov" / "c.mp4").write_bytes(b"x")

    service = MediaService(settings=settings, grok_client=StubGrokClient())
    summary = await service.asset_summary()

    assert summary["videos"] == 3
    assert summary["video_categories"] == {"sm": 1, "pov": 2}
    assert "pov(2)" in service.video_categories_context(media_summary=summary)


@pytest.mark.asyncio
async def test_generate_video_caption_uses_llm(settings, tmp_path):
    settings.enable_llm_video_caption = True
    settings.assets_videos_path = str(tmp_path / "videos")
    clip = tmp_path / "videos" / "pov" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")

    grok_client = StubGrokClient()
    grok_client.video_caption_response = "寸止，憋到我说停。"
    service = MediaService(settings=settings, grok_client=grok_client)
    profile = build_profile(telegram_user_id=18, state=ConversationState.INTENSE)

    caption = await service.generate_video_caption(
        video_path=str(clip),
        user_text="继续",
        response_text="盯着屏幕使劲撸。",
        profile=profile,
        recent_captions=["好好看着。"],
    )

    assert caption == "寸止，憋到我说停。"
    assert grok_client.video_caption_calls
    assert grok_client.video_caption_calls[0]["video_category"] == "pov"
    assert grok_client.video_caption_calls[0]["recent_captions"] == ["好好看着。"]


@pytest.mark.asyncio
async def test_fallback_video_caption_avoids_recent_history(settings, tmp_path):
    settings.enable_llm_video_caption = False
    settings.assets_videos_path = str(tmp_path / "videos")
    clip = tmp_path / "videos" / "pov" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")

    service = MediaService(settings=settings, grok_client=StubGrokClient())
    blocked = POV_VIDEO_CAPTION_FALLBACKS[0]
    caption = await service.generate_video_caption(
        video_path=str(clip),
        user_text="继续",
        response_text="盯着屏幕。",
        recent_captions=[blocked],
    )

    assert caption != blocked


@pytest.mark.asyncio
async def test_video_rotation_picks_from_same_category_pool(settings, tmp_path, monkeypatch):
    settings.assets_images_path = str(tmp_path / "images")
    settings.assets_videos_path = str(tmp_path / "videos")
    settings.video_folder_aliases_csv = "pov=第一视角,撸,寸止"
    settings.video_rotation_score_band = 12

    for name in ("clip_a.mp4", "clip_b.mp4", "clip_c.mp4"):
        clip = tmp_path / "videos" / "pov" / name
        clip.parent.mkdir(parents=True, exist_ok=True)
        clip.write_bytes(b"x")

    profile = build_profile(telegram_user_id=19, state=ConversationState.NORMAL)
    service = MediaService(
        settings=settings,
        grok_client=StubGrokClient(),
        user_service=StubUserService(profile),
    )
    monkeypatch.setattr(media_service_module.random, "random", lambda: 0.0)

    rotation_state = {"index": 0}

    def cycling_choice(seq):
        choice = seq[rotation_state["index"] % len(seq)]
        rotation_state["index"] += 1
        return choice

    monkeypatch.setattr(media_service_module.random, "choice", cycling_choice)

    selected: set[str] = set()
    for index in range(12):
        result = await service.get_or_generate_media(
            user_text="继续",
            response_text=f"第一视角给我使劲撸，好好看着，轮次{index}。",
            user_id=profile.telegram_user_id,
            video_window_ready=True,
        )
        if result.videos:
            selected.add(result.videos[0])

    assert len(selected) >= 2


@pytest.mark.asyncio
async def test_generate_scene_image_wraps_luna_anchor(settings):
    grok_client = StubGrokClient()
    service = MediaService(settings=settings, grok_client=grok_client)
    scene_prompt = "特写 红色 lighting angle"

    result = await service.generate_scene_image(prompt=scene_prompt, count=1)

    assert result == ["https://example.com/generated-0.png"]
    assert grok_client.prompts == [(expected_scene_prompt(settings=settings, scene_prompt=scene_prompt), 1)]
