from __future__ import annotations

from core.models import ConversationState, EngineResult
from bot.handlers.messages import _build_media_items, _detect_media_kind


def test_detect_media_kind_supports_gif_and_video():
    assert _detect_media_kind("assets/images/loop.gif", default="photo") == "animation"
    assert _detect_media_kind("assets/videos/clip.mp4", default="photo") == "video"
    assert _detect_media_kind("https://example.com/image.webp?size=large", default="photo") == "photo"


def test_build_media_items_preserves_expected_order():
    result = EngineResult(
        text="reply",
        state=ConversationState.NORMAL,
        local_image_paths=["assets/images/first.jpg", "assets/images/loop.gif"],
        local_video_paths=["assets/videos/clip.mp4"],
        generated_image_urls=["https://example.com/generated.png"],
    )

    items = _build_media_items(result)

    assert items == [
        ("photo", "assets/images/first.jpg", True),
        ("animation", "assets/images/loop.gif", True),
        ("video", "assets/videos/clip.mp4", True),
        ("photo", "https://example.com/generated.png", False),
    ]
