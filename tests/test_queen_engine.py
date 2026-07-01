from __future__ import annotations

from core.queen_engine import QueenEngine


def test_finalize_media_outputs_drops_local_media_when_generated_images_exist():
    engine = QueenEngine.__new__(QueenEngine)

    local_images, local_videos, generated_urls = engine._finalize_media_outputs(
        media_bundle={
            "images": ["assets/images/local.jpg", "https://example.com/generated-1.png"],
            "videos": ["assets/videos/local.mp4"],
        },
        generated_urls=["https://example.com/generated-0.png"],
        user_id=1,
    )

    assert local_images == []
    assert local_videos == []
    assert generated_urls == [
        "https://example.com/generated-0.png",
        "https://example.com/generated-1.png",
    ]


def test_finalize_media_outputs_keeps_local_media_when_no_generated_images():
    engine = QueenEngine.__new__(QueenEngine)

    local_images, local_videos, generated_urls = engine._finalize_media_outputs(
        media_bundle={
            "images": ["assets/images/local.jpg"],
            "videos": ["assets/videos/local.mp4"],
        },
        generated_urls=[],
        user_id=1,
    )

    assert local_images == ["assets/images/local.jpg"]
    assert local_videos == ["assets/videos/local.mp4"]
    assert generated_urls == []
