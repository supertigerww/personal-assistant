from core.models import Task, TaskIntensity, TaskStatus
from core.queen_engine import QueenEngine
from services.media_service import MediaBundle


def test_finalize_media_outputs_drops_local_media_when_generated_images_exist():
    engine = QueenEngine.__new__(QueenEngine)

    local_images, local_videos, generated_urls, video_caption, text_before_video, suggested = engine._finalize_media_outputs(
        media_bundle=MediaBundle(
            images=["assets/images/local.jpg", "https://example.com/generated-1.png"],
            videos=["assets/videos/local.mp4"],
            video_caption="好好看着。",
            text_before_video=True,
        ),
        generated_urls=["https://example.com/generated-0.png"],
        user_id=1,
    )

    assert local_images == []
    assert local_videos == []
    assert generated_urls == [
        "https://example.com/generated-0.png",
        "https://example.com/generated-1.png",
    ]
    assert video_caption is None
    assert text_before_video is False
    assert suggested is None


def test_finalize_media_outputs_keeps_local_media_when_no_generated_images():
    engine = QueenEngine.__new__(QueenEngine)

    local_images, local_videos, generated_urls, video_caption, text_before_video, suggested = engine._finalize_media_outputs(
        media_bundle=MediaBundle(
            images=["assets/images/local.jpg"],
            videos=["assets/videos/local.mp4"],
            video_caption="跪好看完。",
            text_before_video=True,
        ),
        generated_urls=[],
        user_id=1,
    )

    assert local_images == ["assets/images/local.jpg"]
    assert local_videos == ["assets/videos/local.mp4"]
    assert generated_urls == []
    assert video_caption == "跪好看完。"
    assert text_before_video is True
    assert suggested is None


def test_build_media_context_prefers_user_text_over_model_reply():
    task = Task(
        id="task-1",
        telegram_user_id=1,
        title="fallback task",
        instructions="fallback instructions",
        status=TaskStatus.OPEN,
        intensity=TaskIntensity.NORMAL,
        created_at="2026-07-01T00:00:00Z",
        due_at=None,
        issued_at_turn=3,
        completed_at=None,
        skipped_at=None,
        source="test",
    )

    context = QueenEngine._build_media_context(
        user_text="red heels close-up",
        response_text="generic reply text that should not drive media selection",
        created_task=task,
    )

    assert context == "red heels close-up"


def test_build_media_context_falls_back_when_user_text_is_empty():
    context = QueenEngine._build_media_context(
        user_text="   ",
        response_text="visual fallback",
        created_task=None,
    )

    assert context == "visual fallback"


def test_sanitize_response_text_removes_generation_placeholders():
    cleaned = QueenEngine._sanitize_response_text("先看着。\n（生成中...）")

    assert cleaned == "先看着。"


def test_sanitize_response_text_can_return_empty_for_placeholder_only():
    cleaned = QueenEngine._sanitize_response_text("（图片生成中……）")

    assert cleaned == ""


def test_build_photo_submission_text_includes_description_and_caption():
    text = QueenEngine._build_photo_submission_text(
        caption="完成拍照",
        photo_description="一双穿着黑丝的长腿。",
    )

    assert "[用户发送了一张照片]" in text
    assert "黑丝" in text
    assert "完成拍照" in text


def test_build_photo_submission_text_without_description():
    text = QueenEngine._build_photo_submission_text(caption=None, photo_description="")

    assert "尚未生成视觉描述" in text
