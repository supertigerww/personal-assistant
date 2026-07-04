from core.media_intent import (
    has_explicit_video_request,
    has_video_send_cue,
    resolve_video_attachment,
)


def test_has_explicit_video_request_detects_chinese_and_english():
    assert has_explicit_video_request("发段视频给我")
    assert has_explicit_video_request("send me a video clip")
    assert not has_explicit_video_request("发张图看看")


def test_has_video_send_cue_detects_setup_language():
    assert has_video_send_cue("我给你看一段，跪好，好好看着。")
    assert not has_video_send_cue("继续跪着，别说话。")


def test_resolve_video_attachment_requires_window_or_explicit_request():
    assert resolve_video_attachment(
        user_text="发段视频",
        response_text="随便说说。",
        video_window_ready=False,
    )
    assert not resolve_video_attachment(
        user_text="继续",
        response_text="我给你看一段，跪好。",
        video_window_ready=False,
    )
    assert resolve_video_attachment(
        user_text="继续",
        response_text="我给你看一段，跪好。",
        video_window_ready=True,
    )