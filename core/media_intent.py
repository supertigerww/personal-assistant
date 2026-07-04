from __future__ import annotations

from dataclasses import dataclass


VIDEO_EXPLICIT_MARKERS = (
    "视频",
    "录像",
    "短片",
    "片段",
    "动图",
    "动态",
    "发段",
    "来段",
    "一段",
    "clip",
    "video",
    "footage",
    "watch",
)

VIDEO_SEND_CUE_MARKERS = (
    "给你看",
    "发给你",
    "好好看着",
    "好好看",
    "看清楚",
    "睁大眼睛",
    "不许移开",
    "这段",
    "视频里",
    "录好了",
    "拍好了",
    "盯着看",
    "盯着屏幕",
    "盯着",
    "看完",
    "跪下看",
    "跪好看",
    "示范",
    "学着",
    "使劲撸",
    "给我撸",
    "手放好",
    "跟着节奏",
    "现在开始",
    "准备好了吗",
    "我要你",
    "看着这个贱货",
    "学着这个样子",
    "记住这个画面",
    "等会儿照着做",
    "先给我跪好",
    "眼睛别眨",
    "一字不漏看完",
    "这是给你的",
    "特别给你",
    "watch this",
    "watch closely",
    "look at this",
    "pay attention",
    "focus on",
)


@dataclass(slots=True)
class MediaTurnHints:
    user_wants_video: bool
    user_wants_image: bool
    video_window_ready: bool
    response_sets_up_video: bool = False

    @property
    def may_attach_video(self) -> bool:
        return self.user_wants_video or (
            self.video_window_ready and self.response_sets_up_video
        )


def has_explicit_video_request(text: str) -> bool:
    normalized = text.strip().casefold()
    if not normalized:
        return False
    return any(marker.casefold() in normalized for marker in VIDEO_EXPLICIT_MARKERS)


def has_video_send_cue(text: str) -> bool:
    normalized = text.strip().casefold()
    if not normalized:
        return False
    return any(marker.casefold() in normalized for marker in VIDEO_SEND_CUE_MARKERS)


def has_explicit_image_request(text: str) -> bool:
    normalized = text.strip().casefold()
    if not normalized:
        return False

    image_markers = (
        "图",
        "图片",
        "照片",
        "来张图",
        "发张图",
        "发图",
        "来张",
        "image",
        "images",
        "photo",
        "picture",
        "pic",
    )
    if not any(marker in normalized for marker in image_markers):
        return False
    return not has_explicit_video_request(text)


def build_turn_hints(*, user_text: str, video_window_ready: bool) -> MediaTurnHints:
    return MediaTurnHints(
        user_wants_video=has_explicit_video_request(user_text),
        user_wants_image=has_explicit_image_request(user_text),
        video_window_ready=video_window_ready,
    )


def resolve_video_attachment(
    *,
    user_text: str,
    response_text: str,
    video_window_ready: bool,
) -> bool:
    if has_explicit_video_request(user_text):
        return True
    if not video_window_ready:
        return False
    return has_video_send_cue(response_text)