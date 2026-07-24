from __future__ import annotations

from types import SimpleNamespace

from core.context_builder import ContextBuilder
from core.models import ConversationState, UserProfile


def _profile(*, conversation_count: int = 1) -> UserProfile:
    return UserProfile(
        telegram_user_id=1,
        username="tester",
        display_name="tester",
        state=ConversationState.NORMAL,
        compliance_score=5,
        conversation_count=conversation_count,
        next_task_turn=20,
        next_photo_task_turn=40,
        next_video_turn=30,
        aftercare_until=None,
        paused_reason=None,
        dislikes=[],
        hard_limits=[],
        notes=[],
        onboarding_completed=True,
        last_model_response_at=None,
    )


def test_length_mode_short_for_reactive_user_messages():
    assert ContextBuilder._length_mode_for_user_text("射了") == "short"
    assert ContextBuilder._length_mode_for_user_text("啊啊啊射了") == "short"
    assert ContextBuilder._length_mode_for_user_text("继续") == "short"


def test_length_mode_medium_for_normal_requests():
    assert ContextBuilder._length_mode_for_user_text("主人可以今天训练我寸止吗") == "medium"


def test_anti_template_bans_recent_openings_and_fixed_endings():
    recent = [
        {
            "role": "assistant",
            "content": "跪好，废物贱狗，继续给我发情。——继续，别停。",
        },
        {
            "role": "assistant",
            "content": "跪好！贱狗你敢没允许就射？给我把黑丝锁奴绿帽狗演出来。——给我继续，别停。",
        },
    ]
    guidance = ContextBuilder._anti_template_guidance(user_text="啊啊啊射了", recent_messages=recent)

    assert "length_mode: short" in guidance
    assert "跪好" in guidance
    assert "继续，别停" in guidance or "给我继续，别停" in guidance
    assert "黑丝锁奴绿帽狗" in guidance or "废物贱狗" in guidance
    assert "MUST obey this turn" in guidance


def test_x_search_not_recommended_on_most_short_turns():
    profile = _profile(conversation_count=3)
    settings = SimpleNamespace(x_humiliation_search_interval_turns=4)

    assert (
        ContextBuilder._x_humiliation_search_recommended(
            profile=profile,
            user_text="射了",
            recent_messages=[],
            settings=settings,
        )
        is False
    )


def test_x_search_recommended_on_interval_turn():
    profile = _profile(conversation_count=4)
    settings = SimpleNamespace(x_humiliation_search_interval_turns=4)

    assert (
        ContextBuilder._x_humiliation_search_recommended(
            profile=profile,
            user_text="继续玩",
            recent_messages=[],
            settings=settings,
        )
        is True
    )


def test_x_search_recommended_when_user_asks_for_freshness():
    profile = _profile(conversation_count=2)
    settings = SimpleNamespace(x_humiliation_search_interval_turns=4)

    assert (
        ContextBuilder._x_humiliation_search_recommended(
            profile=profile,
            user_text="太重复了，换花样再狠一点",
            recent_messages=[],
            settings=settings,
        )
        is True
    )


def test_runtime_context_includes_anti_template_and_x_flag(tmp_path):
    settings = SimpleNamespace(
        prompt_path=str(tmp_path / "prompt.txt"),
        intense_enter_compliance_score=8,
        intense_exit_compliance_score=3,
        x_humiliation_search_interval_turns=4,
    )
    (tmp_path / "prompt.txt").write_text("system", encoding="utf-8")
    builder = ContextBuilder(settings)
    profile = _profile(conversation_count=3)
    messages = builder.build_messages(
        profile=profile,
        user_text="啊啊啊射了",
        recent_messages=[
            {
                "role": "assistant",
                "content": "跪好，废物贱狗。——继续，别停。",
            }
        ],
        active_task=None,
        task_window_ready=False,
        local_media_summary={"images": 0, "videos": 0},
    )

    system = messages[0]["content"]
    assert "Anti-template" in system
    assert "x_humiliation_search_recommended: false" in system
    assert "Do NOT call search_x_humiliation" in system
    assert "跪好" in system
