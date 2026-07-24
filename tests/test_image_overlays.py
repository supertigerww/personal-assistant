from core.image_overlays import (
    build_overlay_instruction_block,
    extract_user_requested_phrases,
    rewrite_scene_without_chat_echo,
    select_humiliation_overlays,
)


def test_select_overlays_returns_multiple_short_phrases():
    phrases = select_humiliation_overlays("我在看av主人 寸止 不许射", count=3)
    assert 2 <= len(phrases) <= 5
    assert all(2 <= len(p) <= 16 for p in phrases)
    # Theme denial should surface somewhere in related pools; at least not empty.
    assert any(p for p in phrases)


def test_select_overlays_stable_for_same_context():
    context = "跪着看女王训练 废物"
    a = select_humiliation_overlays(context, count=3)
    b = select_humiliation_overlays(context, count=3)
    assert a == b


def test_extract_user_requested_phrases_from_quotes():
    phrases = extract_user_requested_phrases('生成一张图，写着「我是废物」和「跪好」')
    assert "我是废物" in phrases or "跪好" in phrases


def test_rewrite_scene_does_not_keep_raw_chat_as_only_content():
    raw = "我在看av主人"
    rewritten = rewrite_scene_without_chat_echo(raw)
    assert "photorealistic" in rewritten.lower() or "dominant" in rewritten.lower()
    assert rewritten != raw
    # Full user chat must not reappear (models paint it as captions).
    assert "我在看av主人" not in rewritten


def test_rewrite_scene_keeps_visual_keywords():
    rewritten = rewrite_scene_without_chat_echo("生成一张黑丝高跟鞋跪着的图")
    assert "黑丝" in rewritten or "高跟鞋" in rewritten


def test_overlay_instruction_requires_listed_phrases_and_forbids_chat_echo():
    block = build_overlay_instruction_block(["废物", "跪好", "不许射"])
    assert "废物" in block
    assert "跪好" in block
    assert "不许射" in block
    assert "Do NOT render the user's original chat message" in block
