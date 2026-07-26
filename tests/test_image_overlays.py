from core.image_overlays import (
    OVERLAY_MAX_CHARS,
    build_overlay_instruction_block,
    extract_user_requested_phrases,
    normalize_overlay_phrases,
    parse_llm_overlay_phrases,
    rewrite_scene_without_chat_echo,
    select_humiliation_overlays,
)


def test_select_overlays_returns_few_phrases():
    phrases = select_humiliation_overlays("我在看av主人 寸止 不许射 漏精", count=2)
    assert 1 <= len(phrases) <= 3
    assert all(2 <= len(p) <= OVERLAY_MAX_CHARS for p in phrases)
    assert any(p for p in phrases)


def test_select_overlays_stable_for_same_context():
    context = "跪着看女王训练 废物 无脑狗"
    a = select_humiliation_overlays(context, count=2)
    b = select_humiliation_overlays(context, count=2)
    assert a == b


def test_select_overlays_foot_context_prefers_foot_lines():
    phrases = select_humiliation_overlays("盯着高跟鞋 恋足 鞋底", count=2)
    blob = "".join(phrases)
    assert any(k in blob for k in ("脚", "鞋", "恋足", "丝袜", "踩"))


def test_select_overlays_allows_longer_than_ten():
    phrases = select_humiliation_overlays("锁精 盯脚 寸止", count=2)
    # At least pool contains long lines; selected should accept them if ranked high
    assert all(len(p) <= OVERLAY_MAX_CHARS for p in phrases)
    long_pool = select_humiliation_overlays("恋足 高跟鞋 鞋底 漏精", count=2)
    assert any(len(p) >= 6 for p in long_pool)


def test_extract_user_requested_phrases_from_quotes():
    phrases = extract_user_requested_phrases('生成一张图，写着「我是废物」和「跪好」')
    assert "我是废物" in phrases or "跪好" in phrases


def test_rewrite_scene_does_not_keep_raw_chat_as_only_content():
    raw = "我在看av主人"
    rewritten = rewrite_scene_without_chat_echo(raw)
    assert "photorealistic" in rewritten.lower() or "dominant" in rewritten.lower()
    assert rewritten != raw
    assert "我在看av主人" not in rewritten


def test_rewrite_scene_keeps_visual_keywords():
    rewritten = rewrite_scene_without_chat_echo("生成一张黑丝高跟鞋跪着的图")
    assert "黑丝" in rewritten or "高跟鞋" in rewritten


def test_overlay_instruction_sparse_and_lists_phrases():
    block = build_overlay_instruction_block(["无脑狗", "只许漏不准射！"])
    assert "无脑狗" in block
    assert "只许漏不准射" in block
    assert "SPARSE" in block or "sparse" in block.lower() or "ONLY these" in block


def test_parse_llm_overlay_phrases_from_json():
    phrases = parse_llm_overlay_phrases('["无脑狗只准看鞋尖！", "看着鞋底漏精！"]', count=2)
    assert len(phrases) == 2
    assert all("慢点" not in p for p in phrases)


def test_parse_llm_overlay_phrases_drops_soft_and_too_long():
    raw = (
        '1. 慢点撸\n'
        "2. 只许漏不准射！\n"
        "3. 这是一句超级超级超级长会被丢掉的对话式羞辱句子真的太长了啊啊\n"
        "4. 贱狗"
    )
    phrases = parse_llm_overlay_phrases(raw, count=2)
    assert "慢点撸" not in phrases
    assert all(len(p) <= OVERLAY_MAX_CHARS for p in phrases)
    assert not any("超级超级" in p for p in phrases)


def test_normalize_overlay_phrases_dedupes():
    assert normalize_overlay_phrases(["废物", "废物", "跪好", "跪好"], count=2) == ["废物", "跪好"]
