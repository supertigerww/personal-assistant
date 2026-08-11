from core.luna_visual import (
    ANATOMY_QUALITY_BLOCK,
    build_scene_image_prompt,
)


def test_build_scene_prompt_includes_anatomy_not_middle_finger_by_default():
    prompt = build_scene_image_prompt(
        scene_prompt="sitting on a couch, dominant look",
        visual_anchor="East Asian woman in black latex",
        overlay_block='On-image Chinese text: "废物"',
    )
    assert "middle finger" not in prompt.lower()
    assert "two legs" in prompt.lower() or "TWO legs" in prompt
    assert "fused" in prompt.lower() or "separate" in prompt.lower()
    assert "Anatomy quality" in prompt or ANATOMY_QUALITY_BLOCK[:20] in prompt
    assert "Preferred composition" in prompt or "simple" in prompt.lower()
    assert "废物" in prompt


def test_build_scene_prompt_no_text_still_has_quality_blocks():
    prompt = build_scene_image_prompt(
        scene_prompt="portrait",
        visual_anchor="East Asian woman",
        no_text=True,
    )
    assert "No on-image text" in prompt
    assert "middle finger" not in prompt.lower()
    assert "two legs" in prompt.lower()


def test_middle_finger_can_be_enabled():
    prompt = build_scene_image_prompt(
        scene_prompt="portrait",
        visual_anchor="East Asian woman",
        include_middle_finger=True,
        no_text=True,
    )
    assert "middle finger" in prompt.lower()
    assert "two legs" in prompt.lower()
