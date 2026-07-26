from core.luna_visual import (
    ANATOMY_QUALITY_BLOCK,
    MIDDLE_FINGER_GESTURE,
    build_scene_image_prompt,
)


def test_build_scene_prompt_includes_middle_finger_and_anatomy():
    prompt = build_scene_image_prompt(
        scene_prompt="sitting on a couch, dominant look",
        visual_anchor="East Asian woman in black latex",
        overlay_block='On-image Chinese text: "废物"',
    )
    assert "middle finger" in prompt.lower()
    assert "two legs" in prompt.lower()
    assert "no third leg" in prompt.lower() or "exactly TWO legs" in prompt
    assert MIDDLE_FINGER_GESTURE.split(":")[0] in prompt or "flips the middle finger" in prompt
    assert "Anatomy quality" in prompt or ANATOMY_QUALITY_BLOCK[:20] in prompt
    assert "废物" in prompt


def test_build_scene_prompt_no_text_still_has_quality_blocks():
    prompt = build_scene_image_prompt(
        scene_prompt="portrait",
        visual_anchor="East Asian woman",
        no_text=True,
    )
    assert "No on-image text" in prompt
    assert "middle finger" in prompt.lower()
    assert "two legs" in prompt.lower()


def test_middle_finger_can_be_disabled():
    prompt = build_scene_image_prompt(
        scene_prompt="portrait",
        visual_anchor="East Asian woman",
        include_middle_finger=False,
        no_text=True,
    )
    assert "middle finger" not in prompt.lower()
    assert "two legs" in prompt.lower()
