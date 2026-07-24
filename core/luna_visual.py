from __future__ import annotations

from pathlib import Path
from typing import Any


def load_visual_anchor(settings: Any) -> str:
    prompt_path = Path(getattr(settings, "luna_visual_prompt_path", "prompts/luna_visual.txt"))
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "Highly detailed photorealistic East Asian woman, late 20s to early 30s, "
        "oval soft-elegant face, large almond eyes, confident cold gaze, full rosy lips, "
        "long voluminous wavy black hair, elegant seductive superior aura, "
        "shiny black latex corset, black leather short skirt, sheer black pantyhose, "
        "long black gloves, pointed black stiletto heels with glossy red soles."
    )


def build_scene_image_prompt(
    *,
    scene_prompt: str,
    visual_anchor: str,
    overlay_block: str = "",
    no_text: bool = False,
) -> str:
    cleaned_scene = scene_prompt.strip()
    cleaned_anchor = visual_anchor.strip()
    cleaned_overlay = (overlay_block or "").strip()

    if not cleaned_scene:
        base = cleaned_anchor
    elif not cleaned_anchor:
        base = cleaned_scene
    else:
        anchor_key = cleaned_anchor.casefold()[:48]
        if anchor_key and anchor_key in cleaned_scene.casefold():
            base = cleaned_scene
        else:
            base = f"{cleaned_anchor}\nScene: {cleaned_scene}"

    if no_text:
        return (
            f"{base}\n"
            "No on-image text, no captions, no watermarks, no Chinese characters in the frame."
        ).strip()

    if cleaned_overlay:
        return f"{base}\n{cleaned_overlay}".strip()
    return base
