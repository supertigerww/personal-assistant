from __future__ import annotations

from pathlib import Path
from typing import Any


def load_visual_anchor(settings: Any) -> str:
    prompt_path = Path(getattr(settings, "luna_visual_prompt_path", "prompts/luna_visual.txt"))
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "Highly detailed realistic East Asian woman, mid-30s, sharp elegant facial features, "
        "long straight black hair, fair skin, cold arrogant expression, dominant aura, "
        "black latex corset, black leather skirt, sheer black pantyhose, stiletto heels."
    )


def build_scene_image_prompt(*, scene_prompt: str, visual_anchor: str) -> str:
    cleaned_scene = scene_prompt.strip()
    cleaned_anchor = visual_anchor.strip()
    if not cleaned_scene:
        return cleaned_anchor
    if not cleaned_anchor:
        return cleaned_scene

    anchor_key = cleaned_anchor.casefold()[:48]
    if anchor_key and anchor_key in cleaned_scene.casefold():
        return cleaned_scene

    return f"{cleaned_anchor}\nScene: {cleaned_scene}"