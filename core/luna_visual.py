from __future__ import annotations

from pathlib import Path
from typing import Any

# Optional only — default off (finger gestures often render poorly / moderated).
MIDDLE_FINGER_GESTURE = (
    "Pose gesture (optional): she raises one hand toward the camera/viewer "
    "and flips the middle finger (obscene insult gesture), dominant mocking expression, "
    "middle finger fully extended and easy to see; the other hand may rest on hip, thigh, or leg."
)

ANATOMY_QUALITY_BLOCK = (
    "Anatomy quality (REQUIRED, strict): photorealistic correct human anatomy; "
    "exactly ONE head, TWO arms, TWO hands, TWO legs, TWO feet; "
    "each hand has exactly five fingers, no extra fingers, no missing fingers, no fused fingers; "
    "natural relaxed hands (e.g. one hand on hip, other arm relaxed) — do NOT force middle-finger or odd finger poses; "
    "no extra limbs, no third leg, no duplicated legs, no merged legs, no broken joints; "
    "natural limb placement, coherent seated or standing pose, no warped torso; "
    "clean silhouette, high detail, sharp focus."
)


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
    include_middle_finger: bool = False,
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

    parts = [base]
    if include_middle_finger:
        parts.append(MIDDLE_FINGER_GESTURE)
    parts.append(ANATOMY_QUALITY_BLOCK)

    if no_text:
        parts.append(
            "No on-image text, no captions, no watermarks, no Chinese characters in the frame."
        )
    elif cleaned_overlay:
        parts.append(cleaned_overlay)

    return "\n".join(part for part in parts if part).strip()
