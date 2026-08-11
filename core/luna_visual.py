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
    "Anatomy quality (REQUIRED, strict): photorealistic correct human anatomy of ONE complete woman; "
    "exactly ONE head, ONE neck, ONE torso, TWO arms, TWO hands, TWO legs, TWO feet, TWO shoes; "
    "each hand has exactly five fingers; natural hands (on hip / resting), no middle-finger pose; "
    "legs clearly SEPARATE and correctly jointed at hips/knees/ankles — never fused, melted, shared, "
    "or branching into a third leg; feet point naturally, not mirrored wrong; "
    "no missing torso, no floating limbs, no extra limbs, no broken joints, no warped waist; "
    "body must look like a real continuous person, not a collage."
)

# Prefer poses that image models render reliably (complex crossed-leg angles often break).
SAFE_POSE_BLOCK = (
    "Preferred composition (REQUIRED for stability): single subject only; "
    "simple clear pose — standing three-quarter view OR seated upright with legs side-by-side "
    "or lightly crossed at ankles only; camera at normal eye level; "
    "show continuous torso from shoulders through hips; full or upper-to-mid body; "
    "AVOID extreme contortions, twisted multi-angle bodies, upside-down poses, "
    "heavy overlapping legs that hide joint structure, multi-person composites, "
    "cut-off mid-limb crops that invent extra legs."
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
    parts.append(SAFE_POSE_BLOCK)
    parts.append(ANATOMY_QUALITY_BLOCK)
    parts.append(
        "Negative (must avoid): deformed anatomy, fused legs, extra legs, missing body parts, "
        "bad hands, bad feet, mutated limbs, disembodied limbs, mannequin joints, "
        "blurry face, low quality, text, watermark."
    )

    if no_text:
        parts.append(
            "No on-image text, no captions, no watermarks, no Chinese characters in the frame."
        )
    elif cleaned_overlay:
        parts.append(cleaned_overlay)

    return "\n".join(part for part in parts if part).strip()
