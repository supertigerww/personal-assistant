"""Humiliating Chinese on-image text overlays for scene generation.

Image models tend to paint any conversational Chinese in the prompt as literal
on-image text. These helpers pick short, punchy slogans and instruct the model
to render those instead of echoing the user's chat message.
"""

from __future__ import annotations

import hashlib
import random
import re
from typing import Iterable

# Short phrases render more reliably as on-image glyphs than long sentences.
GENERAL_OVERLAYS: tuple[str, ...] = (
    "废物",
    "贱狗",
    "跪好",
    "低头",
    "听女王的",
    "你算什么",
    "只配跪下",
    "好好学着",
    "别眨眼",
    "瞪大眼睛",
    "欠教训",
    "没用的东西",
    "给我老实点",
    "继续跪着",
    "记住身份",
    "别敢射",
    "憋着",
    "不许乱动",
    "看着我",
    "服从",
)

THEME_OVERLAYS: dict[str, tuple[str, ...]] = {
    "denial": (
        "不许射",
        "寸止",
        "憋住",
        "还没资格",
        "手拿开",
        "再忍着",
        "射了就完蛋",
    ),
    "joi": (
        "跟着撸",
        "手别停",
        "盯着屏幕",
        "按节奏",
        "我说停才停",
    ),
    "cuck": (
        "乖乖看着",
        "你只配看",
        "别碰",
        "在旁边跪好",
        "老实旁观",
    ),
    "sissy": (
        "骚货",
        "女装奴",
        "扭起来",
        "叫得再浪点",
        "好好展示",
    ),
    "foot": (
        "脚奴",
        "低头舔",
        "鞋底",
        "跪到脚边",
        "闻清楚",
    ),
    "training": (
        "训练中",
        "学着点",
        "做错就罚",
        "姿势标准点",
        "再来一遍",
    ),
    "public": (
        "当众丢人",
        "被看光",
        "别遮",
        "承认吧",
        "所有人都看见了",
    ),
    "findom": (
        "交出来",
        "钱包打开",
        "配不上",
        "花钱的货",
    ),
}

THEME_TRIGGERS: dict[str, tuple[str, ...]] = {
    "denial": ("寸止", "边缘", "不许射", "憋", "锁精", "deny", "edging"),
    "joi": ("撸", "自慰", "手", "节奏", "joi", "stroke"),
    "cuck": ("绿帽", "旁观", "看别人", "cuck", "ntr", "戴绿"),
    "sissy": ("女装", "伪娘", "sissy", "裙子", "丝袜", "骚"),
    "foot": ("脚", "鞋", "丝足", "舔脚", "踩", "foot"),
    "training": ("训练", "调教", "任务", "惩罚", "跪下", "sm"),
    "public": ("公开", "当众", "露出", "被看", "社死"),
    "findom": ("交钱", "供养", "钱包", "findom", "贡"),
}

# Phrases that look like the user is asking to put specific text on the image.
_USER_QUOTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[「『\"“]([^」』\"”]{2,24})[」』\"”]"),
    re.compile(r"(?:写着?|显示|屏幕上?写|字幕|文字)[：:]\s*([^\s，。！？]{2,16})"),
    re.compile(r"(?:写着?|显示)\s*([^\s，。！？]{2,16})"),
)


def extract_user_requested_phrases(text: str) -> list[str]:
    """Pull short phrases the user explicitly wanted on the image."""
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _USER_QUOTE_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(1).strip(" 。！？,.!?")
            if len(phrase) < 2 or len(phrase) > 16:
                continue
            key = phrase.casefold()
            if key in seen:
                continue
            # Skip pure request shells without content
            if phrase in {"一张图", "图片", "生成", "形象"}:
                continue
            seen.add(key)
            found.append(phrase)
    return found[:2]


def detect_themes(text: str) -> list[str]:
    normalized = (text or "").casefold()
    if not normalized:
        return []
    themes: list[str] = []
    for theme, triggers in THEME_TRIGGERS.items():
        if any(trigger.casefold() in normalized for trigger in triggers):
            themes.append(theme)
    return themes


def select_humiliation_overlays(
    context: str,
    *,
    count: int = 3,
    rng: random.Random | None = None,
) -> list[str]:
    """Pick short humiliating Chinese slogans for on-image text.

    Mixes theme-matched lines with general lines. Stable-ish per context so the
    same user line does not reshuffle on every retry unless count/pool changes.
    """
    safe_count = max(2, min(int(count), 5))
    user_phrases = extract_user_requested_phrases(context)
    themes = detect_themes(context)

    pool: list[str] = []
    for theme in themes:
        pool.extend(THEME_OVERLAYS.get(theme, ()))
    pool.extend(GENERAL_OVERLAYS)

    # Deterministic shuffle seed from context so overlays feel intentional.
    seed_source = (context or "default").strip().casefold()
    seed = int(hashlib.sha1(seed_source.encode("utf-8")).hexdigest()[:8], 16)
    picker = rng or random.Random(seed)

    # Dedupe while preserving order, then shuffle a working copy.
    unique_pool = list(dict.fromkeys(p.strip() for p in pool if p and p.strip()))
    picker.shuffle(unique_pool)

    selected: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        cleaned = phrase.strip()
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        selected.append(cleaned)

    # Optional: keep at most one user-requested short phrase, then fill with insults.
    for phrase in user_phrases[:1]:
        _add(phrase)

    for phrase in unique_pool:
        if len(selected) >= safe_count:
            break
        _add(phrase)

    # Absolute fallback
    for phrase in GENERAL_OVERLAYS:
        if len(selected) >= safe_count:
            break
        _add(phrase)

    return selected[:safe_count]


_VISUAL_KEYWORD_HINTS: tuple[str, ...] = (
    "黑丝",
    "丝袜",
    "高跟鞋",
    "红底",
    "乳胶",
    "皮裙",
    "手套",
    "鞭",
    "项圈",
    "笼子",
    "跪",
    "脚",
    "鞋",
    "屏幕",
    "电脑",
    "镜子",
    "浴室",
    "卧室",
    "办公室",
    "女装",
    "裙子",
    "特写",
    "全身",
    "俯视",
    "霓虹",
    "红光",
    "紫光",
)


def _extract_visual_keywords(text: str) -> list[str]:
    found: list[str] = []
    for keyword in _VISUAL_KEYWORD_HINTS:
        if keyword in text and keyword not in found:
            found.append(keyword)
    # Short English visual tokens (heels, latex, close-up, etc.)
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text):
        lower = token.lower()
        if lower in {"the", "and", "for", "with", "this", "that", "from", "generate", "image", "scene"}:
            continue
        if lower not in {item.lower() for item in found}:
            found.append(token)
    return found[:6]


def rewrite_scene_without_chat_echo(scene_prompt: str) -> str:
    """Turn chatty user lines into a visual scene brief so models stop painting the chat text."""
    cleaned = (scene_prompt or "").strip()
    if not cleaned:
        return (
            "dominant East Asian woman in a power-play scene, cold superior gaze, "
            "dramatic lighting, photorealistic, full or upper body composition"
        )

    # Already looks like an English visual prompt — keep mostly as-is.
    ascii_ratio = sum(1 for ch in cleaned if ord(ch) < 128) / max(len(cleaned), 1)
    if ascii_ratio >= 0.72 and len(cleaned) >= 24:
        return cleaned

    themes = detect_themes(cleaned)
    theme_hint = "humiliating power-play atmosphere, "
    if "denial" in themes or "joi" in themes:
        theme_hint = "edge-control / hands-off domination atmosphere, "
    elif "cuck" in themes:
        theme_hint = "spectator humiliation atmosphere with a glowing monitor, "
    elif "sissy" in themes:
        theme_hint = "feminization training atmosphere, "
    elif "foot" in themes:
        theme_hint = "foot-worship power dynamic, "
    elif "training" in themes:
        theme_hint = "strict training / discipline atmosphere, "
    elif "public" in themes:
        theme_hint = "public exposure fantasy atmosphere, "
    elif "findom" in themes:
        theme_hint = "financial domination atmosphere, "

    visual_bits = _extract_visual_keywords(cleaned)
    props = f"props/details: {', '.join(visual_bits)}, " if visual_bits else ""

    # Never re-inject the full Chinese chat line — models paint it as captions.
    return (
        f"photorealistic dominant woman, {theme_hint}{props}"
        "cold superior expression, sharp composition, dramatic lighting, "
        "no speech bubbles, no user chat transcript, no long Chinese sentences"
    )


def build_overlay_instruction_block(overlays: Iterable[str]) -> str:
    phrases = [p.strip() for p in overlays if p and p.strip()]
    if not phrases:
        return ""
    listed = "\n".join(f'- "{phrase}"' for phrase in phrases)
    return (
        "\nOn-image Chinese text (REQUIRED):\n"
        "Render LARGE, CLEAR, bold Simplified Chinese characters as visible overlays "
        "(monitor caption, floating labels, neon sign, or subtitle bar).\n"
        "Must include ALL of these exact phrases as separate text elements:\n"
        f"{listed}\n"
        "Put 2-4 humiliating Chinese phrases in the frame. "
        "Do NOT render the user's original chat message, full sentences they typed, "
        "or any long conversational Chinese. Only the short slogans listed above.\n"
        "Text must be sharp and legible, high contrast, not garbled."
    )
