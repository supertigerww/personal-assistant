"""Humiliating Chinese on-image text overlays for scene generation.

Image models tend to paint any conversational Chinese in the prompt as literal
on-image text. These helpers pick short, punchy slogans and instruct the model
to render those instead of echoing the user's chat message.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Iterable

# User-curated style: base slurs + command snipes. Longer lines OK (up to ~16 chars).
# Soft coaching (慢点撸/手别停) is banned elsewhere.

BASE_TITLES: tuple[str, ...] = (
    "贱狗",
    "死贱狗",
    "无脑狗",
    "废物",
    "贱货",
    "傻逼",
    "公狗",
    "骚狗",
    "丧志狗",
    "精虫",
    "废屌",
    "臭鸡巴",
    "肉便器",
    "精厕",
    "精壶",
    "尿壶",
    "公厕",
    "鸡巴套",
)

# Short harsh commands mixed into general pool.
GENERAL_OVERLAYS: tuple[str, ...] = (
    *BASE_TITLES,
    "贱狗！无脑！",
    "射出来！",
    "流精！",
    "漏精！",
    "寸止！",
    "憋住！",
    "再撸！",
    "快速！",
    "龟责！",
    "废物小肉虫给我流水！",
    "无脑狗给我漏！",
    "只许漏不准射！",
    "再敢射满就踩烂！",
    "精液全部流出来！",
    "自己把精液挤干净！",
    "变态恋足废物！",
)

THEME_OVERLAYS: dict[str, tuple[str, ...]] = {
    "foot": (
        "注意力集中！只允许盯着这里看！",
        "看这里！",
        "眼睛锁死在鞋上！",
        "死盯着这双高跟鞋！",
        "视线不许离开我的脚！",
        "无脑狗只准看鞋尖！",
        "再敢抬眼就踩你脸！",
        "盯着脚自己撸！",
        "看着鞋底漏精！",
        "这双鞋才是你的天！",
        "无脑恋足的死贱狗给我漏精！",
        "闻着脚臭自己流水！",
        "丝袜脚底才是你配看的东西！",
        "把脸贴过来闻鞋！",
        "舌头伸出来等着被踩！",
        "恋足废物自己承认！",
        "高跟鞋才是你唯一的主人！",
        "脚臭精虫自己漏精！",
        "丝袜擦你流出来的精！",
        "鞋底把你精液踩烂！",
        "无脑狗射完就给我舔干净再踩！",
        "恋足贱狗的精液只配被踩进地板！",
        "无脑恋足贱狗",
        "变态恋足废物",
        "恋足死狗",
        "恋足精虫",
        "恋足肉便器",
        "恋足公狗",
        "脚垫狗",
        "鞋垫狗",
        "臭脚奴",
        "丝袜狗",
        "恋足精厕",
        "无脑脚奴",
        "高跟鞋狗",
        "恋足漏精狗",
        "盯脚贱狗",
    ),
    "denial": (
        "寸止！",
        "憋住！",
        "只许漏不准射！",
        "再敢射满就踩烂！",
        "漏精！",
        "流精！",
        "射出来！",
        "精液全部流出来！",
        "自己把精液挤干净！",
        "废物小肉虫给我流水！",
        "无脑狗给我漏！",
        "龟责！",
    ),
    "joi": (
        "再撸！",
        "快速！",
        "盯着脚自己撸！",
        "看着鞋底漏精！",
        "无脑狗给我漏！",
        "射出来！",
        "流精！",
        "漏精！",
        "龟责！",
        "废物小肉虫给我流水！",
    ),
    "chastity": (
        "锁着的恋足狗给我漏精！",
        "锁精贱狗盯着脚寸止！",
        "锁死鸡巴只许流水！",
        "锁着跪好盯着鞋！",
        "钥匙在我这，你只配漏！",
        "锁精废物看着脚自己挤精！",
        "锁着也不许射满！",
        "锁奴的精液直接踩烂！",
        "锁奴",
        "锁精狗",
        "锁鸡巴奴",
        "锁精贱狗",
        "锁着的废物",
        "锁精肉便器",
        "锁死公狗",
        "禁射锁奴",
        "锁精恋足狗",
        "锁着漏精的狗",
    ),
    "underslave": (
        "奴下奴只配被踩精！",
        "最贱的奴下奴给我漏！",
        "奴下奴盯着脚自己流水！",
        "比狗还贱的奴下奴自己承认！",
        "奴下奴的精液直接踩进鞋底！",
        "跪在最下面自己漏精！",
        "奴下奴只配闻脚臭撸！",
        "奴下奴",
        "最贱的奴",
        "奴下狗",
        "最低等的畜",
        "奴下肉便器",
        "奴下精厕",
        "比狗还贱的奴",
        "奴下恋足狗",
        "奴下锁精狗",
        "最底层的死狗",
    ),
    "sissy": (
        "女装恋足狗给我漏精！",
        "女装锁精狗盯着脚寸止！",
        "女装贱狗精液被踩烂！",
        "女装无脑狗只准看鞋！",
        "女装狗",
        "伪娘狗",
        "女装贱狗",
        "女装公狗",
        "女装恋足狗",
        "女装锁精狗",
        "裙子精虫",
        "女装肉便器",
        "伪娘精厕",
        "女装无脑狗",
    ),
    "cuck": (
        "绿奴盯着鞋自己流水！",
        "绿帽戴着跪着看脚！",
        "绿奴恋足废物自己漏！",
        "绿奴",
        "绿帽狗",
        "绿奴贱狗",
        "绿帽公狗",
        "绿奴恋足狗",
        "绿奴锁精狗",
        "绿帽精厕",
        "绿奴肉便器",
        "绿奴无脑狗",
    ),
    "training": (
        "贱狗！无脑！",
        "无脑狗给我漏！",
        "再敢抬眼就踩你脸！",
        "再敢射满就踩烂！",
        "注意力集中！只允许盯着这里看！",
    ),
    "service": (
        "舌头伸出来等着被踩！",
        "把脸贴过来闻鞋！",
        "无脑狗射完就给我舔干净再踩！",
        "肉便器",
        "精厕",
        "精壶",
        "尿壶",
        "公厕",
    ),
    "public": (
        "恋足贱狗的精液只配被踩进地板！",
        "鞋底把你精液踩烂！",
        "再敢抬眼就踩你脸！",
    ),
    "findom": (
        "钥匙在我这，你只配漏！",
        "废屌",
        "废物",
        "鸡巴套",
    ),
}

THEME_TRIGGERS: dict[str, tuple[str, ...]] = {
    "denial": (
        "寸止", "边缘", "不许射", "不准射", "憋", "停手", "别射", "漏精", "流精",
        "deny", "edging", "只许漏",
    ),
    "joi": ("撸", "自慰", "手", "joi", "stroke", "硬", "舒服", "快感", "再撸", "快速", "龟责"),
    "cuck": (
        "绿帽", "绿奴", "旁观", "看别人", "女奴", "交合", "别人操", "cuck", "ntr", "戴绿", "男友",
    ),
    "sissy": ("女装", "伪娘", "sissy", "裙子", "丝袜", "娘"),
    "foot": (
        "脚", "鞋", "丝足", "舔脚", "踩", "脚底", "脚趾", "高跟", "丝袜脚", "脚臭", "鞋底",
        "恋足", "盯脚", "鞋尖", "foot",
    ),
    "chastity": ("锁精", "锁奴", "贞操", "上锁", "钥匙", "锁鸡巴", "禁射", "cb", "chastity"),
    "underslave": ("奴下奴", "奴下", "最贱的奴", "最低等", "比狗还贱", "最底层"),
    "training": ("训练", "调教", "任务", "惩罚", "跪下", "sm", "无脑"),
    "public": ("公开", "当众", "露出", "被看", "社死", "地板"),
    "findom": ("交钱", "供养", "钱包", "findom", "贡", "钥匙"),
    "service": ("舔", "口交", "侍奉", "舌头", "含", "口", "深喉", "闻鞋"),
}

# Soft coaching lines that must never appear as on-image text.
SOFT_OVERLAY_BANLIST: frozenset[str] = frozenset(
    {
        "慢点撸",
        "手别停",
        "按我节奏",
        "跟着撸",
        "好好学着",
        "看着我",
        "服从",
        "别眨眼",
        "继续跪着",
        "给我老实点",
        "学着点",
        "再来一遍",
        "按节奏",
        "老实撸",
    }
)

# Local Pillow compose can handle longer lines than image-model painted text.
OVERLAY_MAX_CHARS = 22
OVERLAY_MIN_CHARS = 2

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
            if len(phrase) < OVERLAY_MIN_CHARS or len(phrase) > OVERLAY_MAX_CHARS:
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


def _is_soft_overlay(phrase: str) -> bool:
    cleaned = (phrase or "").strip()
    if not cleaned:
        return True
    if cleaned in SOFT_OVERLAY_BANLIST:
        return True
    # Soft coaching patterns
    soft_bits = ("慢点", "轻轻", "好孩子", "乖", "加油", "继续努力")
    return any(bit in cleaned for bit in soft_bits)


def _context_relevance_score(phrase: str, context: str, themes: list[str]) -> int:
    """Higher = more tied to current chat; used to rank pool fallbacks."""
    if not phrase:
        return -100
    if _is_soft_overlay(phrase):
        return -100
    score = 0
    ctx = context or ""
    # Prefer theme-pool items when themes were detected
    for theme in themes:
        if phrase in THEME_OVERLAYS.get(theme, ()):
            score += 5
    # Token overlap with context characters / keywords
    for token in (
        "射", "精", "撸", "绿", "脚", "鞋", "舔", "跪", "操", "鸡巴", "女奴", "黑丝",
        "硬", "憋", "寸止", "锁", "恋足", "漏", "流精", "女装", "伪娘", "奴下", "踩",
        "高跟", "丝袜", "无脑", "贱狗",
    ):
        if token in phrase and token in ctx:
            score += 3
        elif token in phrase:
            score += 1
    harsh = ("鸡巴", "射", "操", "精", "绿帽", "贱货", "废物", "漏精", "恋足", "锁精", "踩")
    if any(h in ctx for h in harsh) and any(h in phrase for h in harsh):
        score += 2
    return score


def select_humiliation_overlays(
    context: str,
    *,
    count: int = 4,
    rng: random.Random | None = None,
) -> list[str]:
    """Pick humiliating Chinese slogans for on-image text.

    Prefers theme/context-matched harsh lines. Soft coaching slogans are banned.
    Uses fresh randomness each call so the same chat does not always stamp identical captions.
    """
    safe_count = max(2, min(int(count), 5))
    user_phrases = extract_user_requested_phrases(context)
    themes = detect_themes(context)

    pool: list[str] = []
    for theme in themes:
        pool.extend(THEME_OVERLAYS.get(theme, ()))
    # Only add general after themes so relevance ranking can prefer themed lines
    pool.extend(GENERAL_OVERLAYS)

    # Fresh entropy every call (avoid same two slogans every image).
    picker = rng or random.Random()

    unique_pool = [
        p.strip()
        for p in dict.fromkeys(item.strip() for item in pool if item and item.strip())
        if not _is_soft_overlay(p.strip())
    ]
    # Rank by context relevance, then light shuffle within same score band
    unique_pool.sort(
        key=lambda phrase: (
            -_context_relevance_score(phrase, context, themes),
            picker.random(),
        )
    )

    selected: list[str] = []
    seen: set[str] = set()

    def _add(phrase: str) -> None:
        cleaned = re.sub(r"\s+", "", (phrase or "").strip())
        cleaned = cleaned.strip("「」『』\"'“”‘’")
        if not cleaned or _is_soft_overlay(cleaned):
            return
        if len(cleaned) < OVERLAY_MIN_CHARS or len(cleaned) > OVERLAY_MAX_CHARS:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        selected.append(cleaned)

    for phrase in user_phrases[:1]:
        _add(phrase)

    for phrase in unique_pool:
        if len(selected) >= safe_count:
            break
        _add(phrase)

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
            "dramatic lighting, photorealistic, standing three-quarter body, "
            "simple pose, clear two legs, continuous torso"
        )

    # Already looks like an English visual prompt — keep, but strip known hard poses.
    ascii_ratio = sum(1 for ch in cleaned if ord(ch) < 128) / max(len(cleaned), 1)
    if ascii_ratio >= 0.72 and len(cleaned) >= 24:
        return (
            f"{cleaned}, single complete subject, simple stable pose, "
            "clear separate legs, no deformed anatomy"
        )

    themes = detect_themes(cleaned)
    theme_hint = "humiliating power-play atmosphere, "
    pose_hint = (
        "standing three-quarter view or seated upright with legs together, "
        "show full continuous torso, "
    )
    if "denial" in themes or "joi" in themes:
        theme_hint = "edge-control / hands-off domination atmosphere, "
    elif "cuck" in themes:
        theme_hint = "spectator humiliation atmosphere with a glowing monitor, "
    elif "sissy" in themes:
        theme_hint = "feminization training atmosphere, "
    elif "foot" in themes:
        theme_hint = "foot-worship power dynamic, glossy heels visible, "
        # Pointing a shoe at camera is fine; extreme crossed multi-limb poses often break.
        pose_hint = (
            "seated upright on a chair or bench, torso fully visible, "
            "legs side-by-side or gently extended toward camera, two clear separate legs, "
        )
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
        f"photorealistic dominant woman, {theme_hint}{props}{pose_hint}"
        "cold superior expression, sharp composition, dramatic lighting, "
        "single subject only, correct anatomy, exactly two legs and two arms, "
        "no fused legs, no extra limbs, no missing torso, "
        "no speech bubbles, no user chat transcript, no long Chinese sentences"
    )


def normalize_overlay_phrases(
    raw_phrases: Iterable[str],
    *,
    count: int = 4,
) -> list[str]:
    """Clamp overlay phrases to Chinese slogans suitable for local Pillow stamp."""
    safe_count = max(2, min(int(count), 5))
    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_phrases:
        cleaned = re.sub(r"\s+", "", (raw or "").strip())
        # Keep ！ for meme punch; strip soft punctuation only.
        cleaned = cleaned.strip("「」『』\"'“”‘’。？?，,、；;：:.-—…·")
        # Drop list markers like "1." "2、" "-"
        cleaned = re.sub(r"^[\d]+[\.\)、．]\s*", "", cleaned)
        cleaned = re.sub(r"^[-*•]\s*", "", cleaned)
        if len(cleaned) < OVERLAY_MIN_CHARS or len(cleaned) > OVERLAY_MAX_CHARS:
            continue
        if _is_soft_overlay(cleaned):
            continue
        # Allow ！ but not multi-clause sentences with ，。？
        if any(sep in cleaned for sep in ("。", "？", "?", "，", ",", "；", ";")):
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(cleaned)
        if len(selected) >= safe_count:
            break
    return selected


def parse_llm_overlay_phrases(text: str, *, count: int = 4) -> list[str]:
    """Parse LLM output into short on-image slogans.

    Accepts JSON arrays, one-phrase-per-line, or comma-separated phrases.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    candidates: list[str] = []

    # Prefer a JSON array if present anywhere in the response.
    json_match = re.search(r"\[[\s\S]*?\]", raw)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                candidates.extend(str(item) for item in parsed if item is not None)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not candidates:
        # Line-first, then Chinese/English comma splits.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if "、" in line or "，" in line or "," in line:
                parts = re.split(r"[、，,]+", line)
                candidates.extend(parts)
            else:
                candidates.append(line)

    return normalize_overlay_phrases(candidates, count=count)


def build_overlay_instruction_block(overlays: Iterable[str]) -> str:
    phrases = [p.strip() for p in overlays if p and p.strip()][:5]
    if not phrases:
        return ""
    listed = "\n".join(f'- "{phrase}"' for phrase in phrases)
    n = len(phrases)
    return (
        "\nOn-image Chinese text (REQUIRED, keep SPARSE):\n"
        f"Render ONLY these {n} crude Simplified Chinese slogans "
        f"(exactly {n} text elements, no more):\n"
        f"{listed}\n"
        "Layout: leave most of the image empty for the woman; place text in 1–2 corners "
        "or one bottom bar only. Do NOT fill the frame with many labels. "
        "Do NOT add extra Chinese phrases beyond the list. "
        "Style: bold, high-contrast, vulgar meme captions — large but few "
        "(foot/chastity/cuck humiliation tone like 无脑狗/恋足贱狗/锁精狗). "
        "Do NOT render the user's chat message. "
        "Do NOT invent soft coaching lines (no 慢点撸/手别停). "
        "Text must be sharp and legible, not garbled."
    )
