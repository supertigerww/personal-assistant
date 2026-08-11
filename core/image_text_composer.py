"""Local Chinese text overlays on generated images (Pillow).

Generate a clean base image via xAI (no on-image glyphs), then stamp crude
Simplified Chinese slogans here so moderation is less likely to reject the API call.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Iterable, Sequence
from uuid import uuid4

logger = logging.getLogger(__name__)

# Candidate font files: project fonts first, then common system CJK faces.
_FONT_CANDIDATES: tuple[str, ...] = (
    "assets/fonts/NotoSansSC-Bold.otf",
    "assets/fonts/NotoSansSC-Bold.ttf",
    "assets/fonts/NotoSansCJKsc-Bold.otf",
    "assets/fonts/SourceHanSansSC-Bold.otf",
    "assets/fonts/msyhbd.ttc",
    "assets/fonts/msyh.ttc",
    "assets/fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Bold.otf",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyhbd.ttf",
)


# Fill colors cycle so multi-line captions don't look identical.
_FILL_CYCLE: tuple[tuple[int, int, int, int], ...] = (
    (255, 245, 200, 255),  # pale gold
    (255, 90, 90, 255),  # red
    (255, 255, 255, 255),  # white
    (255, 180, 220, 255),  # pink
    (255, 220, 100, 255),  # yellow
)


def compose_humiliation_overlays(
    image_path: str | Path,
    phrases: Sequence[str],
    *,
    output_dir: str | Path | None = None,
    font_path: str | None = None,
) -> str:
    """Draw multiple crude Chinese slogans onto a local image; return output path.

    On any failure, returns the original path unchanged.
    """
    source = Path(image_path)
    cleaned = [p.strip() for p in phrases if p and str(p).strip()][:5]
    if not cleaned:
        return str(source)
    if not source.exists():
        logger.warning("compose_humiliation_overlays: source missing %s", source)
        return str(source)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        logger.warning("Pillow not installed; skipping local text compose.")
        return str(source)

    try:
        with Image.open(source) as raw:
            image = raw.convert("RGBA")
            width, height = image.size
            draw = ImageDraw.Draw(image)

            slots = _layout_slots(len(cleaned), width, height)
            for index, (phrase, (anchor, xy, size_scale)) in enumerate(zip(cleaned, slots)):
                font = _load_font(font_path, image_height=height, size_scale=size_scale)
                fill = _FILL_CYCLE[index % len(_FILL_CYCLE)]
                _draw_outlined_text(
                    draw,
                    text=phrase,
                    xy=xy,
                    font=font,
                    fill=fill,
                    outline=(0, 0, 0, 255),
                    outline_width=max(2, height // 260),
                    anchor=anchor,
                    max_width=int(width * 0.90),
                )

            out_dir = Path(output_dir) if output_dir else source.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"overlay_{uuid4().hex}.png"
            image.convert("RGB").save(out_path, format="PNG", optimize=True)
            logger.info(
                "Composed local text overlay source=%s phrases=%s output=%s",
                source.name,
                cleaned,
                out_path.name,
            )
            return str(out_path)
    except Exception as exc:
        logger.exception("Failed to compose text overlays on %s: %s", source, exc)
        return str(source)


def resolve_cjk_font_path(preferred: str | None = None) -> Path | None:
    """Return first usable CJK font path, or None."""
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    for item in _FONT_CANDIDATES:
        candidates.append(Path(item))
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def _load_font(preferred: str | None, *, image_height: int, size_scale: float = 1.0):
    from PIL import ImageFont

    base = max(26, min(64, image_height // 18))
    size = max(22, int(base * size_scale))
    font_path = resolve_cjk_font_path(preferred)
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError as exc:
            logger.warning("Failed to load font %s: %s", font_path, exc)
    logger.warning("No CJK font found; text may not render Chinese correctly.")
    return ImageFont.load_default()


def _layout_slots(
    count: int,
    width: int,
    height: int,
) -> list[tuple[str, tuple[int, int], float]]:
    """Return (anchor, xy, size_scale) for up to 5 slogans — meme-style coverage."""
    mx = int(width * 0.05)
    my = int(height * 0.04)
    # (anchor, x, y, size_scale)
    catalog = [
        ("mt", width // 2, my + int(height * 0.03), 1.15),  # top center big
        ("mt", width // 2, my + int(height * 0.12), 0.95),  # top second line
        ("la", mx, int(height * 0.38), 0.85),  # mid left
        ("ra", width - mx, int(height * 0.52), 0.85),  # mid right
        ("ms", width // 2, height - my - int(height * 0.08), 1.10),  # bottom
    ]
    if count <= 1:
        picks = [catalog[4]]
    elif count == 2:
        picks = [catalog[0], catalog[4]]
    elif count == 3:
        picks = [catalog[0], catalog[2], catalog[4]]
    elif count == 4:
        picks = [catalog[0], catalog[1], catalog[2], catalog[4]]
    else:
        picks = catalog[:5]
    return [(a, (x, y), s) for a, x, y, s in picks]


def _draw_outlined_text(
    draw,
    *,
    text: str,
    xy: tuple[int, int],
    font,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    outline_width: int,
    anchor: str,
    max_width: int,
) -> None:
    # Soft wrap longer slogans for readability.
    wrapped = text
    if len(text) > 11:
        wrapped = "\n".join(textwrap.wrap(text, width=11) or [text])

    # Pillow multiline_text does not support anchor — compute top-left ourselves.
    spacing = 4
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align="center")
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ax, ay = xy
    if anchor in {"mt", "ms", "mb"}:
        left = ax - tw // 2
    elif anchor in {"rt", "ra", "rb", "rm"}:
        left = ax - tw
    else:
        left = ax
    if anchor in {"mt", "lt", "rt"}:
        top = ay
    elif anchor in {"ms", "la", "ra", "mm"}:
        top = ay - th // 2
    else:
        top = ay - th

    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy > outline_width * outline_width:
                continue
            draw.multiline_text(
                (left + dx, top + dy),
                wrapped,
                font=font,
                fill=outline,
                align="center",
                spacing=spacing,
            )
    draw.multiline_text(
        (left, top),
        wrapped,
        font=font,
        fill=fill,
        align="center",
        spacing=spacing,
    )
