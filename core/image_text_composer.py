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


def compose_humiliation_overlays(
    image_path: str | Path,
    phrases: Sequence[str],
    *,
    output_dir: str | Path | None = None,
    font_path: str | None = None,
) -> str:
    """Draw sparse crude Chinese slogans onto a local image; return output path.

    On any failure, returns the original path unchanged.
    """
    source = Path(image_path)
    cleaned = [p.strip() for p in phrases if p and str(p).strip()][:3]
    if not cleaned:
        return str(source)
    if not source.exists():
        logger.warning("compose_humiliation_overlays: source missing %s", source)
        return str(source)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow not installed; skipping local text compose.")
        return str(source)

    try:
        with Image.open(source) as raw:
            image = raw.convert("RGBA")
            width, height = image.size
            draw = ImageDraw.Draw(image)

            font = _load_font(font_path, image_height=height)
            # Layout: top-right + bottom bar (or single bottom if one phrase)
            positions = _layout_slots(len(cleaned), width, height)
            for phrase, (anchor, xy) in zip(cleaned, positions):
                _draw_outlined_text(
                    draw,
                    text=phrase,
                    xy=xy,
                    font=font,
                    fill=(255, 245, 200, 255),
                    outline=(0, 0, 0, 255),
                    outline_width=max(2, height // 280),
                    anchor=anchor,
                    max_width=int(width * 0.88),
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


def _load_font(preferred: str | None, *, image_height: int):
    from PIL import ImageFont

    size = max(28, min(72, image_height // 16))
    font_path = resolve_cjk_font_path(preferred)
    if font_path is not None:
        try:
            return ImageFont.truetype(str(font_path), size=size)
        except OSError as exc:
            logger.warning("Failed to load font %s: %s", font_path, exc)
    logger.warning("No CJK font found; text may not render Chinese correctly.")
    return ImageFont.load_default()


def _layout_slots(count: int, width: int, height: int) -> list[tuple[str, tuple[int, int]]]:
    """Return (anchor, xy) pairs for 1–3 slogans."""
    margin_x = int(width * 0.06)
    margin_y = int(height * 0.06)
    if count <= 1:
        return [("ms", (width // 2, height - margin_y - int(height * 0.04)))]
    if count == 2:
        return [
            ("ra", (width - margin_x, margin_y + int(height * 0.02))),
            ("ms", (width // 2, height - margin_y - int(height * 0.04))),
        ]
    return [
        ("ra", (width - margin_x, margin_y + int(height * 0.02))),
        ("la", (margin_x, int(height * 0.42))),
        ("ms", (width // 2, height - margin_y - int(height * 0.04))),
    ]


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
    # Soft wrap very long lines for the image model-friendly slogans up to ~18 chars.
    wrapped = text
    if len(text) > 12:
        # Prefer wrap at punctuation
        wrapped = "\n".join(textwrap.wrap(text, width=10) or [text])

    x, y = xy
    # Manual outline for broad Pillow compatibility
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy > outline_width * outline_width:
                continue
            draw.multiline_text(
                (x + dx, y + dy),
                wrapped,
                font=font,
                fill=outline,
                anchor=anchor,
                align="center",
                spacing=6,
            )
    draw.multiline_text(
        (x, y),
        wrapped,
        font=font,
        fill=fill,
        anchor=anchor,
        align="center",
        spacing=6,
    )
