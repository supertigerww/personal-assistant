from pathlib import Path

import pytest

from core.image_text_composer import compose_humiliation_overlays, resolve_cjk_font_path


def test_compose_returns_original_when_no_phrases(tmp_path):
    img = tmp_path / "base.png"
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    Image.new("RGB", (200, 300), color=(30, 30, 30)).save(img)
    out = compose_humiliation_overlays(img, [], output_dir=tmp_path)
    assert out == str(img)


def test_compose_writes_new_file_with_phrases(tmp_path):
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("Pillow not installed")
    img = tmp_path / "base.png"
    Image.new("RGB", (400, 600), color=(40, 20, 20)).save(img)
    out = compose_humiliation_overlays(
        img,
        ["无脑狗只准看鞋尖！", "只许漏不准射！", "寸止！", "绿奴跪着看！"],
        output_dir=tmp_path,
    )
    assert Path(out).exists()
    assert Path(out).name.startswith("overlay_")
    # Output should be a real image
    with Image.open(out) as composed:
        assert composed.size == (400, 600)


def test_resolve_font_path_does_not_crash():
    # May be None on minimal CI without fonts; must not raise.
    path = resolve_cjk_font_path(None)
    assert path is None or path.is_file()
