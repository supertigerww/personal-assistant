from pathlib import Path

from core.media_categories import VideoCategoryIndex


def test_video_category_index_matches_folder_name_and_aliases(tmp_path):
    videos_root = tmp_path / "videos"
    (videos_root / "sm").mkdir(parents=True)
    (videos_root / "pov").mkdir(parents=True)
    (videos_root / "sm" / "a.mp4").write_bytes(b"x")
    (videos_root / "pov" / "b.mp4").write_bytes(b"x")

    index = VideoCategoryIndex.build(
        videos_path=videos_root,
        video_paths=[
            videos_root / "sm" / "a.mp4",
            videos_root / "pov" / "b.mp4",
        ],
        aliases_csv="sm=调教,SM,训练;pov=第一视角,撸,寸止",
    )

    assert index.folder_counts == {"sm": 1, "pov": 1}
    assert index.match_categories_in_text("跪好看看SM调教示范") == ["sm"]
    assert index.match_categories_in_text("第一视角寸止不许射") == ["pov"]
    assert index.format_for_context() == "pov(1)[第一视角/撸/寸止], sm(1)[调教/sm/训练]"


def test_primary_category_for_path_uses_top_level_folder(tmp_path):
    videos_root = tmp_path / "videos"
    clip = videos_root / "pov" / "set01" / "clip.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"x")

    index = VideoCategoryIndex.build(
        videos_path=videos_root,
        video_paths=[clip],
        aliases_csv="",
    )

    assert index.primary_category_for_path(videos_path=videos_root, asset_path=clip) == "pov"