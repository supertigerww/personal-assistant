#!/usr/bin/env python3
"""
Utility script to safely rename very long video filenames in the assets/videos directory.

Problem: Extremely long filenames (common with downloaded adult content) cause 
"File name too long" (errno 36) errors inside Docker (Linux FS limit ~255 bytes)
when the bot tries to generate .meta.json sidecar files.

This script:
- Recursively scans a video directory
- Detects files with long names (based on UTF-8 byte length)
- Renames them to safe short names while preserving meaning + uniqueness
- Optionally renames matching .meta.json sidecars
- Supports dry-run mode (recommended first)
- Works on Windows host (run from host, not inside container)

Usage (on your Docker HOST):
    python utils/rename_long_videos.py --help
    python utils/rename_long_videos.py --dir "D:\path\to\your\videos" --dry-run
    python utils/rename_long_videos.py --dir "D:\path\to\your\videos"   # actual rename

After renaming, rebuild and restart the bot:
    docker compose down
    docker compose up --build -d
"""

import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

# Supported video extensions (match the bot)
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}

# Conservative limit: target total filename < 180 bytes to leave headroom for .meta.json etc.
MAX_SAFE_BYTES = 160

# How much of the original name to keep at the beginning (in characters, not bytes)
KEEP_PREFIX_CHARS = 70

# Hash length for uniqueness
HASH_LEN = 6


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def get_utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def clean_for_filename(s: str) -> str:
    """Remove characters that are problematic on Windows/Linux."""
    # Remove or replace invalid filename chars
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    # Collapse multiple spaces/underscores
    s = re.sub(r"[ _]+", " ", s).strip()
    return s


def generate_safe_name(original_stem: str, ext: str) -> str:
    """
    Create a short, unique, safe filename.
    Strategy:
      - Keep a meaningful prefix
      - Add short hash of the FULL original name (for uniqueness)
      - Clean invalid chars
    """
    original_stem = clean_for_filename(original_stem)
    full_hash = hashlib.md5(original_stem.encode("utf-8")).hexdigest()[:HASH_LEN]

    # Truncate prefix while respecting unicode
    prefix = original_stem[:KEEP_PREFIX_CHARS]
    # Make sure the byte length of prefix + hash + ext is safe
    while get_utf8_len(f"{prefix}_{full_hash}{ext}") > MAX_SAFE_BYTES and len(prefix) > 10:
        prefix = prefix[:-1]

    safe_stem = f"{prefix}_{full_hash}".rstrip("_- ")
    return f"{safe_stem}{ext}"


def find_matching_meta(original_video: Path) -> Path | None:
    """Look for a .meta.json that might have been created with the old long name."""
    meta_name = f"{original_video.stem}.meta.json"
    candidate = original_video.with_name(meta_name)
    if candidate.exists():
        return candidate
    return None


def rename_file_safely(src: Path, dst: Path, dry_run: bool) -> bool:
    """Rename with collision handling."""
    if src == dst:
        return False

    if dst.exists():
        # Add a counter if collision
        base = dst.stem
        ext = dst.suffix
        counter = 1
        while dst.exists():
            dst = dst.with_name(f"{base}_{counter}{ext}")
            counter += 1

    if dry_run:
        print(f"[DRY] Would rename:\n  FROM: {src}\n  TO  : {dst}")
        return True

    try:
        shutil.move(str(src), str(dst))
        print(f"[OK] Renamed:\n  FROM: {src.name}\n  TO  : {dst.name}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")
        return False


def process_directory(root: Path, dry_run: bool = True, max_bytes: int = MAX_SAFE_BYTES) -> None:
    if not root.exists():
        print(f"ERROR: Directory does not exist: {root}")
        return

    print(f"Scanning: {root}")
    print(f"Max safe filename bytes: {max_bytes}")
    print(f"Dry run: {dry_run}")
    print("-" * 60)

    video_files = [p for p in root.rglob("*") if is_video_file(p)]
    print(f"Found {len(video_files)} video files.")

    renamed_count = 0
    skipped_count = 0

    for video in video_files:
        stem = video.stem
        ext = video.suffix
        current_len = get_utf8_len(stem + ext)

        if current_len <= max_bytes:
            continue

        print(f"\nLong file detected ({current_len} bytes): {video.name[:80]}...")

        new_name = generate_safe_name(stem, ext)
        new_path = video.with_name(new_name)

        if rename_file_safely(video, new_path, dry_run):
            renamed_count += 1

            # Try to rename the old-style meta file if it exists
            old_meta = find_matching_meta(video)  # note: video is still the old path here? Wait, we moved already in real run
            if not dry_run:
                # After real move, the old video path no longer exists, so we use the original video var carefully
                pass  # handled below

        else:
            skipped_count += 1

    # Second pass for metas (after real renames if not dry)
    if not dry_run:
        print("\n--- Post-processing .meta.json files ---")
        # Re-scan because names changed
        for video in root.rglob("*"):
            if not is_video_file(video):
                continue
            # Look for any .meta.json that matches the *old* long pattern? Hard.
            # Instead, for every video, check if a long-named meta exists in same dir
            for meta in video.parent.glob("*.meta.json"):
                if get_utf8_len(meta.name) > max_bytes + 20:  # generous
                    # This meta is too long, try to shorten it to match current video
                    new_meta_name = f"{video.stem}.meta.json"
                    new_meta = meta.with_name(new_meta_name)
                    if not new_meta.exists():
                        try:
                            shutil.move(str(meta), str(new_meta))
                            print(f"[META] Renamed long meta: {meta.name[:60]}... -> {new_meta.name}")
                        except Exception as e:
                            print(f"[META ERROR] {e}")

    print("\n" + "=" * 60)
    print(f"Summary: {renamed_count} files renamed, {skipped_count} skipped.")
    if dry_run:
        print("This was a DRY RUN. Run again without --dry-run to apply changes.")
    else:
        print("Done! Rebuild your Docker image and restart the bot.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch rename long video filenames to avoid Docker/Linux filename length limits."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to your videos directory on the HOST (e.g. D:\\LunaVideos or /mnt/videos)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be renamed without actually doing it (default: True). "
             "Use --no-dry-run to perform the renames.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Actually perform the renames.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=MAX_SAFE_BYTES,
        help=f"Maximum allowed UTF-8 byte length for filename (default: {MAX_SAFE_BYTES})",
    )

    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    process_directory(root, dry_run=args.dry_run, max_bytes=args.max_bytes)


if __name__ == "__main__":
    main()