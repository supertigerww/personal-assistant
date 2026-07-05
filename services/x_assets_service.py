from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class XAssetsService:
    """Service for local X assets DB (posts + media from user's download script).
    Schema:
      - posts: id, tweet_id, author_x_user_id, username, created_at, tweet_url, text, lang, raw_json, inserted_at
      - media: id, post_id, media_key, type, local_path, original_url, alt_text, width, height, image_description, tags, ...
      - media_search_fts: tweet_id, post_text, alt_text, image_description, tags
    local_path in media is relative to the x_assets mount point.
    """

    def __init__(
        self,
        db_path: str = "/app/assets/x_data/x_assets.db",
        assets_root: str = "/app/assets/x_assets",
    ) -> None:
        # db_path and assets_root are container paths.
        # Mount:
        #   HOST_X_ASSETS_PATH (your images dir) -> CONTAINER_X_ASSETS_PATH=/app/assets/x_assets
        #   HOST_X_DB_PATH (your data dir with x_assets.db) -> /app/assets/x_data
        # local_path in DB media table is relative to the x_assets mount.
        self.db_path = Path(db_path)
        self.assets_root = Path(assets_root)
        self._conn: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            if not self.db_path.exists():
                logger.warning("X assets DB not found at %s", self.db_path)
                # return a dummy conn? but better raise or handle
            self._conn = await aiosqlite.connect(self.db_path.as_posix())
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def search_humiliation_posts(
        self,
        keywords: list[str],
        limit: int = 1,
        styles: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using media_search_fts for relevance, then join to posts + media.
        Returns posts with text, author (username), media_paths (full container paths), etc.
        Keywords are used for FTS MATCH.
        """
        if not keywords:
            keywords = ["羞辱", "调教", "女王", "绿帽", "母狗", "圣水", "寸止"]

        conn = await self._get_conn()
        if not conn:
            return []

        # Build FTS match query (simple OR for the keywords)
        match_terms = " OR ".join(f'"{kw}"' for kw in keywords if kw)
        if not match_terms:
            match_terms = "*"

        # Query via FTS for relevant tweet_ids, then join posts + media
        # Limit fetch to avoid huge results, then take distinct posts
        query = """
            SELECT DISTINCT
                p.id,
                p.tweet_id,
                p.username,
                p.text,
                p.tweet_url,
                p.created_at,
                m.local_path,
                m.type,
                m.alt_text,
                m.image_description,
                m.tags
            FROM media_search_fts fts
            JOIN posts p ON p.tweet_id = fts.tweet_id
            JOIN media m ON (m.post_id = p.id OR m.post_id = p.tweet_id)
            WHERE fts.media_search_fts MATCH ?
              AND m.type IN ('photo', 'video')
              AND m.local_path IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """

        try:
            async with conn.execute(query, (match_terms, limit * 3)) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed to query X assets DB: %s", exc)
            return []

        # Group media by post
        posts_dict = {}
        for row in rows:
            tweet_id = row["tweet_id"]
            if tweet_id not in posts_dict:
                posts_dict[tweet_id] = {
                    "id": row["id"],
                    "tweet_id": tweet_id,
                    "author": row["username"],
                    "text": row["text"],
                    "tweet_url": row["tweet_url"],
                    "created_at": row["created_at"],
                    "media_paths": [],
                }
            local_path = row["local_path"]
            if local_path:
                full_path = self._build_full_media_path(local_path)
                posts_dict[tweet_id]["media_paths"].append(full_path)

        results = []
        for p in list(posts_dict.values())[:limit]:
            if p.get("media_paths"):
                p["media_paths"] = p["media_paths"][:1]
            results.append(p)
        logger.info("Fetched %s X humiliation posts for keywords=%s via FTS", len(results), keywords)
        return results

    async def get_random_humiliation_post(self, limit: int = 1) -> list[dict[str, Any]]:
        """Get random posts with media for variety (uses posts + media join)."""
        conn = await self._get_conn()
        if not conn:
            return []

        query = """
            SELECT p.id, p.tweet_id, p.username, p.text, p.tweet_url, p.created_at,
                   m.local_path, m.type
            FROM posts p
            JOIN media m ON (m.post_id = p.id OR m.post_id = p.tweet_id)
            WHERE m.type IN ('photo', 'video')
              AND m.local_path IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """
        try:
            async with conn.execute(query, (limit * 3,)) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed random X query: %s", exc)
            return []

        posts_dict = {}
        for row in rows:
            tweet_id = row["tweet_id"]
            if tweet_id not in posts_dict:
                posts_dict[tweet_id] = {
                    "id": row["id"],
                    "tweet_id": tweet_id,
                    "author": row["username"],
                    "text": row["text"],
                    "tweet_url": row["tweet_url"],
                    "created_at": row["created_at"],
                    "media_paths": [],
                }
            local_path = row["local_path"]
            if local_path:
                full_path = self._build_full_media_path(local_path)
                posts_dict[tweet_id]["media_paths"].append(full_path)

        results = []
        for p in list(posts_dict.values())[:limit]:
            if p.get("media_paths"):
                p["media_paths"] = p["media_paths"][:1]
            results.append(p)
        return results

    def _build_full_media_path(self, local_path: str) -> str:
        """Build full container path, stripping common wrong prefixes that may be stored in the DB.
        The mount makes the images dir available at self.assets_root (/app/assets/x_assets).
        """
        if not local_path:
            return ""
        cleaned = local_path.lstrip("/")
        # Strip bad prefixes that the download script apparently recorded (e.g. 'app/images/...')
        # This fixes paths like /app/assets/x_assets/app/images/... becoming correct /app/assets/x_assets/...
        bad_prefixes = [
            "app/images/", "images/", "app/assets/images/", "assets/images/",
            "app/", "docker/assets/images/"
        ]
        for bad in bad_prefixes:
            if cleaned.startswith(bad):
                cleaned = cleaned[len(bad):]
                break
        return str(self.assets_root / cleaned)
