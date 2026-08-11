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

    DEFAULT_SEARCH_KEYWORDS: tuple[str, ...] = (
        "羞辱",
        "调教",
        "女王",
        "绿帽",
        "母狗",
        "圣水",
        "寸止",
        "足控",
        "脚",
        "黑丝",
        "丝袜",
        "高跟",
        "女装",
        "伪娘",
        "锁精",
        "跪",
    )
    ALLOWED_CHAT_KEYWORDS: frozenset[str] = frozenset(
        {
            "羞辱",
            "调教",
            "女王",
            "绿帽",
            "绿奴",
            "母狗",
            "圣水",
            "寸止",
            "边缘",
            "足控",
            "脚",
            "鞋",
            "黑丝",
            "丝袜",
            "高跟",
            "女装",
            "伪娘",
            "锁精",
            "锁奴",
            "跪",
            "踩",
            "口交",
            "sissy",
            "cuck",
            "femdom",
            "joi",
        }
    )

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

    @classmethod
    def sanitize_search_keywords(cls, keywords: list[str] | None) -> list[str]:
        """Keep only short, high-value terms — drop chat fragments / English greetings."""
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in keywords or []:
            token = (raw or "").strip()
            if not token:
                continue
            # Drop pure English greetings / filler
            lower = token.casefold()
            if lower in {"hi", "hello", "hey", "ok", "yes", "no", "the", "and", "for"}:
                continue
            # Chinese chat crumbs are usually long; keep short fetish terms only
            if len(token) > 8:
                continue
            # Prefer allowlist; also allow 2–4 char pure CJK fetish-ish tokens
            is_cjk = all("\u4e00" <= ch <= "\u9fff" for ch in token)
            if token.casefold() not in {a.casefold() for a in cls.ALLOWED_CHAT_KEYWORDS}:
                if not (is_cjk and 2 <= len(token) <= 4):
                    continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(token)
            if len(cleaned) >= 8:
                break
        if not cleaned:
            cleaned = list(cls.DEFAULT_SEARCH_KEYWORDS)
        return cleaned

    async def search_humiliation_posts(
        self,
        keywords: list[str],
        limit: int = 1,
        styles: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using media_search_fts for relevance, then join to posts + media.

        Always prefers files that exist on disk. If FTS misses or files are gone,
        falls back to random posts that still have valid media paths.
        """
        safe_limit = max(1, int(limit))
        keywords = self.sanitize_search_keywords(keywords)
        _ = styles  # reserved for future style-weighted ranking

        conn = await self._get_conn()
        if not conn:
            logger.warning("X assets DB unavailable; cannot search posts.")
            return []

        fts_rows = await self._query_fts_rows(keywords, fetch_limit=safe_limit * 8)
        fts_hit_count = len(fts_rows)
        results = await self._rows_to_posts_with_existing_media(fts_rows, limit=safe_limit)
        existing_count = len(results)

        if results:
            logger.info(
                "X FTS ok keywords=%s fts_rows=%s existing=%s",
                keywords,
                fts_hit_count,
                existing_count,
            )
            return results

        # Distinguish empty index hits vs files-all-missing for ops logs.
        if fts_hit_count == 0:
            logger.info(
                "X FTS empty for keywords=%s; falling back to random existing media",
                keywords,
            )
        else:
            logger.warning(
                "X FTS matched %s rows for keywords=%s but 0 files exist on disk "
                "(check HOST_X_ASSETS_PATH mount / local_path prefixes); "
                "falling back to random existing media",
                fts_hit_count,
                keywords,
            )

        random_results = await self.get_random_humiliation_post(limit=safe_limit)
        if random_results:
            logger.info(
                "X random fallback returned %s post(s) with existing files",
                len(random_results),
            )
        else:
            logger.warning(
                "X random fallback also empty — DB empty or no media files exist under assets_root=%s",
                self.assets_root,
            )
        return random_results

    async def _query_fts_rows(self, keywords: list[str], *, fetch_limit: int) -> list[Any]:
        conn = await self._get_conn()
        if not conn:
            return []

        # Try full OR first; if FTS errors or empty, try single-keyword queries.
        candidates = [" OR ".join(keywords)] + list(keywords)
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
        all_rows: list[Any] = []
        seen_paths: set[str] = set()
        for match_terms in candidates:
            if not match_terms:
                continue
            try:
                async with conn.execute(query, (match_terms, fetch_limit)) as cursor:
                    rows = await cursor.fetchall()
            except Exception as exc:
                logger.debug("X FTS MATCH failed for %r: %s", match_terms, exc)
                continue
            for row in rows:
                key = f"{row['tweet_id']}:{row['local_path']}"
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                all_rows.append(row)
            if len(all_rows) >= fetch_limit:
                break
        return all_rows[:fetch_limit]

    async def _rows_to_posts_with_existing_media(
        self,
        rows: list[Any],
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        posts_dict: dict[str, dict[str, Any]] = {}
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

        results: list[dict[str, Any]] = []
        for post in posts_dict.values():
            valid_paths = await self._filter_existing_paths(post.get("media_paths", []))
            if not valid_paths:
                continue
            post["media_paths"] = valid_paths[:1]
            results.append(post)
            if len(results) >= limit:
                break
        return results

    async def _filter_existing_paths(self, paths: list[str]) -> list[str]:
        valid: list[str] = []
        for mp in paths:
            try:
                if mp and Path(mp).exists():
                    valid.append(mp)
                elif mp:
                    logger.warning("X asset file missing on disk, skipping: %s", mp)
                    try:
                        folder = Path(mp).parts[0] if Path(mp).parts else ""
                        # Prefer relative folder under assets root if path is absolute
                        try:
                            rel = Path(mp).relative_to(self.assets_root)
                            folder = rel.parts[0] if rel.parts else folder
                        except ValueError:
                            pass
                        if folder:
                            await self.cleanup_folder(folder)
                    except Exception:
                        pass
            except Exception:
                logger.warning("X asset path invalid, skipping: %s", mp)
        return valid

    async def get_random_humiliation_post(self, limit: int = 1) -> list[dict[str, Any]]:
        """Get random posts that still have media files on disk."""
        conn = await self._get_conn()
        if not conn:
            return []

        safe_limit = max(1, int(limit))
        # Oversample: many DB rows may point at deleted files.
        fetch_n = max(safe_limit * 15, 30)
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
            async with conn.execute(query, (fetch_n,)) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed random X query: %s", exc)
            return []

        return await self._rows_to_posts_with_existing_media(rows, limit=safe_limit)

    def _build_full_media_path(self, local_path: str) -> str:
        """Build full container path, stripping common wrong prefixes that may be stored in the DB.
        The mount makes the images dir available at self.assets_root (/app/assets/x_assets).
        """
        if not local_path:
            return ""
        cleaned = local_path.lstrip("/")
        # Strip bad prefixes that the download script apparently recorded.
        bad_prefixes = [
            "app/images/", "images/", "app/assets/images/", "assets/images/",
            "app/", "docker/assets/images/", "/app/assets/x_assets/"
        ]
        for bad in bad_prefixes:
            if cleaned.startswith(bad):
                cleaned = cleaned[len(bad):]
                break
        full_path = str(self.assets_root / cleaned)
        # Log for debugging path issues
        if local_path != cleaned:
            logger.debug("Built X media path: original=%s -> %s", local_path, full_path)
        return full_path

    async def cleanup_folder(self, folder: str) -> int:
        """Delete all media records whose local_path belongs to the given top-level folder.
        Used when user deletes a subfolder (e.g. 'Linmistresssh/') so we never try to serve
        missing files again. Also removes orphan posts.
        """
        if not folder:
            return 0
        conn = await self._get_conn()
        if not conn:
            return 0

        prefix = folder.rstrip("/") + "/"
        try:
            # Delete matching media
            async with conn.execute(
                "DELETE FROM media WHERE local_path LIKE ?",
                (prefix + "%",)
            ) as cursor:
                deleted = cursor.rowcount if getattr(cursor, "rowcount", None) is not None else 0

            # Clean up orphan posts
            await conn.execute(
                """
                DELETE FROM posts
                WHERE id NOT IN (SELECT DISTINCT COALESCE(post_id, 0) FROM media)
                  AND tweet_id NOT IN (SELECT DISTINCT tweet_id FROM media)
                """
            )

            await conn.commit()
            if deleted > 0:
                logger.info("Cleaned up %s media records for deleted X folder: %s", deleted, folder)
            return deleted
        except Exception as exc:
            logger.exception("Failed to cleanup folder %s from X DB: %s", folder, exc)
            return 0
