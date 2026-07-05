from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)


class XAssetsService:
    """Service to query local X (Twitter) assets downloaded from user's script.
    DB contains posts with text, author, media paths (relative to x_assets dir), metadata.
    """

    def __init__(
        self,
        db_path: str = "/app/assets/x_data/x_assets.db",
        assets_root: str = "/app/assets/x_assets",
    ) -> None:
        # db_path and assets_root are container paths.
        # User sets HOST_X_ASSETS_PATH and HOST_X_DB_PATH in .env for docker mount.
        # DB may be in separate dir from media root.
        """DB and root are container paths after Docker volume mount.
        Set via .env HOST_X_ASSETS_PATH and CONTAINER_X_ASSETS_PATH.
        Assumes your download script saves 'media_paths' as list of paths RELATIVE to the assets_root (e.g. 'subfolder/post123.jpg').
        If your DB uses absolute host paths, adjust the media path construction here or normalize in your script.
        """
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
        limit: int = 5,
        styles: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search posts whose text matches any of the keywords (for humiliation).
        Returns list of dicts with text, author, media_paths (full container paths), metadata.
        NOTE: Assumes table 'posts' with columns: id, author, text, media_paths (JSON list of RELATIVE paths under x_assets), metadata (JSON), created_at.
        Adjust table name / columns to match your download script's schema if different.
        """
        if not keywords:
            keywords = ["羞辱", "调教", "女王", "绿帽", "母狗"]  # default

        conn = await self._get_conn()
        if not conn:
            return []

        # Build query: match any keyword in text (case insensitive, simple LIKE for simplicity)
        # For better, could use FTS, but assume simple.
        like_clauses = []
        params: list[str] = []
        for kw in keywords:
            like_clauses.append("text LIKE ?")
            params.append(f"%{kw}%")

        where = " OR ".join(like_clauses) if like_clauses else "1=1"

        # Optionally filter by style in metadata if stored
        if styles:
            # assume metadata has "styles" or text has them
            where += " AND (text LIKE ? OR metadata LIKE ?)"
            params.extend([f"%{styles}%", f"%{styles}%"])

        query = f"""
            SELECT id, author, text, media_paths, metadata, created_at
            FROM posts
            WHERE {where}
            ORDER BY RANDOM()
            LIMIT ?
        """
        params.append(str(limit))

        try:
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed to query X assets DB: %s", exc)
            return []

        results = []
        for row in rows:
            try:
                media_list = json.loads(row["media_paths"] or "[]")
            except Exception:
                media_list = []

            full_media = []
            for m in media_list:
                # media_paths are relative to the x_assets dir
                full_path = str(self.assets_root / m)
                full_media.append(full_path)

            meta = {}
            try:
                meta = json.loads(row["metadata"] or "{}")
            except Exception:
                pass

            results.append({
                "id": row["id"],
                "author": row["author"],
                "text": row["text"],
                "media_paths": full_media,  # container paths
                "metadata": meta,
                "created_at": row["created_at"],
            })

        logger.info("Fetched %s X humiliation posts for keywords=%s", len(results), keywords)
        return results

    async def get_random_humiliation_post(self, limit: int = 1) -> list[dict[str, Any]]:
        """Get random recent posts for variety."""
        conn = await self._get_conn()
        if not conn:
            return []

        query = """
            SELECT id, author, text, media_paths, metadata, created_at
            FROM posts
            ORDER BY RANDOM()
            LIMIT ?
        """
        try:
            async with conn.execute(query, (limit,)) as cursor:
                rows = await cursor.fetchall()
        except Exception as exc:
            logger.exception("Failed random X query: %s", exc)
            return []

        results = []
        for row in rows:
            try:
                media_list = json.loads(row["media_paths"] or "[]")
            except Exception:
                media_list = []
            full_media = [str(self.assets_root / m) for m in media_list]
            meta = json.loads(row["metadata"] or "{}") if row["metadata"] else {}
            results.append({
                "id": row["id"],
                "author": row["author"],
                "text": row["text"],
                "media_paths": full_media,
                "metadata": meta,
                "created_at": row["created_at"],
            })
        return results
