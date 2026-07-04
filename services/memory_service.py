from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryFact:
    category: str
    text: str


class MemoryService:
    EXTRACTION_PATTERNS: tuple[tuple[str, str], ...] = (
        (r"记住[：:\s]*(.+)", "explicit_remember"),
        (r"别忘了[：:\s]*(.+)", "explicit_remember"),
        (r"我喜欢\s*(.+)", "preference"),
        (r"我讨厌\s*(.+)", "dislike"),
        (r"我的癖好[是\s]*(.+)", "preference"),
        (r"我习惯\s*(.+)", "habit"),
        (r"叫我\s*(.+)", "preference"),
        (r"称呼我\s*(.+)", "preference"),
    )

    def __init__(
        self,
        *,
        database: Any,
        chroma_client: Any | None = None,
        settings: Any | None = None,
    ) -> None:
        self.database = database
        self.chroma_client = chroma_client
        self.settings = settings

    async def store_message(
        self,
        telegram_user_id: int,
        role: str,
        content: str,
        *,
        message_kind: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.database.execute(
            """
            INSERT INTO conversation_events (
                telegram_user_id, role, content, message_kind, metadata, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                telegram_user_id,
                role,
                content,
                message_kind,
                json.dumps(metadata or {}, ensure_ascii=False),
                utc_now_iso(),
            ),
        )

    async def recent_messages(self, telegram_user_id: int, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = await self.database.fetchall(
            """
            SELECT role, content, message_kind, metadata, created_at
            FROM conversation_events
            WHERE telegram_user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (telegram_user_id, limit),
        )
        events = [
            {
                "role": row["role"],
                "content": row["content"],
                "message_kind": row["message_kind"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
        return list(reversed(events))

    async def recent_video_captions(self, telegram_user_id: int, *, limit: int = 8) -> list[str]:
        safe_limit = max(1, int(limit))
        rows = await self.database.fetchall(
            """
            SELECT metadata
            FROM conversation_events
            WHERE telegram_user_id = ? AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (telegram_user_id, safe_limit * 4),
        )

        captions: list[str] = []
        seen: set[str] = set()
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            raw_caption = metadata.get("video_caption")
            if not isinstance(raw_caption, str):
                continue
            cleaned = raw_caption.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            captions.append(cleaned)
            if len(captions) >= safe_limit:
                break
        return captions

    async def patch_last_assistant_metadata(
        self,
        telegram_user_id: int,
        patch: dict[str, Any],
    ) -> None:
        if not patch:
            return

        row = await self.database.fetchone(
            """
            SELECT id, metadata
            FROM conversation_events
            WHERE telegram_user_id = ? AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (telegram_user_id,),
        )
        if row is None:
            return

        metadata = json.loads(row["metadata"] or "{}")
        metadata.update(patch)
        await self.database.execute(
            """
            UPDATE conversation_events
            SET metadata = ?
            WHERE id = ?
            """,
            (json.dumps(metadata, ensure_ascii=False), row["id"]),
        )

    async def ingest_user_turn(
        self,
        telegram_user_id: int,
        user_text: str,
        *,
        explicit_limits: list[str] | None = None,
    ) -> None:
        if not self._chroma_enabled():
            return

        facts = self.extract_memories_from_text(user_text)
        for limit in explicit_limits or []:
            cleaned = limit.strip()
            if cleaned:
                facts.append(MemoryFact(category="dislike", text=f"用户不喜欢：{cleaned}"))

        for fact in facts:
            await self._upsert_fact(telegram_user_id, fact)

    async def ingest_profile_updates(
        self,
        telegram_user_id: int,
        *,
        dislikes: list[str] | None = None,
        hard_limits: list[str] | None = None,
        notes: list[str] | None = None,
    ) -> None:
        if not self._chroma_enabled():
            return

        for dislike in dislikes or []:
            cleaned = dislike.strip()
            if cleaned:
                await self._upsert_fact(telegram_user_id, MemoryFact(category="dislike", text=f"用户不喜欢：{cleaned}"))
        for hard_limit in hard_limits or []:
            cleaned = hard_limit.strip()
            if cleaned:
                await self._upsert_fact(
                    telegram_user_id,
                    MemoryFact(category="hard_limit", text=f"用户硬限：{cleaned}"),
                )
        for note in notes or []:
            cleaned = note.strip()
            if cleaned:
                await self._upsert_fact(telegram_user_id, MemoryFact(category="note", text=cleaned))

    async def recall_relevant(
        self,
        telegram_user_id: int,
        *,
        query: str,
        profile: Any | None = None,
    ) -> list[dict[str, Any]]:
        if not self._chroma_enabled():
            return []

        search_query = self._build_search_query(query=query, profile=profile)
        if not search_query:
            return []

        try:
            matches = await self.chroma_client.search(
                telegram_user_id=telegram_user_id,
                query=search_query,
                limit=self._search_limit(),
            )
        except Exception as exc:
            logger.exception("Failed to recall long-term memories for user_id=%s: %s", telegram_user_id, exc)
            return []

        deduped: list[dict[str, Any]] = []
        seen_text: set[str] = set()
        for match in matches:
            text = str(match.get("text", "")).strip()
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            deduped.append(match)
        return deduped

    @classmethod
    def extract_memories_from_text(cls, text: str) -> list[MemoryFact]:
        facts: list[MemoryFact] = []
        for pattern, category in cls.EXTRACTION_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!?:;，。！？")
            if not candidate:
                continue
            label = cls._category_label(category)
            normalized = f"{label}{candidate}"
            if not any(fact.text == normalized for fact in facts):
                facts.append(MemoryFact(category=category, text=normalized))
            if category == "explicit_remember":
                return facts[:1]
        return facts

    async def _upsert_fact(self, telegram_user_id: int, fact: MemoryFact) -> None:
        if not self._chroma_enabled():
            return

        memory_id = self._memory_id(telegram_user_id, fact.category, fact.text)
        try:
            await self.chroma_client.upsert_memory(
                telegram_user_id=telegram_user_id,
                memory_id=memory_id,
                text=fact.text,
                category=fact.category,
            )
            logger.info(
                "Stored long-term memory user_id=%s category=%s memory_id=%s",
                telegram_user_id,
                fact.category,
                memory_id,
            )
        except Exception as exc:
            logger.exception(
                "Failed to store long-term memory user_id=%s category=%s: %s",
                telegram_user_id,
                fact.category,
                exc,
            )

    def _build_search_query(self, *, query: str, profile: Any | None) -> str:
        parts: list[str] = []
        cleaned_query = query.strip()
        if len(cleaned_query) >= self._min_query_length():
            parts.append(cleaned_query)

        if profile is not None:
            if getattr(profile, "dislikes", None):
                parts.append(" ".join(profile.dislikes[:3]))
            if getattr(profile, "notes", None):
                parts.append(" ".join(profile.notes[-3:]))
            if getattr(profile, "hard_limits", None):
                parts.append(" ".join(profile.hard_limits[:2]))

        return " ".join(part for part in parts if part).strip()

    def _chroma_enabled(self) -> bool:
        return self.chroma_client is not None and bool(getattr(self.chroma_client, "enabled", False))

    def _search_limit(self) -> int:
        if self.settings is None:
            return 5
        return max(1, int(getattr(self.settings, "memory_search_limit", 5)))

    def _min_query_length(self) -> int:
        if self.settings is None:
            return 4
        return max(1, int(getattr(self.settings, "memory_min_query_length", 4)))

    @staticmethod
    def _memory_id(telegram_user_id: int, category: str, text: str) -> str:
        digest = hashlib.sha256(f"{telegram_user_id}:{category}:{text.casefold()}".encode("utf-8")).hexdigest()[:16]
        return f"{telegram_user_id}:{category}:{digest}"

    @staticmethod
    def _category_label(category: str) -> str:
        labels = {
            "explicit_remember": "用户要求记住：",
            "preference": "用户偏好：",
            "dislike": "用户不喜欢：",
            "habit": "用户习惯：",
            "hard_limit": "用户硬限：",
            "note": "档案备注：",
        }
        return labels.get(category, "")