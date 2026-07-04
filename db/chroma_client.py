from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when dependency missing
    chromadb = None
    CHROMADB_AVAILABLE = False


class ChromaMemoryClient:
    def __init__(
        self,
        *,
        enabled: bool = False,
        persist_path: str = "data/chroma",
        collection_name: str = "queen_memories",
    ) -> None:
        self.enabled = bool(enabled) and CHROMADB_AVAILABLE
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self._client: Any | None = None
        self._collection: Any | None = None

        if enabled and not CHROMADB_AVAILABLE:
            logger.warning("ENABLE_CHROMA is true but chromadb is not installed. Long-term memory disabled.")
            return

        if self.enabled:
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_path.as_posix())
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "Chroma memory enabled path=%s collection=%s count=%s",
                self.persist_path,
                self.collection_name,
                self._collection.count(),
            )

    async def upsert_memory(
        self,
        *,
        telegram_user_id: int,
        memory_id: str,
        text: str,
        category: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._collection is None:
            return

        cleaned = text.strip()
        if not cleaned:
            return

        payload = dict(metadata or {})
        payload["telegram_user_id"] = int(telegram_user_id)
        payload["category"] = category
        sanitized_metadata = self._sanitize_metadata(payload)

        await asyncio.to_thread(
            self._collection.upsert,
            ids=[memory_id],
            documents=[cleaned],
            metadatas=[sanitized_metadata],
        )

    async def search(
        self,
        *,
        telegram_user_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.enabled or self._collection is None:
            return []

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        results = await asyncio.to_thread(
            self._query_sync,
            telegram_user_id=telegram_user_id,
            query=cleaned_query,
            limit=max(1, int(limit)),
        )
        return results

    def _query_sync(self, *, telegram_user_id: int, query: str, limit: int) -> list[dict[str, Any]]:
        assert self._collection is not None

        response = self._collection.query(
            query_texts=[query],
            n_results=limit,
            where={"telegram_user_id": telegram_user_id},
            include=["documents", "metadatas", "distances"],
        )

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        matches: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            if not document:
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            matches.append(
                {
                    "text": document,
                    "category": metadata.get("category", "unknown"),
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return matches

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        sanitized: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif value is None:
                continue
            else:
                sanitized[key] = str(value)
        return sanitized


def create_chroma_client(settings: Any) -> ChromaMemoryClient:
    return ChromaMemoryClient(
        enabled=bool(getattr(settings, "enable_chroma", False)),
        persist_path=str(getattr(settings, "chroma_path", "data/chroma")),
        collection_name=str(getattr(settings, "chroma_collection_name", "queen_memories")),
    )