from __future__ import annotations


class ChromaMemoryClient:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    async def upsert_memory(self, *_: object, **__: object) -> None:
        return None

    async def search(self, *_: object, **__: object) -> list[dict[str, object]]:
        return []

