from __future__ import annotations

import pytest

from services.memory_service import MemoryService


class FakeChromaClient:
    def __init__(self) -> None:
        self.enabled = True
        self._records: dict[str, dict[str, object]] = {}

    async def upsert_memory(
        self,
        *,
        telegram_user_id: int,
        memory_id: str,
        text: str,
        category: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._records[memory_id] = {
            "telegram_user_id": telegram_user_id,
            "text": text,
            "category": category,
            "metadata": metadata or {},
        }

    async def search(
        self,
        *,
        telegram_user_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        normalized_query = query.casefold()
        for record in self._records.values():
            if record["telegram_user_id"] != telegram_user_id:
                continue
            text = str(record["text"])
            if any(token in text.casefold() for token in normalized_query.split() if len(token) >= 2):
                matches.append(
                    {
                        "text": text,
                        "category": record["category"],
                        "metadata": record["metadata"],
                        "distance": 0.1,
                    }
                )
            if len(matches) >= limit:
                break
        return matches


@pytest.fixture()
def memory_service(database, settings):
    return MemoryService(
        database=database,
        chroma_client=FakeChromaClient(),
        settings=settings,
    )


def test_extract_memories_from_text():
    facts = MemoryService.extract_memories_from_text("记住我喜欢穿丝袜")
    assert len(facts) == 1
    assert facts[0].category == "explicit_remember"
    assert "丝袜" in facts[0].text

    preference_facts = MemoryService.extract_memories_from_text("我喜欢被叫贱狗")
    assert len(preference_facts) == 1
    assert preference_facts[0].category == "preference"


@pytest.mark.asyncio
async def test_ingest_and_recall_user_memory(memory_service):
    await memory_service.ingest_user_turn(41, "记住我喜欢穿黑丝")

    recalled = await memory_service.recall_relevant(41, query="黑丝 丝袜")

    assert len(recalled) >= 1
    assert any("黑丝" in item["text"] for item in recalled)


@pytest.mark.asyncio
async def test_ingest_profile_updates(memory_service):
    await memory_service.ingest_profile_updates(
        42,
        dislikes=["公开羞辱"],
        hard_limits=["实拍"],
        notes=["称呼偏好：贱狗"],
    )

    recalled = await memory_service.recall_relevant(42, query="羞辱 贱狗")

    assert len(recalled) >= 2


@pytest.mark.asyncio
async def test_recall_disabled_without_chroma(database, settings):
    service = MemoryService(database=database, chroma_client=None, settings=settings)
    await service.ingest_user_turn(43, "记住我喜欢乳胶")
    recalled = await service.recall_relevant(43, query="乳胶")
    assert recalled == []


@pytest.mark.asyncio
async def test_recent_video_captions_reads_assistant_metadata(database, settings):
    service = MemoryService(database=database, chroma_client=None, settings=settings)
    await service.store_message(51, "assistant", "铺垫文字", metadata={"video_caption": "手别停。"})
    await service.store_message(51, "assistant", "另一条", metadata={"video_caption": "寸止憋住。"})
    await service.store_message(51, "assistant", "无视频", metadata={})

    captions = await service.recent_video_captions(51, limit=5)

    assert captions == ["寸止憋住。", "手别停。"]


@pytest.mark.asyncio
async def test_patch_last_assistant_metadata_updates_latest_row(database, settings):
    service = MemoryService(database=database, chroma_client=None, settings=settings)
    await service.store_message(52, "assistant", "旧回复", metadata={"video_caption": None})
    await service.store_message(52, "assistant", "新回复", metadata={})

    await service.patch_last_assistant_metadata(52, {"video_caption": "盯着看。"})
    recent = await service.recent_messages(52, limit=1)

    assert recent[0]["metadata"]["video_caption"] == "盯着看。"