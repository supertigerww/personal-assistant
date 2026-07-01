from __future__ import annotations

import pytest

from core.config import Settings
from db.database import Database
from services.task_service import TaskService
from services.user_service import UserService


@pytest.fixture()
def settings(tmp_path):
    return Settings.model_construct(
        bot_token="test-token",
        xai_api_key="test-key",
        sqlite_path=str(tmp_path / "test.sqlite3"),
        prompt_path="prompts/system_prompt.txt",
        log_level="INFO",
        recent_message_limit=12,
        enable_image_generation=False,
        enable_chroma=False,
        task_normal_min_turns=8,
        task_normal_max_turns=15,
        task_intense_min_turns=5,
        task_intense_max_turns=10,
        task_offer_probability_normal=0.4,
        task_offer_probability_intense=0.6,
        task_retry_min_turns_normal=2,
        task_retry_max_turns_normal=4,
        task_retry_min_turns_intense=1,
        task_retry_max_turns_intense=2,
        aftercare_minutes=45,
        assets_images_path=str(tmp_path / "images"),
        assets_videos_path=str(tmp_path / "videos"),
        max_local_video_size_mb=45,
        media_send_probability_normal=0.35,
        media_send_probability_intense=0.55,
        media_send_probability_aftercare=0.0,
        media_send_probability_paused=0.0,
        media_random_fallback_probability=0.25,
        media_max_items_per_message=1,
        safewords_csv="red,红色,stop,pause,停,结束,暂停,停止,停下",
    )


@pytest.fixture()
async def database(settings):
    db = Database(settings.sqlite_path)
    await db.initialize()
    return db


@pytest.fixture()
async def user_service(database, settings):
    return UserService(database=database, settings=settings)


@pytest.fixture()
async def task_service(database, settings, user_service):
    return TaskService(database=database, settings=settings, user_service=user_service)
