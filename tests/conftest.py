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
        chroma_path="data/chroma",
        chroma_collection_name="queen_memories",
        memory_search_limit=5,
        memory_min_query_length=4,
        user_uploads_path=str(tmp_path / "user_uploads"),
        enable_photo_vision=False,
        task_normal_min_turns=10,
        task_normal_max_turns=18,
        task_intense_min_turns=6,
        task_intense_max_turns=12,
        task_offer_probability_normal=0.3,
        task_offer_probability_intense=0.5,
        task_retry_min_turns_normal=2,
        task_retry_max_turns_normal=4,
        task_retry_min_turns_intense=1,
        task_retry_max_turns_intense=2,
        task_completion_score_delta=2,
        task_refusal_score_delta=-2,
        task_failure_score_delta=-1,
        task_ignore_score_delta=-1,
        photo_task_normal_min_turns=22,
        photo_task_normal_max_turns=36,
        photo_task_intense_min_turns=16,
        photo_task_intense_max_turns=26,
        photo_task_offer_probability_normal=0.12,
        photo_task_offer_probability_intense=0.2,
        photo_task_retry_min_turns_normal=4,
        photo_task_retry_max_turns_normal=8,
        photo_task_retry_min_turns_intense=3,
        photo_task_retry_max_turns_intense=6,
        aftercare_minutes=45,
        intense_enter_compliance_score=8,
        intense_exit_compliance_score=3,
        red_safewords_csv="red,红色,stop",
        yellow_safewords_csv="yellow,黄色",
        safewords_csv=None,
        assets_images_path=str(tmp_path / "images"),
        assets_videos_path=str(tmp_path / "videos"),
        max_local_video_size_mb=45,
        media_send_probability_normal=0.35,
        media_send_probability_intense=0.55,
        media_send_probability_aftercare=0.0,
        media_send_probability_paused=0.0,
        media_random_fallback_probability=0.25,
        media_max_items_per_message=1,
        luna_visual_prompt_path="prompts/luna_visual.txt",
        media_repeat_cooldown_hours=48,
        media_repeat_penalty_score=24,
        asset_meta_filename=".meta.json",
        video_normal_min_turns=18,
        video_normal_max_turns=28,
        video_intense_min_turns=10,
        video_intense_max_turns=18,
        video_offer_probability_normal=0.18,
        video_offer_probability_intense=0.28,
        video_retry_min_turns_normal=3,
        video_retry_max_turns_normal=6,
        video_retry_min_turns_intense=2,
        video_retry_max_turns_intense=4,
        video_folder_aliases_csv="sm=调教,SM,训练;pov=第一视角,撸,寸止,羞辱",
        enable_llm_video_caption=False,
        video_caption_history_limit=8,
        video_rotation_score_band=6,
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
