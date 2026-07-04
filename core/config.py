from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bot_token: str
    bot_endpoint: str = "http://telegram-bot-api:8081"
    xai_api_key: str | None = None
    xai_model: str = "grok-4.3"
    xai_base_url: str = "https://api.x.ai/v1"
    xai_image_model: str = "grok-imagine-image-quality"
    xai_max_retries: int = 2
    xai_retry_delay_seconds: float = 1.0
    sqlite_path: str = "data/queen_bot.sqlite3"
    prompt_path: str = "prompts/system_prompt.txt"
    log_level: str = "INFO"
    recent_message_limit: int = 12
    enable_image_generation: bool = False
    enable_chroma: bool = False
    chroma_path: str = "data/chroma"
    chroma_collection_name: str = "queen_memories"
    memory_search_limit: int = 5
    memory_min_query_length: int = 4
    user_uploads_path: str = "data/user_uploads"
    enable_photo_vision: bool = True
    task_normal_min_turns: int = 10
    task_normal_max_turns: int = 18
    task_intense_min_turns: int = 6
    task_intense_max_turns: int = 12
    task_offer_probability_normal: float = 0.3
    task_offer_probability_intense: float = 0.5
    task_retry_min_turns_normal: int = 2
    task_retry_max_turns_normal: int = 4
    task_retry_min_turns_intense: int = 1
    task_retry_max_turns_intense: int = 2
    task_completion_score_delta: int = 2
    task_refusal_score_delta: int = -2
    task_failure_score_delta: int = -1
    task_ignore_score_delta: int = -1
    photo_task_normal_min_turns: int = 22
    photo_task_normal_max_turns: int = 36
    photo_task_intense_min_turns: int = 16
    photo_task_intense_max_turns: int = 26
    photo_task_offer_probability_normal: float = 0.12
    photo_task_offer_probability_intense: float = 0.2
    photo_task_retry_min_turns_normal: int = 4
    photo_task_retry_max_turns_normal: int = 8
    photo_task_retry_min_turns_intense: int = 3
    photo_task_retry_max_turns_intense: int = 6
    aftercare_minutes: int = Field(
        default=30,
        validation_alias=AliasChoices("AFTERCARE_MINUTES", "AFTERCARE_DURATION_MINUTES"),
    )
    intense_enter_compliance_score: int = 8
    intense_exit_compliance_score: int = 3
    red_safewords_csv: str = Field(default="red,红色,stop")
    yellow_safewords_csv: str = Field(default="yellow,黄色")
    safewords_csv: str | None = Field(default=None)
    assets_images_path: str = "assets/images"
    assets_videos_path: str = "assets/videos"
    generated_images_path: str = "data/generated_images"
    max_local_video_size_mb: int = 45
    media_send_probability_normal: float = 0.35
    media_send_probability_intense: float = 0.55
    media_send_probability_aftercare: float = 0.0
    media_send_probability_paused: float = 0.0
    media_random_fallback_probability: float = 0.25
    media_max_items_per_message: int = 1
    luna_visual_prompt_path: str = "prompts/luna_visual.txt"
    media_repeat_cooldown_hours: int = 48
    media_repeat_penalty_score: int = 24
    asset_meta_filename: str = ".meta.json"
    video_normal_min_turns: int = 18
    video_normal_max_turns: int = 28
    video_intense_min_turns: int = 10
    video_intense_max_turns: int = 18
    video_offer_probability_normal: float = 0.18
    video_offer_probability_intense: float = 0.28
    video_retry_min_turns_normal: int = 3
    video_retry_max_turns_normal: int = 6
    video_retry_min_turns_intense: int = 2
    video_retry_max_turns_intense: int = 4
    enable_llm_video_caption: bool = True
    video_caption_history_limit: int = 8
    video_rotation_score_band: int = 6
    video_folder_aliases_csv: str = (
        "sm=调教,SM,训练,支配,鞭打,束缚,女王,贱狗,跪下,学着,示范;"
        "pov=第一视角,视角,羞辱,撸,自慰,手,寸止,边缘,不许射,停下,憋住,对着录,盯着,POV,fpov,看着我,射"
    )
    @property
    def aftercare_duration_minutes(self) -> int:
        return self.aftercare_minutes

    @property
    def red_safewords(self) -> tuple[str, ...]:
        return self._parse_word_csv(self.red_safewords_csv)

    @property
    def yellow_safewords(self) -> tuple[str, ...]:
        return self._parse_word_csv(self.yellow_safewords_csv)

    @property
    def safewords(self) -> tuple[str, ...]:
        if self.safewords_csv:
            return self._parse_word_csv(self.safewords_csv)
        return self.red_safewords + self.yellow_safewords

    @staticmethod
    def _parse_word_csv(value: str) -> tuple[str, ...]:
        return tuple(word.strip().casefold() for word in value.split(",") if word.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
