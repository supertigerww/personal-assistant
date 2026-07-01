from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bot_token: str
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
    task_normal_min_turns: int = 8
    task_normal_max_turns: int = 15
    task_intense_min_turns: int = 5
    task_intense_max_turns: int = 10
    aftercare_minutes: int = Field(
        default=30,
        validation_alias=AliasChoices("AFTERCARE_MINUTES", "AFTERCARE_DURATION_MINUTES"),
    )
    assets_images_path: str = "assets/images"
    assets_videos_path: str = "assets/videos"
    max_local_video_size_mb: int = 45
    media_send_probability_normal: float = 0.35
    media_send_probability_intense: float = 0.55
    media_send_probability_aftercare: float = 0.0
    media_send_probability_paused: float = 0.0
    media_random_fallback_probability: float = 0.25
    media_max_items_per_message: int = 1
    safewords_csv: str = Field(default="red,红色,stop,pause,停,结束,暂停,停止,停下")

    @property
    def aftercare_duration_minutes(self) -> int:
        return self.aftercare_minutes

    @property
    def safewords(self) -> tuple[str, ...]:
        return tuple(word.strip().casefold() for word in self.safewords_csv.split(",") if word.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
