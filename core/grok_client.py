from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class GrokClient:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._validate_settings()
        self._max_retries = max(0, int(getattr(settings, "xai_max_retries", 2)))
        self._retry_delay_seconds = max(0.0, float(getattr(settings, "xai_retry_delay_seconds", 1.0)))
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            logger.info("Initializing xAI client with base_url=%s", self.settings.xai_base_url)
            self._client = AsyncOpenAI(
                api_key=self.settings.xai_api_key,
                base_url=self.settings.xai_base_url,
            )
        return self._client

    async def create_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        previous_response_id: str | None = None,
    ) -> Any:
        normalized_input = self._normalize_payload(input_items)
        request_tools = tools or []

        for attempt in range(1, self._max_retries + 2):
            try:
                logger.info(
                    "Calling xAI responses.create attempt=%s model=%s previous_response_id=%s input_items=%s tools=%s",
                    attempt,
                    self.settings.xai_model,
                    previous_response_id,
                    len(normalized_input),
                    len(request_tools),
                )
                response = await self.client.responses.create(
                    model=self.settings.xai_model,
                    input=normalized_input,
                    tools=request_tools,
                    previous_response_id=previous_response_id,
                    parallel_tool_calls=True,
                )
                logger.info(
                    "xAI responses.create succeeded attempt=%s response_id=%s",
                    attempt,
                    getattr(response, "id", None),
                )
                return response
            except Exception as exc:
                await self._handle_retryable_error(
                    operation="responses.create",
                    attempt=attempt,
                    error=exc,
                )

    async def generate_image(self, *, prompt: str, count: int = 1) -> list[str]:
        if not self.settings.enable_image_generation:
            logger.debug("Skipping image generation because it is disabled in settings.")
            return []

        normalized_prompt = self._normalize_text(prompt)
        if not normalized_prompt:
            logger.debug("Skipping image generation because prompt is empty after normalization.")
            return []

        requested_count = max(1, int(count))
        for attempt in range(1, self._max_retries + 2):
            try:
                logger.info(
                    "Calling xAI images.generate attempt=%s model=%s count=%s prompt_length=%s",
                    attempt,
                    self.settings.xai_image_model,
                    requested_count,
                    len(normalized_prompt),
                )
                response = await self.client.images.generate(
                    model=self.settings.xai_image_model,
                    prompt=normalized_prompt,
                    n=requested_count,
                )
                sources = await self._extract_generated_image_sources(response)
                logger.info(
                    "xAI images.generate succeeded attempt=%s generated_images=%s",
                    attempt,
                    len(sources),
                )
                return sources
            except Exception as exc:
                await self._handle_retryable_error(
                    operation="images.generate",
                    attempt=attempt,
                    error=exc,
                )

        return []

    async def describe_user_photo(self, *, image_path: str) -> str:
        if not getattr(self.settings, "enable_photo_vision", True):
            return ""

        path = Path(image_path)
        if not path.exists():
            logger.warning("Photo vision skipped because file does not exist: %s", image_path)
            return ""

        image_bytes = await asyncio.to_thread(path.read_bytes)
        if not image_bytes:
            return ""

        mime_type = self._mime_type_for_path(path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        prompt = (
            "用一两句中文客观描述这张用户提交的照片画面内容，"
            "用于后续对话上下文。只描述可见内容，不要客套。"
        )

        for attempt in range(1, self._max_retries + 2):
            try:
                response = await self.client.responses.create(
                    model=self.settings.xai_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:{mime_type};base64,{encoded}",
                                },
                            ],
                        }
                    ],
                )
                description = self.extract_text(response).strip()
                if description:
                    return description
                return ""
            except Exception as exc:
                await self._handle_retryable_error(
                    operation="responses.create.photo_vision",
                    attempt=attempt,
                    error=exc,
                )

        return ""

    async def generate_video_caption(
        self,
        *,
        video_category: str | None,
        response_text: str,
        user_text: str,
        state: str,
        recent_captions: list[str] | None = None,
    ) -> str:
        trimmed_response = self._normalize_text(response_text)[:280]
        trimmed_user = self._normalize_text(user_text)[:120]
        category_label = video_category or "unknown"
        history_block = self._format_recent_captions_for_prompt(recent_captions or [])

        for attempt in range(1, self._max_retries + 2):
            try:
                logger.info(
                    "Calling xAI responses.create for video caption attempt=%s category=%s",
                    attempt,
                    category_label,
                )
                response = await self.client.responses.create(
                    model=self.settings.xai_model,
                    input=[
                        {
                            "role": "system",
                            "content": (
                                "你是 Luna，强势 SM 女王。为即将单独发送的补充视频写一句极短中文 caption（12-28 字），"
                                "语气命令式、下流、羞辱，像真人聊天。不要重复用户或你上文里已经出现过的句子，"
                                "也不要重复近期已用过的视频 caption。每次必须换新说法。"
                                "只输出这一句，不要引号、不要解释。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"当前状态: {state}\n"
                                f"视频文件夹类型: {category_label}\n"
                                f"用户刚说: {trimmed_user or '（无）'}\n"
                                f"你上文: {trimmed_response or '（无）'}\n"
                                f"{history_block}\n"
                                "写视频下方 caption:"
                            ),
                        },
                    ],
                )
                caption = self.extract_text(response).strip()
                if caption:
                    return caption
                return ""
            except Exception as exc:
                await self._handle_retryable_error(
                    operation="responses.create.video_caption",
                    attempt=attempt,
                    error=exc,
                )

        return ""

    @staticmethod
    def _format_recent_captions_for_prompt(recent_captions: list[str]) -> str:
        cleaned = [item.strip() for item in recent_captions if item and item.strip()]
        if not cleaned:
            return "近期用过的视频 caption: 无"
        lines = "\n".join(f"- {item}" for item in cleaned[:8])
        return f"近期用过的视频 caption（禁止重复或仅换一两个词）:\n{lines}"

    @staticmethod
    def extract_text(response: Any) -> str:
        if getattr(response, "output_text", None):
            return response.output_text.strip()

        chunks: list[str] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    @staticmethod
    def extract_function_calls(response: Any) -> list[dict[str, str]]:
        calls: list[dict[str, str]] = []
        for item in getattr(response, "output", []):
            if getattr(item, "type", None) != "function_call":
                continue
            calls.append(
                {
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id,
                }
            )
        return calls

    def _validate_settings(self) -> None:
        missing_fields: list[str] = []
        if not getattr(self.settings, "xai_api_key", None):
            missing_fields.append("xai_api_key")
        if not getattr(self.settings, "xai_base_url", None):
            missing_fields.append("xai_base_url")
        if not getattr(self.settings, "xai_model", None):
            missing_fields.append("xai_model")
        if getattr(self.settings, "enable_image_generation", False) and not getattr(self.settings, "xai_image_model", None):
            missing_fields.append("xai_image_model")

        if missing_fields:
            raise RuntimeError(f"Missing required xAI settings: {', '.join(missing_fields)}")

    async def _extract_generated_image_sources(self, response: Any) -> list[str]:
        sources: list[str] = []
        for index, item in enumerate(getattr(response, "data", []) or [], start=1):
            source = await self._materialize_generated_image(item=item, index=index)
            if source:
                sources.append(source)
        return sources

    async def _materialize_generated_image(self, *, item: Any, index: int) -> str | None:
        b64_json = self._response_field(item, "b64_json")
        if isinstance(b64_json, str) and b64_json.strip():
            try:
                image_bytes = base64.b64decode(b64_json)
                saved_path = await self._save_generated_image(
                    image_bytes=image_bytes,
                    content_type="image/png",
                    source_name=f"inline-{index}.png",
                )
                logger.info("Saved inline generated image item=%s path=%s", index, saved_path)
                return saved_path
            except Exception as exc:
                logger.warning("Failed to decode inline generated image item=%s: %s", index, exc)

        url = self._response_field(item, "url")
        if isinstance(url, str) and url.strip():
            cleaned_url = url.strip()
            try:
                image_bytes, content_type = await self._download_generated_image(cleaned_url)
                saved_path = await self._save_generated_image(
                    image_bytes=image_bytes,
                    content_type=content_type,
                    source_name=cleaned_url,
                )
                logger.info("Downloaded generated image item=%s path=%s", index, saved_path)
                return saved_path
            except Exception as exc:
                logger.warning(
                    "Failed to download generated image item=%s url=%s: %s. Falling back to remote URL.",
                    index,
                    cleaned_url,
                    exc,
                )
                return cleaned_url

        logger.warning("Generated image item=%s did not include a supported payload.", index)
        return None

    async def _download_generated_image(self, url: str) -> tuple[bytes, str | None]:
        return await asyncio.to_thread(self._download_generated_image_sync, url)

    def _download_generated_image_sync(self, url: str) -> tuple[bytes, str | None]:
        request = Request(url, headers={"User-Agent": "queen-bot/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read()
            content_type = response.headers.get_content_type() if response.headers else None
        return data, content_type

    async def _save_generated_image(
        self,
        *,
        image_bytes: bytes,
        content_type: str | None,
        source_name: str,
    ) -> str:
        return await asyncio.to_thread(
            self._save_generated_image_sync,
            image_bytes,
            content_type,
            source_name,
        )

    def _save_generated_image_sync(self, image_bytes: bytes, content_type: str | None, source_name: str) -> str:
        output_dir = self._generated_images_dir()
        suffix = (
            self._suffix_from_content_type(content_type)
            or self._suffix_from_url(source_name)
            or self._guess_image_suffix(image_bytes)
            or ".png"
        )
        output_path = output_dir / f"grok_{uuid4().hex}{suffix}"
        output_path.write_bytes(image_bytes)
        return str(output_path)

    def _generated_images_dir(self) -> Path:
        path = Path(getattr(self.settings, "generated_images_path", "data/generated_images"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _handle_retryable_error(
        self,
        *,
        operation: str,
        attempt: int,
        error: Exception,
    ) -> None:
        retryable = self._is_retryable_error(error)
        if attempt > self._max_retries or not retryable:
            if retryable:
                logger.exception(
                    "xAI %s failed after %s attempt(s): %s",
                    operation,
                    attempt,
                    error,
                )
            else:
                logger.error(
                    "xAI %s failed with a non-retryable error on attempt %s: %s",
                    operation,
                    attempt,
                    error,
                    exc_info=error,
                )
            raise error

        logger.warning(
            "xAI %s failed on attempt %s/%s: %s. Retrying in %.2f seconds.",
            operation,
            attempt,
            self._max_retries + 1,
            error,
            self._retry_delay_seconds,
        )
        if self._retry_delay_seconds > 0:
            await asyncio.sleep(self._retry_delay_seconds)

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code in {408, 409, 429} or status_code >= 500
        return True

    def _normalize_payload(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._normalize_text(value)
        if isinstance(value, list):
            return [self._normalize_payload(item) for item in value]
        if isinstance(value, tuple):
            return [self._normalize_payload(item) for item in value]
        if isinstance(value, dict):
            return {key: self._normalize_payload(item) for key, item in value.items()}
        return value

    @staticmethod
    def _normalize_text(text: str) -> str:
        sanitized = text.replace("\x00", "").replace("\ufeff", "")
        return sanitized.encode("utf-8", errors="ignore").decode("utf-8")

    @staticmethod
    def _response_field(item: Any, field_name: str) -> Any:
        if isinstance(item, dict):
            return item.get(field_name)
        return getattr(item, field_name, None)

    @staticmethod
    def _mime_type_for_path(path: Path) -> str:
        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        return mapping.get(path.suffix.casefold(), "image/jpeg")

    @staticmethod
    def _suffix_from_content_type(content_type: str | None) -> str | None:
        mapping = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        if not content_type:
            return None
        return mapping.get(content_type.casefold())

    @staticmethod
    def _suffix_from_url(source_name: str) -> str | None:
        path = urlparse(source_name).path if "://" in source_name else source_name
        suffix = Path(path).suffix.casefold()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            return ".jpg" if suffix == ".jpeg" else suffix
        return None

    @staticmethod
    def _guess_image_suffix(image_bytes: bytes) -> str | None:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return ".gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return None
