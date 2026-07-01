from __future__ import annotations

import asyncio
import logging
from typing import Any

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
                urls = [item.url for item in response.data if getattr(item, "url", None)]
                logger.info(
                    "xAI images.generate succeeded attempt=%s generated_urls=%s",
                    attempt,
                    len(urls),
                )
                return urls
            except Exception as exc:
                await self._handle_retryable_error(
                    operation="images.generate",
                    attempt=attempt,
                    error=exc,
                )

        return []

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
