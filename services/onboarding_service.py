from __future__ import annotations

import logging
import re
from typing import Any

from core.models import UserProfile

logger = logging.getLogger(__name__)


class OnboardingService:
    NICKNAME_PATTERNS: tuple[str, ...] = (
        r"叫我\s*([^。；;!\n]+)",
        r"称呼我\s*([^。；;!\n]+)",
        r"称呼[：:\s]*([^。；;!\n]+)",
    )
    HARD_LIMIT_PATTERNS: tuple[str, ...] = (
        r"硬限[：:\s]*([^。；;!\n]+)",
        r"绝对不要[：:\s]*([^。；;!\n]+)",
        r"不要碰[：:\s]*([^。；;!\n]+)",
        r"红线[：:\s]*([^。；;!\n]+)",
    )
    INTENSITY_PATTERNS: tuple[str, ...] = (
        r"强度[：:\s]*(轻|中|重|normal|intense)",
        r"想要?(轻|中|重)度",
    )

    def __init__(self, *, user_service: Any, memory_service: Any | None = None) -> None:
        self.user_service = user_service
        self.memory_service = memory_service

    async def complete_from_user_text(self, telegram_user_id: int, user_text: str) -> UserProfile:
        profile = await self.user_service.get_profile(telegram_user_id)
        if profile.onboarding_completed:
            return profile

        parsed = self.parse_setup_text(user_text)
        nickname = parsed.get("nickname")
        if nickname:
            await self.user_service.append_notes(telegram_user_id, [f"称呼偏好：{nickname}"])

        hard_limits = parsed.get("hard_limits") or []
        if hard_limits:
            await self.user_service.append_hard_limits(telegram_user_id, hard_limits)

        intensity = parsed.get("intensity")
        if intensity:
            await self.user_service.append_notes(telegram_user_id, [f"强度偏好：{intensity}"])

        await self.user_service.append_notes(telegram_user_id, [f"入门回复：{user_text.strip()}"])

        if self.memory_service is not None:
            await self.memory_service.ingest_profile_updates(
                telegram_user_id,
                hard_limits=hard_limits,
                notes=[note for note in [f"称呼偏好：{nickname}" if nickname else None, f"强度偏好：{intensity}" if intensity else None] if note],
            )
            await self.memory_service.ingest_user_turn(telegram_user_id, user_text)

        await self.user_service.mark_onboarding_completed(telegram_user_id)
        logger.info("Completed onboarding for user_id=%s", telegram_user_id)
        return await self.user_service.get_profile(telegram_user_id)

    @classmethod
    def parse_setup_text(cls, user_text: str) -> dict[str, Any]:
        text = user_text.strip()
        result: dict[str, Any] = {"nickname": None, "hard_limits": [], "intensity": None}
        if not text:
            return result

        for pattern in cls.NICKNAME_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .,!?:;，。！？")
                if candidate:
                    result["nickname"] = candidate
                    break

        hard_limits: list[str] = []
        for pattern in cls.HARD_LIMIT_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!?:;，。！？")
            if candidate and candidate not in hard_limits:
                hard_limits.append(candidate)
        result["hard_limits"] = hard_limits

        for pattern in cls.INTENSITY_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1).casefold()
            result["intensity"] = cls._normalize_intensity(raw)
            break

        return result

    @staticmethod
    def _normalize_intensity(raw: str) -> str:
        mapping = {
            "轻": "轻",
            "中": "中",
            "重": "重",
            "normal": "中",
            "intense": "重",
        }
        return mapping.get(raw, raw)