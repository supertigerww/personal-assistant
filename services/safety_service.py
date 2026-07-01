from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.models import ConversationState, SafetyDecision, UserProfile

logger = logging.getLogger(__name__)


class SafetyService:
    LIMIT_PATTERNS = (
        r"(?:i do not like|i don't like|dont like)\s+(.+)",
        r"(?:i do not want|i don't want|dont want)\s+(.+)",
        r"(?:please avoid)\s+(.+)",
        r"(?:不喜欢|不要|别再|不想要|讨厌)\s*(.+)",
    )

    def __init__(self, *, settings: Any, user_service: Any, task_service: Any) -> None:
        self.settings = settings
        self.user_service = user_service
        self.task_service = task_service

    def detect_safeword(self, text: str) -> bool:
        normalized = self._normalize_safeword(text)
        matched = normalized in self._normalized_safewords()
        if matched:
            logger.warning("Safeword detected from inbound text.")
        return matched

    def extract_limits(self, text: str) -> list[str]:
        extracted: list[str] = []
        for pattern in self.LIMIT_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!?:;")
            if candidate and candidate not in extracted:
                extracted.append(candidate)
        if extracted:
            logger.info("Extracted explicit user limits: %s", extracted)
        return extracted

    async def handle_safeword(self, profile: UserProfile) -> SafetyDecision:
        reason = "safeword"
        logger.info(f"Safeword triggered for user {profile.telegram_user_id}, reason: {reason}")

        await self.task_service.pause_all_tasks(profile.telegram_user_id, reason=reason)
        aftercare_until = (
            datetime.now(timezone.utc)
            + timedelta(minutes=self.settings.aftercare_duration_minutes)
        ).isoformat()
        await self.user_service.update_state(
            profile.telegram_user_id,
            ConversationState.AFTERCARE,
            paused_reason=reason,
            aftercare_until=aftercare_until,
        )

        reply = (
            "安全词已接收。一切停止。\n"
            "现在深呼吸，慢慢来。\n"
            "告诉我你现在感觉如何？我在这里陪着你。"
        )

        logger.info(
            "User_id=%s moved into aftercare until=%s",
            profile.telegram_user_id,
            aftercare_until,
        )
        return SafetyDecision(
            triggered=True,
            reply=reply,
            state=ConversationState.AFTERCARE,
        )

    def _normalized_safewords(self) -> set[str]:
        return {
            self._normalize_safeword(word)
            for word in self.settings.safewords
            if self._normalize_safeword(word)
        }

    @staticmethod
    def _normalize_safeword(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())
