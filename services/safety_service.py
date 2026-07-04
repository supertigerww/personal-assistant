from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from core.models import ConversationState, SafewordLevel, SafetyDecision, UserProfile

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

    def classify_safeword(self, text: str) -> SafewordLevel | None:
        normalized = self._normalize_safeword(text)
        if not normalized:
            return None

        if normalized in self._normalized_words(self.settings.red_safewords):
            logger.warning("Red safeword detected from inbound text.")
            return SafewordLevel.RED
        if normalized in self._normalized_words(self.settings.yellow_safewords):
            logger.warning("Yellow safeword detected from inbound text.")
            return SafewordLevel.YELLOW
        return None

    def detect_safeword(self, text: str) -> bool:
        return self.classify_safeword(text) is not None

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

    async def handle_safeword(self, profile: UserProfile, *, level: SafewordLevel) -> SafetyDecision:
        if level == SafewordLevel.YELLOW:
            return await self._handle_yellow_safeword(profile)
        return await self._handle_red_safeword(profile)

    async def _handle_red_safeword(self, profile: UserProfile) -> SafetyDecision:
        reason = "safeword_red"
        logger.info("Red safeword triggered for user %s", profile.telegram_user_id)

        await self.task_service.pause_all_tasks(profile.telegram_user_id, reason=reason)
        aftercare_until = (
            datetime.now(timezone.utc) + timedelta(minutes=self.settings.aftercare_duration_minutes)
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
            safeword_level=SafewordLevel.RED,
        )

    async def _handle_yellow_safeword(self, profile: UserProfile) -> SafetyDecision:
        reason = "safeword_yellow"
        logger.info("Yellow safeword triggered for user %s state=%s", profile.telegram_user_id, profile.state)

        if profile.state == ConversationState.AFTERCARE:
            reply = (
                "黄色安全词收到。我们继续保持温柔节奏。\n"
                "不用着急，告诉我你现在感觉如何？"
            )
            return SafetyDecision(
                triggered=True,
                reply=reply,
                state=ConversationState.AFTERCARE,
                safeword_level=SafewordLevel.YELLOW,
            )

        if profile.state == ConversationState.PAUSED:
            reply = (
                "黄色安全词收到。会话仍然暂停中。\n"
                "准备好了就告诉我，我们继续慢慢来。"
            )
            return SafetyDecision(
                triggered=True,
                reply=reply,
                state=ConversationState.PAUSED,
                safeword_level=SafewordLevel.YELLOW,
            )

        target_state = profile.state
        if profile.state == ConversationState.INTENSE:
            target_state = ConversationState.NORMAL
            await self.user_service.update_state(
                profile.telegram_user_id,
                target_state,
                paused_reason=reason,
                aftercare_until=None,
            )

        reply = (
            "黄色安全词收到。我们慢一点。\n"
            "强度已经降下来了。告诉我你现在感觉如何？"
        )
        return SafetyDecision(
            triggered=True,
            reply=reply,
            state=target_state,
            safeword_level=SafewordLevel.YELLOW,
        )

    def _normalized_words(self, words: tuple[str, ...]) -> set[str]:
        return {self._normalize_safeword(word) for word in words if self._normalize_safeword(word)}

    @staticmethod
    def _normalize_safeword(text: str) -> str:
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())