from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.models import ConversationState

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AssetCandidate:
    path: Path
    score: int
    filename_hits: int
    folder_hits: int


@dataclass(slots=True)
class AssetSearchOutcome:
    images: list[AssetCandidate]
    videos: list[AssetCandidate]
    keywords: list[str]

    @property
    def best_score(self) -> int:
        top_scores = []
        if self.images:
            top_scores.append(self.images[0].score)
        if self.videos:
            top_scores.append(self.videos[0].score)
        return max(top_scores, default=0)


class MediaService:
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
    SPECIAL_SCENE_MARKERS = (
        "specific",
        "special",
        "custom",
        "detailed",
        "close-up",
        "lighting",
        "outfit",
        "location",
        "scene",
        "angle",
        "特殊",
        "具体",
        "定制",
        "细节",
        "特写",
        "光线",
        "服装",
        "地点",
        "场景",
        "角度",
    )

    def __init__(self, *, settings: Any, grok_client: Any, user_service: Any | None = None) -> None:
        self.settings = settings
        self.grok_client = grok_client
        self.user_service = user_service
        self.images_path = Path(settings.assets_images_path)
        self.videos_path = Path(settings.assets_videos_path)
        self.images_path.mkdir(parents=True, exist_ok=True)
        self.videos_path.mkdir(parents=True, exist_ok=True)

    async def asset_summary(self) -> dict[str, int]:
        try:
            images = self._list_assets(self.images_path, self.IMAGE_SUFFIXES)
            videos = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
            return {"images": len(images), "videos": len(videos)}
        except Exception as exc:
            logger.exception("Failed to summarize local assets: %s", exc)
            return {"images": 0, "videos": 0}

    async def pick_relevant_assets(self, *, text: str) -> dict[str, list[str]]:
        try:
            outcome = self._search_assets(text)
            if self._has_scored_match(outcome):
                logger.debug("Resolved relevant local assets with keywords=%s", outcome.keywords)
                return self._outcome_to_payload(outcome)

            logger.debug("No direct local media match found; returning random assets.")
            return await self.get_random_assets()
        except Exception as exc:
            logger.exception("Failed to pick relevant assets: %s", exc)
            return {"images": [], "videos": []}

    async def get_or_generate_media(self, *, context: str, user_id: int) -> dict[str, list[str]]:
        try:
            outcome = self._search_assets(context)
            local_payload = self._outcome_to_payload(outcome)

            if self._is_good_match(outcome, context):
                logger.info("Using strong local media match for user_id=%s with score=%s", user_id, outcome.best_score)
                return local_payload

            if await self._should_generate(context, user_id):
                try:
                    images = await self.generate_scene_image(prompt=context)
                    if images:
                        logger.info("Generated supplemental images for user_id=%s", user_id)
                        return {"images": images, "videos": []}
                except Exception as exc:
                    logger.exception("生成图片失败: %s", exc)

            fallback = await self.get_random_assets()
            if fallback["images"] or fallback["videos"]:
                logger.info("Using random local media fallback for user_id=%s", user_id)
                return fallback

            logger.debug("No random fallback assets available; returning best-effort local payload.")
            return local_payload
        except Exception as exc:
            logger.exception("Failed to resolve media for user_id=%s: %s", user_id, exc)
            return {"images": [], "videos": []}

    async def get_random_assets(self, *, image_count: int = 1, video_count: int = 1) -> dict[str, list[str]]:
        try:
            image_candidates = self._list_assets(self.images_path, self.IMAGE_SUFFIXES)
            video_candidates = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
            selected_images = self._sample_assets(image_candidates, image_count)
            selected_videos = self._sample_assets(video_candidates, video_count)
            return {
                "images": [str(path) for path in selected_images],
                "videos": [str(path) for path in selected_videos],
            }
        except Exception as exc:
            logger.exception("Failed to load random assets: %s", exc)
            return {"images": [], "videos": []}

    async def generate_scene_image(self, *, prompt: str, count: int = 1) -> list[str]:
        prompt_text = prompt.strip()
        if not prompt_text:
            logger.debug("Skipping image generation because prompt is empty.")
            return []

        logger.info("Requesting generated image(s) with prompt length=%s", len(prompt_text))
        try:
            return await self.grok_client.generate_image(prompt=prompt_text, count=count)
        except Exception as exc:
            logger.exception("Image generation request failed: %s", exc)
            raise

    async def _should_generate(self, context: str, user_id: int) -> bool:
        if not self.settings.enable_image_generation:
            logger.debug("Image generation is disabled in settings.")
            return False

        profile = await self._safe_get_profile(user_id)
        if profile is not None and profile.state == ConversationState.AFTERCARE:
            logger.info("Skipping image generation for user_id=%s because state is aftercare.", user_id)
            return False

        normalized = context.strip().casefold()
        if not normalized:
            return False

        keywords = self._extract_keywords(context)
        if any(marker in normalized for marker in self.SPECIAL_SCENE_MARKERS):
            return True

        return len(keywords) >= 2 or len(normalized) >= 24

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        preserved = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_\s-]", " ", text)
        normalized = preserved.casefold()

        keywords: list[str] = []
        for raw_token in re.split(r"[\s_-]+", normalized):
            token = raw_token.strip()
            if not token:
                continue

            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                for candidate in MediaService._expand_chinese_token(token):
                    if candidate not in keywords:
                        keywords.append(candidate)
                continue

            if len(token) >= 2 and token not in keywords:
                keywords.append(token)

            if len(keywords) >= 16:
                break

        return keywords[:16]

    @staticmethod
    def _expand_chinese_token(token: str) -> list[str]:
        if len(token) <= 2:
            return [token]

        expanded: list[str] = [token]
        max_window = min(4, len(token))
        for window in range(max_window, 1, -1):
            for start in range(0, len(token) - window + 1):
                piece = token[start : start + window]
                if piece not in expanded:
                    expanded.append(piece)
                if len(expanded) >= 10:
                    return expanded
        return expanded

    def _search_assets(self, context: str) -> AssetSearchOutcome:
        keywords = self._extract_keywords(context)
        images = self._rank_assets(self.images_path, self.IMAGE_SUFFIXES, keywords)
        videos = self._rank_assets(self.videos_path, self.VIDEO_SUFFIXES, keywords)
        return AssetSearchOutcome(images=images, videos=videos, keywords=keywords)

    def _rank_assets(self, base_path: Path, suffixes: set[str], keywords: list[str]) -> list[AssetCandidate]:
        if not keywords:
            return []

        ranked: list[AssetCandidate] = []
        for path in self._list_assets(base_path, suffixes):
            candidate = self._score_asset(path, base_path, keywords)
            if candidate.score > 0:
                ranked.append(candidate)

        ranked.sort(
            key=lambda item: (
                item.score,
                item.folder_hits,
                item.filename_hits,
                item.path.as_posix().casefold(),
            ),
            reverse=True,
        )
        return ranked

    def _score_asset(self, path: Path, base_path: Path, keywords: list[str]) -> AssetCandidate:
        relative = path.relative_to(base_path)
        relative_text = relative.as_posix().casefold()
        filename_text = path.stem.casefold()
        filename_tokens = set(self._extract_keywords(path.stem))

        folder_parts = [part.casefold() for part in relative.parts[:-1]]
        folder_text = "/".join(folder_parts)
        folder_tokens: set[str] = set()
        for part in relative.parts[:-1]:
            folder_tokens.update(self._extract_keywords(part))

        score = 0
        filename_hits = 0
        folder_hits = 0

        for keyword in keywords:
            keyword_bonus = min(len(keyword), 4)

            if keyword in folder_tokens:
                score += 12 + keyword_bonus
                folder_hits += 1
            elif keyword in folder_text:
                score += 8 + keyword_bonus
                folder_hits += 1

            if keyword in filename_tokens:
                score += 10 + keyword_bonus
                filename_hits += 1
            elif keyword in filename_text:
                score += 7 + keyword_bonus
                filename_hits += 1
            elif keyword in relative_text:
                score += 3

        return AssetCandidate(
            path=path,
            score=score,
            filename_hits=filename_hits,
            folder_hits=folder_hits,
        )

    def _outcome_to_payload(self, outcome: AssetSearchOutcome) -> dict[str, list[str]]:
        return {
            "images": [str(candidate.path) for candidate in outcome.images[:1]],
            "videos": [str(candidate.path) for candidate in outcome.videos[:1]],
        }

    @staticmethod
    def _has_scored_match(outcome: AssetSearchOutcome) -> bool:
        return outcome.best_score > 0

    def _is_good_match(self, outcome: AssetSearchOutcome, context: str) -> bool:
        keywords = outcome.keywords or self._extract_keywords(context)
        if not keywords:
            return False

        if outcome.best_score >= 16:
            return True
        if outcome.best_score >= 12 and len(keywords) <= 4:
            return True
        return outcome.best_score >= 10 and any(len(keyword) >= 4 for keyword in keywords)

    async def _safe_get_profile(self, user_id: int) -> Any | None:
        if self.user_service is None:
            return None

        try:
            return await self.user_service.get_profile(user_id)
        except Exception as exc:
            logger.warning("Could not load user profile for media decision user_id=%s: %s", user_id, exc)
            return None

    @staticmethod
    def _list_assets(base_path: Path, suffixes: set[str]) -> list[Path]:
        return [path for path in base_path.rglob("*") if path.is_file() and path.suffix.lower() in suffixes]

    @staticmethod
    def _sample_assets(paths: list[Path], count: int) -> list[Path]:
        if not paths or count <= 0:
            return []
        if len(paths) <= count:
            return paths
        return random.sample(paths, count)
