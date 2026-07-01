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
        top_scores: list[int] = []
        if self.images:
            top_scores.append(self.images[0].score)
        if self.videos:
            top_scores.append(self.videos[0].score)
        return max(top_scores, default=0)


class MediaService:
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
    SPECIAL_SCENE_MARKERS = (
        "特写", "详细", "具体", "定制", "姿势", "角度",
        "让我看看", "发给我看", "看着这个", "对比", "幻想", "想象",
         "寸止", "撸管", "自慰", "羞辱",
        "母猪", "尿壶", "脚奴", "绿帽", "公开", "验证",
        "close-up", "detailed", "specific", "pose", "angle", "show me",
        "imagine", "kneel", "edging", "jerk", "humiliate", "degrade",
        "cuckold", "public", "verification", "comparison",
    )

    def __init__(self, *, settings: Any, grok_client: Any, user_service: Any | None = None) -> None:
        self.settings = settings
        self.grok_client = grok_client
        self.user_service = user_service
        self.images_path = Path(settings.assets_images_path)
        self.videos_path = Path(settings.assets_videos_path)
        self._prepare_media_root(self.images_path, label="images")
        self._prepare_media_root(self.videos_path, label="videos")
        logger.info(
            "MediaService configured for recursive asset scan. images_path=%s videos_path=%s",
            self.images_path,
            self.videos_path,
        )

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
            profile = await self._safe_get_profile(user_id)
            outcome = self._search_assets(context)

            if not self._should_attach_media(
                context=context,
                user_id=user_id,
                profile=profile,
                outcome=outcome,
            ):
                logger.info("Skipped automatic media for user_id=%s after probability gate.", user_id)
                return {"images": [], "videos": []}

            local_payload = self._outcome_to_payload(outcome)
            if self._is_good_match(outcome, context) and (local_payload["images"] or local_payload["videos"]):
                logger.info(
                    "Using strong local media match for user_id=%s with score=%s",
                    user_id,
                    outcome.best_score,
                )
                return local_payload

            if await self._should_generate(context, user_id, profile=profile):
                try:
                    images = await self.generate_scene_image(prompt=context)
                    if images:
                        logger.info("Generated supplemental images for user_id=%s", user_id)
                        return self._compact_payload(images=images, videos=[], prefer_random=False)
                except Exception as exc:
                    logger.exception(
                        "Failed to generate supplemental image for user_id=%s: %s",
                        user_id,
                        exc,
                    )

            if self._should_use_random_fallback(user_id=user_id, profile=profile):
                fallback = await self.get_random_assets()
                if fallback["images"] or fallback["videos"]:
                    logger.info("Using random local media fallback for user_id=%s", user_id)
                    return fallback

            logger.debug("No automatic media selected for user_id=%s after fallback checks.", user_id)
            return {"images": [], "videos": []}
        except Exception as exc:
            logger.exception("Failed to resolve media for user_id=%s: %s", user_id, exc)
            return {"images": [], "videos": []}

    async def get_random_assets(self, *, image_count: int = 1, video_count: int = 1) -> dict[str, list[str]]:
        try:
            image_candidates = self._list_assets(self.images_path, self.IMAGE_SUFFIXES)
            video_candidates = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
            selected_images = [str(path) for path in self._sample_assets(image_candidates, image_count)]
            selected_videos = [str(path) for path in self._sample_assets(video_candidates, video_count)]
            return self._compact_payload(images=selected_images, videos=selected_videos, prefer_random=True)
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

    async def _should_generate(
        self,
        context: str,
        user_id: int,
        *,
        profile: Any | None = None,
    ) -> bool:
        if not self.settings.enable_image_generation:
            logger.debug("Image generation is disabled in settings.")
            return False

        if profile is None:
            profile = await self._safe_get_profile(user_id)
        if profile is not None and profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            logger.info("Skipping image generation for user_id=%s because state=%s.", user_id, profile.state)
            return False

        normalized = context.strip().casefold()
        if not normalized:
            return False

        if self._has_special_scene_marker(normalized):
            return True

        keywords = self._extract_keywords(context)
        return len(keywords) >= 3 or len(normalized) >= 30

    def _should_attach_media(
        self,
        *,
        context: str,
        user_id: int,
        profile: Any | None,
        outcome: AssetSearchOutcome,
    ) -> bool:
        chance = self._media_send_probability(profile)
        if chance <= 0:
            logger.debug("Automatic media disabled by state for user_id=%s.", user_id)
            return False

        normalized = context.strip().casefold()
        if self._is_good_match(outcome, context):
            chance = min(1.0, chance + 0.15)
        if self._has_special_scene_marker(normalized):
            chance = min(1.0, chance + 0.10)

        roll = random.random()
        decision = roll < chance
        logger.debug(
            "Automatic media gate user_id=%s state=%s chance=%.2f roll=%.2f best_score=%s decision=%s",
            user_id,
            getattr(profile, "state", ConversationState.NORMAL),
            chance,
            roll,
            outcome.best_score,
            decision,
        )
        return decision

    def _should_use_random_fallback(self, *, user_id: int, profile: Any | None) -> bool:
        chance = self._clamp_probability(getattr(self.settings, "media_random_fallback_probability", 0.25))
        if chance <= 0:
            return False

        roll = random.random()
        decision = roll < chance
        logger.debug(
            "Random fallback media gate user_id=%s state=%s chance=%.2f roll=%.2f decision=%s",
            user_id,
            getattr(profile, "state", ConversationState.NORMAL),
            chance,
            roll,
            decision,
        )
        return decision

    def _media_send_probability(self, profile: Any | None) -> float:
        state = getattr(profile, "state", ConversationState.NORMAL)
        if state == ConversationState.INTENSE:
            return self._clamp_probability(getattr(self.settings, "media_send_probability_intense", 0.55))
        if state == ConversationState.AFTERCARE:
            return self._clamp_probability(getattr(self.settings, "media_send_probability_aftercare", 0.0))
        if state == ConversationState.PAUSED:
            return self._clamp_probability(getattr(self.settings, "media_send_probability_paused", 0.0))
        return self._clamp_probability(getattr(self.settings, "media_send_probability_normal", 0.35))

    @staticmethod
    def _clamp_probability(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

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
        ranked: list[tuple[str, AssetCandidate]] = [("image", item) for item in outcome.images]
        ranked.extend(("video", item) for item in outcome.videos)
        if not ranked:
            return {"images": [], "videos": []}

        ranked.sort(key=lambda item: item[1].score, reverse=True)
        limit = self._max_media_items_per_message()
        if limit <= 0:
            return {"images": [], "videos": []}

        selected: list[tuple[str, AssetCandidate]]
        if limit == 1:
            best_score = ranked[0][1].score
            top_choices = [item for item in ranked if item[1].score == best_score]
            selected = [random.choice(top_choices)]
        else:
            selected = ranked[:limit]

        payload = {"images": [], "videos": []}
        for kind, candidate in selected:
            payload["images" if kind == "image" else "videos"].append(str(candidate.path))
        return payload

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
    def _prepare_media_root(path: Path, *, label: str) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning("Could not ensure %s media path %s: %s", label, path, exc)

        if path.exists() and not path.is_dir():
            logger.warning("Configured %s media path is not a directory: %s", label, path)
        elif not path.exists():
            logger.warning("Configured %s media path does not exist: %s", label, path)

    def _list_assets(self, base_path: Path, suffixes: set[str]) -> list[Path]:
        if not base_path.exists():
            logger.warning("Media path does not exist, skipping scan: %s", base_path)
            return []
        if not base_path.is_dir():
            logger.warning("Media path is not a directory, skipping scan: %s", base_path)
            return []

        max_bytes = self._max_asset_bytes_for_suffixes(suffixes)
        assets: list[Path] = []
        skipped_oversized = 0

        for path in base_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes:
                continue

            if max_bytes is not None:
                try:
                    size_bytes = path.stat().st_size
                except OSError as exc:
                    logger.warning("Could not read media size for path=%s: %s", path, exc)
                    continue

                if size_bytes > max_bytes:
                    skipped_oversized += 1
                    logger.debug(
                        "Skipping oversized media file path=%s size_mb=%.2f limit_mb=%s",
                        path,
                        size_bytes / (1024 * 1024),
                        getattr(self.settings, "max_local_video_size_mb", 45),
                    )
                    continue

            assets.append(path)

        if skipped_oversized:
            logger.info("Skipped %s oversized media file(s) under %s", skipped_oversized, base_path)
        return assets

    def _compact_payload(
        self,
        *,
        images: list[str],
        videos: list[str],
        prefer_random: bool,
    ) -> dict[str, list[str]]:
        limit = self._max_media_items_per_message()
        if limit <= 0:
            return {"images": [], "videos": []}

        combined: list[tuple[str, str]] = [("image", path) for path in images]
        combined.extend(("video", path) for path in videos)
        if len(combined) <= limit:
            return {"images": images[:], "videos": videos[:]}

        if prefer_random:
            selected = random.sample(combined, limit)
        else:
            selected = combined[:limit]

        payload = {"images": [], "videos": []}
        for kind, path in selected:
            payload["images" if kind == "image" else "videos"].append(path)
        return payload

    def _max_media_items_per_message(self) -> int:
        return max(0, int(getattr(self.settings, "media_max_items_per_message", 1)))

    @staticmethod
    def _has_special_scene_marker(normalized_context: str) -> bool:
        return any(marker in normalized_context for marker in MediaService.SPECIAL_SCENE_MARKERS)

    def _max_asset_bytes_for_suffixes(self, suffixes: set[str]) -> int | None:
        if suffixes != self.VIDEO_SUFFIXES:
            return None
        return self._video_size_limit_bytes()

    def _video_size_limit_bytes(self) -> int | None:
        limit_mb = int(getattr(self.settings, "max_local_video_size_mb", 45) or 0)
        if limit_mb <= 0:
            return None
        return limit_mb * 1024 * 1024

    @staticmethod
    def _sample_assets(paths: list[Path], count: int) -> list[Path]:
        if not paths or count <= 0:
            return []
        if len(paths) <= count:
            return paths
        return random.sample(paths, count)