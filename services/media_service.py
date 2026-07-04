from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.luna_visual import build_scene_image_prompt, load_visual_anchor
from core.media_categories import VideoCategoryIndex
from core.media_intent import (
    build_turn_hints,
    has_explicit_video_request,
    resolve_video_attachment,
)
from core.models import ConversationState, UserProfile
from services.media_delivery_service import MediaDeliveryService

logger = logging.getLogger(__name__)

POV_VIDEO_CAPTION_FALLBACKS = (
    "手别停，盯紧屏幕，漏一眼有你好看。",
    "跟着节奏撸，我说停才准停。",
    "寸止憋住，敢射出来试试。",
    "越羞耻越要看完，别躲。",
    "眼睛和手都给我老实点。",
)

SM_VIDEO_CAPTION_FALLBACKS = (
    "跪直了，一字不漏看完。",
    "看女王怎么收拾贱狗，学着点。",
    "记清楚每个细节，下次轮到你。",
    "这是示范，别眨眼。",
    "看明白了再说话。",
)


@dataclass(slots=True)
class MediaBundle:
    images: list[str]
    videos: list[str]
    video_caption: str | None = None
    text_before_video: bool = False
    suggested_foreshadow: str | None = None  # Rich creative setup text for the Queen to adapt/use


@dataclass(slots=True)
class AssetCandidate:
    path: Path
    score: int
    filename_hits: int
    folder_hits: int
    tag_hits: int = 0


@dataclass(slots=True)
class AssetSearchOutcome:
    images: list[AssetCandidate]
    videos: list[AssetCandidate]
    keywords: list[str]
    matched_categories: list[str] = field(default_factory=list)
    category_index: VideoCategoryIndex | None = None

    @property
    def best_score(self) -> int:
        top_scores: list[int] = []
        if self.images:
            top_scores.append(self.images[0].score)
        if self.videos:
            top_scores.append(self.videos[0].score)
        return max(top_scores, default=0)

    @property
    def best_video_score(self) -> int:
        return self.videos[0].score if self.videos else 0


class MediaService:
    IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
    EXPLICIT_MEDIA_MARKERS = (
        "\u56fe",
        "\u56fe\u7247",
        "\u7167\u7247",
        "\u6765\u5f20\u56fe",
        "\u53d1\u5f20\u56fe",
        "\u53d1\u56fe",
        "\u6765\u5f20",
        "image",
        "images",
        "photo",
        "picture",
        "pic",
    )
    SPECIAL_SCENE_MARKERS = (
        "\u7279\u5199",
        "\u7ec6\u8282",
        "\u8be6\u7ec6",
        "\u5177\u4f53",
        "\u5b9a\u5236",
        "\u59ff\u52bf",
        "\u89d2\u5ea6",
        "\u706f\u5149",
        "\u670d\u88c5",
        "\u573a\u666f",
        "\u5730\u70b9",
        "\u80cc\u666f",
        "\u955c\u5934",
        "close-up",
        "close up",
        "detailed",
        "detail",
        "specific",
        "custom",
        "pose",
        "angle",
        "lighting",
        "outfit",
        "location",
        "background",
        "scene",
        "shot",
    )

    def __init__(
        self,
        *,
        settings: Any,
        grok_client: Any,
        user_service: Any | None = None,
        database: Any | None = None,
    ) -> None:
        self.settings = settings
        self.grok_client = grok_client
        self.user_service = user_service
        self.delivery_service = MediaDeliveryService(database=database, settings=settings)
        self._visual_anchor = load_visual_anchor(settings)
        self.images_path = Path(settings.assets_images_path)
        self.videos_path = Path(settings.assets_videos_path)
        self._prepare_media_root(self.images_path, label="images")
        self._prepare_media_root(self.videos_path, label="videos")
        logger.info(
            "MediaService configured for recursive asset scan. images_path=%s videos_path=%s image_generation_enabled=%s generated_images_path=%s",
            self.images_path,
            self.videos_path,
            bool(getattr(self.settings, "enable_image_generation", False)),
            getattr(self.settings, "generated_images_path", "data/generated_images"),
        )

    async def asset_summary(self) -> dict[str, int | dict[str, int]]:
        try:
            images = self._list_assets(self.images_path, self.IMAGE_SUFFIXES)
            videos = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
            category_index = self._build_video_category_index(videos)
            return {
                "images": len(images),
                "videos": len(videos),
                "video_categories": category_index.folder_counts,
            }
        except Exception as exc:
            logger.exception("Failed to summarize local assets: %s", exc)
            return {"images": 0, "videos": 0, "video_categories": {}}

    def video_categories_context(self, *, media_summary: dict[str, int | dict[str, int]] | None = None) -> str:
        if media_summary is not None:
            raw_categories = media_summary.get("video_categories", {})
            if isinstance(raw_categories, dict) and raw_categories:
                category_index = VideoCategoryIndex(
                    folder_counts={str(key).casefold(): int(value) for key, value in raw_categories.items()},
                    aliases={
                        key.casefold(): tuple(alias.casefold() for alias in value)
                        for key, value in VideoCategoryIndex._parse_aliases_csv(
                            str(getattr(self.settings, "video_folder_aliases_csv", "")),
                        ).items()
                    },
                )
                return category_index.format_for_context()
        videos = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
        return self._build_video_category_index(videos).format_for_context()

    def _build_video_category_index(self, video_paths: list[Path]) -> VideoCategoryIndex:
        return VideoCategoryIndex.build(
            videos_path=self.videos_path,
            video_paths=video_paths,
            aliases_csv=str(getattr(self.settings, "video_folder_aliases_csv", "")),
        )

    async def record_deliveries(self, telegram_user_id: int, asset_paths: list[str]) -> None:
        await self.delivery_service.record_deliveries(telegram_user_id, asset_paths)

    async def pick_relevant_assets(
        self,
        *,
        text: str,
        user_id: int | None = None,
    ) -> dict[str, list[str]]:
        try:
            recent_paths = await self._recent_paths_for_user(user_id)
            outcome = await self._search_assets(text, user_id=user_id)
            if self._has_scored_match(outcome):
                logger.debug("Resolved relevant local assets with keywords=%s", outcome.keywords)
                return self._outcome_to_payload(
                    outcome,
                    recent_paths=recent_paths,
                    media_kind="image",
                )

            logger.debug("No direct local media match found; returning random assets.")
            return await self.get_random_assets(image_count=1, video_count=0, user_id=user_id)
        except Exception as exc:
            logger.exception("Failed to pick relevant assets: %s", exc)
            return {"images": [], "videos": []}

    async def evaluate_video_window(self, *, profile: UserProfile) -> tuple[UserProfile, bool]:
        if self.user_service is None:
            return profile, False
        if profile.state in {ConversationState.AFTERCARE, ConversationState.PAUSED}:
            return profile, False
        if profile.conversation_count < profile.next_video_turn:
            return profile, False

        chance = self._video_offer_probability(profile.state)
        roll = random.random()
        if roll < chance:
            logger.info(
                "Video window opened for user %s at turn=%s chance=%.2f roll=%.2f",
                profile.telegram_user_id,
                profile.conversation_count,
                chance,
                roll,
            )
            return profile, True

        retry_interval = self._pick_video_retry_interval(profile.state)
        next_turn = profile.conversation_count + retry_interval
        await self.user_service.update_next_video_turn(profile.telegram_user_id, next_turn)
        logger.info(
            "Deferred video window for user %s from turn=%s to next_video_turn=%s chance=%.2f roll=%.2f",
            profile.telegram_user_id,
            profile.conversation_count,
            next_turn,
            chance,
            roll,
        )
        return await self.user_service.get_profile(profile.telegram_user_id), False

    async def schedule_next_video_turn(self, telegram_user_id: int, state: ConversationState) -> int:
        if self.user_service is None:
            return 0

        interval = self._pick_video_interval(state)
        profile = await self.user_service.get_profile(telegram_user_id)
        next_turn = profile.conversation_count + interval
        await self.user_service.update_next_video_turn(telegram_user_id, next_turn)
        logger.info(
            "Scheduled next video window for user_id=%s at turn=%s interval=%s",
            telegram_user_id,
            next_turn,
            interval,
        )
        return next_turn

    def analyze_turn_hints(self, *, user_text: str, video_window_ready: bool):
        return build_turn_hints(user_text=user_text, video_window_ready=video_window_ready)

    async def get_or_generate_media(
        self,
        *,
        user_text: str,
        response_text: str,
        user_id: int,
        video_window_ready: bool = False,
    ) -> MediaBundle:
        empty = MediaBundle(images=[], videos=[])
        try:
            profile = await self._safe_get_profile(user_id)
            media_context = self._compose_media_context(user_text=user_text, response_text=response_text)
            recent_paths = await self._recent_paths_for_user(user_id)
            outcome = await self._search_assets(media_context, user_id=user_id)

            should_attach_video = resolve_video_attachment(
                user_text=user_text,
                response_text=response_text,
                video_window_ready=video_window_ready,
            )
            image_payload = self._outcome_to_payload(
                outcome,
                recent_paths=recent_paths,
                media_kind="image",
            )
            video_payload = (
                self._outcome_to_payload(
                    outcome,
                    recent_paths=recent_paths,
                    media_kind="video",
                )
                if should_attach_video
                else {"images": [], "videos": []}
            )

            normalized = media_context.strip().casefold()
            has_explicit_image_request = self._has_explicit_media_request(normalized) and not has_explicit_video_request(
                media_context
            )
            strong_image_match = self._is_good_match(outcome, media_context) and image_payload["images"]
            strong_video_match = should_attach_video and self._is_good_video_match(outcome, media_context)
            should_generate = await self._should_generate(media_context, user_id, profile=profile)

            logger.info(
                "Media decision user_id=%s attach_video=%s explicit_image=%s strong_image=%s strong_video=%s should_generate=%s best_score=%s",
                user_id,
                should_attach_video,
                has_explicit_image_request,
                bool(strong_image_match),
                bool(strong_video_match),
                should_generate,
                outcome.best_score,
            )

            if should_attach_video and strong_video_match:
                profile_hint = await self._safe_get_profile(user_id) if user_id else None
                return self._bundle_from_payload(
                    video_payload,
                    text_before_video=True,
                    add_foreshadow=True,
                    profile_for_foreshadow=profile_hint,
                    context_hint=media_context[:60],
                )

            if should_attach_video and has_explicit_video_request(user_text):
                explicit_video = await self._resolve_explicit_video(user_id=user_id, outcome=outcome)
                if explicit_video.videos:
                    return explicit_video

            if has_explicit_image_request and strong_image_match:
                logger.info("Using local image for explicit image request user_id=%s", user_id)
                return self._bundle_from_payload(image_payload)

            if should_generate:
                try:
                    images = await self.generate_scene_image(prompt=media_context)
                    if images:
                        logger.info(
                            "Generated supplemental images for user_id=%s reason=%s",
                            user_id,
                            "explicit_image_request" if has_explicit_image_request else "scene_decision",
                        )
                        compact = self._compact_payload(images=images, videos=[], prefer_random=False)
                        return self._bundle_from_payload(compact)
                except Exception as exc:
                    logger.exception(
                        "Failed to generate supplemental image for user_id=%s: %s",
                        user_id,
                        exc,
                    )

            if strong_image_match:
                if self._should_attach_media(
                    context=media_context,
                    user_id=user_id,
                    profile=profile,
                    outcome=outcome,
                ):
                    logger.info(
                        "Using strong local image match for user_id=%s with score=%s",
                        user_id,
                        outcome.best_score,
                    )
                    return self._bundle_from_payload(image_payload)

                logger.info("Skipped strong local image for user_id=%s after probability gate.", user_id)
                return empty

            if has_explicit_image_request:
                fallback = await self.get_random_assets(image_count=1, video_count=0, user_id=user_id)
                if fallback["images"]:
                    logger.info("Using random image fallback for explicit image request user_id=%s", user_id)
                    return self._bundle_from_payload(fallback)
                logger.info("No image available to satisfy explicit image request user_id=%s", user_id)
                return empty

            if not self._should_attach_media(
                context=media_context,
                user_id=user_id,
                profile=profile,
                outcome=outcome,
            ):
                logger.info("Skipped automatic media for user_id=%s after probability gate.", user_id)
                return empty

            if self._should_use_random_fallback(
                user_id=user_id,
                profile=profile,
                context=media_context,
                outcome=outcome,
            ):
                fallback = await self.get_random_assets(image_count=1, video_count=0, user_id=user_id)
                if fallback["images"]:
                    logger.info("Using random local image fallback for user_id=%s", user_id)
                    return self._bundle_from_payload(fallback)

            logger.debug("No automatic media selected for user_id=%s after fallback checks.", user_id)
            return empty
        except Exception as exc:
            logger.exception("Failed to resolve media for user_id=%s: %s", user_id, exc)
            return empty

    async def get_random_assets(
        self,
        *,
        image_count: int = 1,
        video_count: int = 1,
        user_id: int | None = None,
    ) -> dict[str, list[str]]:
        try:
            recent_paths = await self._recent_paths_for_user(user_id)
            image_candidates = self._prefer_fresh_assets(
                self._list_assets(self.images_path, self.IMAGE_SUFFIXES),
                recent_paths,
            )
            video_candidates = self._prefer_fresh_assets(
                self._list_assets(self.videos_path, self.VIDEO_SUFFIXES),
                recent_paths,
            )
            selected_images = [str(path) for path in self._sample_assets(image_candidates, image_count)]
            selected_videos = [str(path) for path in self._sample_assets(video_candidates, video_count)]
            return self._compact_payload(images=selected_images, videos=selected_videos, prefer_random=True)
        except Exception as exc:
            logger.exception("Failed to load random assets: %s", exc)
            return {"images": [], "videos": []}

    async def _resolve_explicit_video(
        self,
        *,
        user_id: int,
        outcome: AssetSearchOutcome,
    ) -> MediaBundle:
        recent_paths = await self._recent_paths_for_user(user_id)
        if outcome.videos:
            payload = self._outcome_to_payload(
                outcome,
                recent_paths=recent_paths,
                media_kind="video",
            )
            if payload["videos"]:
                profile_hint = await self._safe_get_profile(user_id) if user_id else None
                return self._bundle_from_payload(
                    payload, text_before_video=True, add_foreshadow=True, profile_for_foreshadow=profile_hint
                )

        fallback = await self.get_random_assets(image_count=0, video_count=1, user_id=user_id)
        if fallback["videos"]:
            profile_hint = await self._safe_get_profile(user_id) if user_id else None
            return self._bundle_from_payload(
                fallback, text_before_video=True, add_foreshadow=True, profile_for_foreshadow=profile_hint
            )
        return MediaBundle(images=[], videos=[])

    @staticmethod
    def _compose_media_context(*, user_text: str, response_text: str) -> str:
        parts: list[str] = []
        cleaned_user = user_text.strip()
        cleaned_response = response_text.strip()
        if cleaned_user:
            parts.append(cleaned_user)
        if cleaned_response:
            parts.append(cleaned_response)
        return " ".join(parts).strip()

    def _bundle_from_payload(
        self,
        payload: dict[str, list[str]],
        *,
        text_before_video: bool = False,
        add_foreshadow: bool = False,
        profile_for_foreshadow: Any = None,
        context_hint: str = "",
    ) -> MediaBundle:
        videos = payload.get("videos", [])[:]
        bundle = MediaBundle(
            images=payload.get("images", [])[:],
            videos=videos,
            video_caption=None,
            text_before_video=text_before_video and bool(videos),
        )
        if add_foreshadow and videos:
            comp = getattr(profile_for_foreshadow, "compliance_score", 5) if profile_for_foreshadow else 5
            st = str(getattr(profile_for_foreshadow, "state", "normal")) if profile_for_foreshadow else "normal"
            bundle.suggested_foreshadow = self.build_creative_video_foreshadow(
                video_path=videos[0],
                category=None,
                state=st,
                compliance_score=comp,
                context_hint=context_hint,
            )
        return bundle

    async def generate_video_caption(
        self,
        *,
        video_path: str,
        user_text: str,
        response_text: str,
        profile: UserProfile | None = None,
        recent_captions: list[str] | None = None,
    ) -> str:
        category = self._video_category_for_path(video_path)
        state = str(getattr(profile, "state", ConversationState.NORMAL))
        history = [item.strip() for item in (recent_captions or []) if item and item.strip()]

        if getattr(self.settings, "enable_llm_video_caption", True):
            try:
                caption = await self.grok_client.generate_video_caption(
                    video_category=category,
                    response_text=response_text,
                    user_text=user_text,
                    state=state,
                    recent_captions=history,
                )
                cleaned = self._sanitize_video_caption(caption)
                if cleaned and not self._caption_recently_used(cleaned, history):
                    logger.info(
                        "Generated LLM video caption category=%s length=%s",
                        category,
                        len(cleaned),
                    )
                    return cleaned
                if cleaned:
                    logger.info("LLM video caption duplicated recent history; using fallback rotation.")
            except Exception as exc:
                logger.exception("LLM video caption generation failed: %s", exc)

        return self._fallback_video_caption(video_path, avoid_captions=history)

    def build_creative_video_foreshadow(
        self,
        *,
        video_path: str | None = None,
        category: str | None = None,
        state: str = "normal",
        compliance_score: int = 5,
        context_hint: str = "",
    ) -> str:
        """Generate rich, varied foreshadowing text the Queen can adapt or use as inspiration.
        This makes video insertion feel special and queen-controlled.
        """
        if not category and video_path:
            category = self._video_category_for_path(video_path) or "sm"

        cat = (category or "sm").lower()
        is_pov = "pov" in cat
        is_intense = "intense" in state.lower()
        high_compliance = compliance_score >= 10
        low_compliance = compliance_score <= 4

        if is_pov:
            templates = [
                "跪好，把手放上去……盯着屏幕上的每一个动作。我现在要你学得一模一样，一点都不能差。",
                "眼睛给我睁大，手别停。对着这个画面，跟着节奏慢慢撸……我说停你才准停。",
                "今天给你看这段，是因为我想看你边看边难受的样子。开始吧，贱货。",
                "看着屏幕上那个可怜的家伙……这就是你现在该有的下场。手给我动起来，寸止住。",
                "把视频当女王的脸好好看着。每一个指令都照着做，敢走神我就知道。",
            ]
        else:
            templates = [
                "先给我跪直了，一字不漏看完。女王是怎么把贱狗玩弄到哭的，你都要学着点。",
                "好好看着这个示范。每一个细节都记在脑子里，等会儿我照着它来收拾你。",
                "跪着看。别眨眼，这是我今天特意挑给你看的——学着怎么当一个听话的玩具。",
                "记住这个画面里的每一个动作。女王现在要你想象自己就是里面那个被玩弄的。",
                "看清楚了再说话。这就是你该有的样子——彻底的、无助的、只属于我的。",
            ]

        if is_intense:
            templates.extend([
                "这次没有温柔。把视频看完，然后立刻告诉我你有多想被这么对待。",
                "强度拉满。盯着屏幕，想象女王现在就站在你身后，一样对待你。",
            ])

        if high_compliance:
            templates.append("因为你最近还算听话，今天破例让你看一段更精致的……但看完要立刻汇报感受，而且不许碰自己。")
        elif low_compliance:
            templates.append("你最近表现太差了，所以只给你看这个最羞耻的片段。好好看着，记清楚自己的位置。")

        if context_hint:
            # lightly incorporate hint
            templates = [t + f" （结合：{context_hint[:30]}）" for t in templates] + templates

        chosen = random.choice(templates)
        # Add variety in length and command style
        if random.random() > 0.6:
            chosen = chosen.replace("。", "，").replace("！", "。") + " 现在就开始。"

        return chosen

    @staticmethod
    def _caption_recently_used(caption: str, recent_captions: list[str]) -> bool:
        normalized = caption.strip().casefold()
        if not normalized:
            return False
        return any(normalized == item.strip().casefold() for item in recent_captions)

    def _video_category_for_path(self, video_path: str) -> str | None:
        videos = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
        category_index = self._build_video_category_index(videos)
        return category_index.primary_category_for_path(
            videos_path=self.videos_path,
            asset_path=Path(video_path),
        )

    @staticmethod
    def _sanitize_video_caption(caption: str) -> str:
        cleaned = caption.strip().strip("「」\"'“”‘’")
        cleaned = cleaned.replace("\n", " ").strip()
        if len(cleaned) > 48:
            cleaned = cleaned[:48].rstrip("，。！？、 ")
        return cleaned

    def _fallback_video_caption(
        self,
        video_path: str,
        *,
        avoid_captions: list[str] | None = None,
    ) -> str:
        category = self._video_category_for_path(video_path)
        if category == "pov":
            pool = list(POV_VIDEO_CAPTION_FALLBACKS)
        elif category == "sm":
            pool = list(SM_VIDEO_CAPTION_FALLBACKS)
        else:
            pool = list(POV_VIDEO_CAPTION_FALLBACKS + SM_VIDEO_CAPTION_FALLBACKS)

        blocked = {item.strip().casefold() for item in (avoid_captions or []) if item and item.strip()}
        fresh_pool = [item for item in pool if item.casefold() not in blocked]
        base = random.choice(fresh_pool or pool)

        # Occasionally enrich with creative style for more variety
        if random.random() > 0.65:
            try:
                creative = self.build_creative_video_foreshadow(
                    video_path=video_path,
                    category=category,
                    state="normal",
                    compliance_score=5,
                )
                # Use a short punchy version of creative as caption
                short = creative.split("。")[0][:45].strip()
                if short and len(short) > 8:
                    return short
            except Exception:
                pass
        return base

    async def generate_scene_image(self, *, prompt: str, count: int = 1) -> list[str]:
        prompt_text = build_scene_image_prompt(
            scene_prompt=prompt.strip(),
            visual_anchor=self._visual_anchor,
        )
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
            logger.info("Skipping image generation for user_id=%s because ENABLE_IMAGE_GENERATION is false.", user_id)
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
            logger.debug("Image generation enabled by visual scene marker for user_id=%s.", user_id)
            return True

        if self._has_explicit_media_request(normalized) and len(self._extract_context_terms(context)) >= 2:
            logger.debug("Image generation enabled by explicit media request for user_id=%s.", user_id)
            return True

        context_terms = self._extract_context_terms(context)
        descriptive_terms = [term for term in context_terms if len(term) >= 3]
        decision = len(descriptive_terms) >= 2 and len(normalized) >= 18
        logger.debug(
            "Image generation heuristic user_id=%s terms=%s descriptive_terms=%s decision=%s",
            user_id,
            context_terms[:4],
            len(descriptive_terms),
            decision,
        )
        return decision

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

    def _should_use_random_fallback(
        self,
        *,
        user_id: int,
        profile: Any | None,
        context: str,
        outcome: AssetSearchOutcome,
    ) -> bool:
        if outcome.keywords or self._extract_context_terms(context):
            logger.debug(
                "Skipping random fallback for user_id=%s because context was specific but had no strong local match.",
                user_id,
            )
            return False

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

    def _video_offer_probability(self, state: ConversationState) -> float:
        if state == ConversationState.INTENSE:
            return self._clamp_probability(getattr(self.settings, "video_offer_probability_intense", 0.28))
        return self._clamp_probability(getattr(self.settings, "video_offer_probability_normal", 0.18))

    def _pick_video_interval(self, state: ConversationState) -> int:
        if state == ConversationState.INTENSE:
            low = int(getattr(self.settings, "video_intense_min_turns", 10))
            high = int(getattr(self.settings, "video_intense_max_turns", 18))
        else:
            low = int(getattr(self.settings, "video_normal_min_turns", 18))
            high = int(getattr(self.settings, "video_normal_max_turns", 28))
        return random.randint(min(low, high), max(low, high))

    def _pick_video_retry_interval(self, state: ConversationState) -> int:
        if state == ConversationState.INTENSE:
            low = int(getattr(self.settings, "video_retry_min_turns_intense", 2))
            high = int(getattr(self.settings, "video_retry_max_turns_intense", 4))
        else:
            low = int(getattr(self.settings, "video_retry_min_turns_normal", 3))
            high = int(getattr(self.settings, "video_retry_max_turns_normal", 6))
        return random.randint(min(low, high), max(low, high))

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
    def _extract_context_terms(text: str) -> list[str]:
        preserved = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_\s-]", " ", text)
        normalized = preserved.casefold()

        terms: list[str] = []
        for raw_token in re.split(r"[\s_-]+", normalized):
            token = raw_token.strip()
            if len(token) < 2 or token in terms:
                continue

            terms.append(token)
            if len(terms) >= 12:
                break

        return terms

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

    async def _recent_paths_for_user(self, user_id: int | None) -> set[str]:
        if user_id is None:
            return set()
        return await self.delivery_service.recently_delivered_paths(user_id)

    async def _search_assets(self, context: str, *, user_id: int | None = None) -> AssetSearchOutcome:
        keywords = self._extract_keywords(context)
        video_paths = self._list_assets(self.videos_path, self.VIDEO_SUFFIXES)
        category_index = self._build_video_category_index(video_paths)
        matched_categories = category_index.match_categories_in_text(context)
        keywords = self._expand_keywords_with_categories(keywords, matched_categories)
        recent_paths = await self._recent_paths_for_user(user_id)
        images = self._rank_assets(
            self.images_path,
            self.IMAGE_SUFFIXES,
            keywords,
            recent_paths=recent_paths,
        )
        videos = self._rank_assets(
            self.videos_path,
            self.VIDEO_SUFFIXES,
            keywords,
            recent_paths=recent_paths,
            matched_categories=matched_categories,
            category_index=category_index,
        )
        return AssetSearchOutcome(
            images=images,
            videos=videos,
            keywords=keywords,
            matched_categories=matched_categories,
            category_index=category_index,
        )

    @staticmethod
    def _expand_keywords_with_categories(keywords: list[str], matched_categories: list[str]) -> list[str]:
        if not matched_categories:
            return keywords

        expanded = keywords[:]
        for category in matched_categories:
            if category not in expanded:
                expanded.append(category)
        return expanded[:24]

    def _rank_assets(
        self,
        base_path: Path,
        suffixes: set[str],
        keywords: list[str],
        *,
        recent_paths: set[str] | None = None,
        matched_categories: list[str] | None = None,
        category_index: VideoCategoryIndex | None = None,
    ) -> list[AssetCandidate]:
        if not keywords:
            return []

        recent_paths = recent_paths or set()
        matched_categories = matched_categories or []
        ranked: list[AssetCandidate] = []
        for path in self._list_assets(base_path, suffixes):
            candidate = self._apply_repeat_penalty(
                self._score_asset(
                    path,
                    base_path,
                    keywords,
                    matched_categories=matched_categories,
                    category_index=category_index if suffixes == self.VIDEO_SUFFIXES else None,
                ),
                recent_paths,
            )
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

    def _score_asset(
        self,
        path: Path,
        base_path: Path,
        keywords: list[str],
        *,
        matched_categories: list[str] | None = None,
        category_index: VideoCategoryIndex | None = None,
    ) -> AssetCandidate:
        relative = path.relative_to(base_path)
        relative_text = relative.as_posix().casefold()
        filename_text = path.stem.casefold()
        filename_tokens = set(self._extract_keywords(path.stem))

        folder_parts = [part.casefold() for part in relative.parts[:-1]]
        folder_text = "/".join(folder_parts)
        folder_tokens: set[str] = set()
        for part in relative.parts[:-1]:
            folder_tokens.update(self._extract_keywords(part))

        tag_tokens: set[str] = set()
        for tag in self._load_asset_tags(path):
            tag_tokens.update(self._extract_keywords(tag))

        score = 0
        filename_hits = 0
        folder_hits = 0
        tag_hits = 0
        category_bonus = 0

        if category_index is not None and matched_categories:
            primary_category = category_index.primary_category_for_path(
                videos_path=base_path,
                asset_path=path,
            )
            if primary_category and primary_category in matched_categories:
                category_bonus = 22
                score += category_bonus
                folder_hits += 1

        for keyword in keywords:
            keyword_bonus = min(len(keyword), 4)

            if keyword in folder_tokens:
                score += 12 + keyword_bonus
                folder_hits += 1
            elif keyword in folder_text:
                score += 8 + keyword_bonus
                folder_hits += 1

            if keyword in tag_tokens:
                score += 14 + keyword_bonus
                tag_hits += 1
            elif keyword in filename_tokens:
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
            tag_hits=tag_hits,
        )

    def _meta_sidecar_path(self, asset_path: Path) -> Path:
        suffix = str(getattr(self.settings, "asset_meta_filename", ".meta.json")).strip() or ".meta.json"
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        return asset_path.with_name(f"{asset_path.stem}{suffix}")

    def _load_asset_meta(self, path: Path) -> dict[str, Any]:
        meta_path = self._meta_sidecar_path(path)
        if not meta_path.exists():
            return {}

        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read asset metadata for path=%s: %s", path, exc)
            return {}

        return payload if isinstance(payload, dict) else {}

    def _load_asset_tags(self, path: Path) -> list[str]:
        payload = self._load_asset_meta(path)
        raw_tags = payload.get("tags", payload if isinstance(payload, list) else [])
        if not isinstance(raw_tags, list):
            return []

        tags: list[str] = []
        for item in raw_tags:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned:
                    tags.append(cleaned)
        return tags

    def _load_asset_caption(self, path: Path) -> str | None:
        payload = self._load_asset_meta(path)
        for key in ("caption", "video_caption", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _apply_repeat_penalty(self, candidate: AssetCandidate, recent_paths: set[str]) -> AssetCandidate:
        if str(candidate.path) not in recent_paths:
            return candidate

        penalty = max(0, int(getattr(self.settings, "media_repeat_penalty_score", 24)))
        return AssetCandidate(
            path=candidate.path,
            score=max(0, candidate.score - penalty),
            filename_hits=candidate.filename_hits,
            folder_hits=candidate.folder_hits,
            tag_hits=candidate.tag_hits,
        )

    @staticmethod
    def _prefer_fresh_assets(paths: list[Path], recent_paths: set[str]) -> list[Path]:
        if not recent_paths:
            return paths

        fresh_paths = [path for path in paths if str(path) not in recent_paths]
        return fresh_paths or paths

    def _select_rotated_video(
        self,
        outcome: AssetSearchOutcome,
        *,
        recent_paths: set[str],
    ) -> str | None:
        if not outcome.videos:
            return None

        pool = outcome.videos
        if outcome.matched_categories and outcome.category_index is not None:
            category_pool = [
                candidate
                for candidate in pool
                if outcome.category_index.primary_category_for_path(
                    videos_path=self.videos_path,
                    asset_path=candidate.path,
                )
                in outcome.matched_categories
            ]
            if category_pool:
                pool = category_pool

        best_score = max(candidate.score for candidate in pool)
        score_band = max(0, int(getattr(self.settings, "video_rotation_score_band", 6)))
        tier = [candidate for candidate in pool if candidate.score >= best_score - score_band and candidate.score > 0]
        if not tier:
            return None

        fresh_tier = [candidate for candidate in tier if str(candidate.path) not in recent_paths]
        chosen = random.choice(fresh_tier or tier)
        logger.debug(
            "Rotated video selection pool=%s tier=%s fresh=%s chosen=%s score=%s",
            len(pool),
            len(tier),
            len(fresh_tier),
            chosen.path,
            chosen.score,
        )
        return str(chosen.path)

    def _outcome_to_payload(
        self,
        outcome: AssetSearchOutcome,
        *,
        recent_paths: set[str] | None = None,
        media_kind: str = "any",
    ) -> dict[str, list[str]]:
        recent_paths = recent_paths or set()
        limit = self._max_media_items_per_message()
        if limit <= 0:
            return {"images": [], "videos": []}

        if media_kind == "video":
            rotated_video = self._select_rotated_video(outcome, recent_paths=recent_paths)
            if rotated_video:
                return {"images": [], "videos": [rotated_video]}
            return {"images": [], "videos": []}

        ranked: list[tuple[str, AssetCandidate]] = []
        if media_kind in {"any", "image"}:
            ranked.extend(("image", item) for item in outcome.images)
        if media_kind in {"any", "video"}:
            ranked.extend(("video", item) for item in outcome.videos)
        if not ranked:
            return {"images": [], "videos": []}

        ranked.sort(key=lambda item: item[1].score, reverse=True)

        selected: list[tuple[str, AssetCandidate]]
        if limit == 1:
            best_score = ranked[0][1].score
            top_choices = [item for item in ranked if item[1].score == best_score]
            fresh_choices = [
                item for item in top_choices if str(item[1].path) not in recent_paths
            ]
            selected = [random.choice(fresh_choices or top_choices)]
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

    def _is_good_video_match(self, outcome: AssetSearchOutcome, context: str) -> bool:
        if not outcome.videos:
            return False

        keywords = outcome.keywords or self._extract_keywords(context)
        if not keywords:
            return False

        best_video_score = outcome.best_video_score
        if best_video_score >= 16:
            return True
        if best_video_score >= 12 and len(keywords) <= 4:
            return True
        return best_video_score >= 10 and any(len(keyword) >= 4 for keyword in keywords)

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

    @staticmethod
    def _has_explicit_media_request(normalized_context: str) -> bool:
        return any(marker in normalized_context for marker in MediaService.EXPLICIT_MEDIA_MARKERS)

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
