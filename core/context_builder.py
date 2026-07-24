from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.media_intent import MediaTurnHints
from core.models import Task, TaskFollowupKind, UserProfile


class ContextBuilder:
    # Openings / endings that commonly collapse into robotic templates.
    TEMPLATE_OPENING_PREFIXES = (
        "跪好",
        "跪好！",
        "跪好，",
        "废物贱狗",
        "贱狗你",
        "贱狗，",
    )
    TEMPLATE_ENDING_PHRASES = (
        "继续，别停",
        "继续别停",
        "给我继续，别停",
        "给我继续别停",
        "——继续",
        "—继续",
    )
    HIGH_FREQUENCY_PHRASE_CANDIDATES = (
        "废物贱狗",
        "黑丝锁奴绿帽狗",
        "黑丝锁奴",
        "绿帽狗",
        "只配给妓女服务",
        "狗屌永远不准高潮",
        "一周都不准喷",
        "声音再惨点",
        "屁股撅高",
        "继续发料",
        # Sticky JOI / confession loops
        "对着屏幕",
        "毁掉快感",
        "毁掉每一次快感",
        "握紧那根",
        "握紧",
        "只准神着跳",
        "只准在脑子里",
        "先给我老实坦白",
        "盯着我的黑丝",
        "继续握着",
        "讲清楚",
        "原话讲",
        "详细说给我听",
        "说给我听",
        "在脑子里",
        "画面原话",
        "只准看不准碰",
        "手先停",
        "停手",
        "慢下来",
    )

    # Rotating play axes — prefer COMMANDS / teasing over "make user write essays".
    PLAY_AXIS_HINTS = (
        "foot_command: 足控——你下令盯脚/闻鞋/跪姿，用你自己的羞辱叙述加压，少逼他写作文",
        "service_tease: 侍奉戏弄——你口述他该怎么舔/服侍，夹嘲讽，直接给动作不要连环提问",
        "allow_stroke_edge: 允许慢撸/按节奏边缘，夹脏话羞辱，不是全程停手",
        "ruined_tease: 快到边缘时突然改规则（慢/停/只准蹭），用戏弄口吻，不是审讯",
        "identity_insult: 强制贱称+你单方面骂到位，再给一个短动作命令",
        "compare_mock: 对比正常男人/女奴男友，你直接嘲讽判决，少问他问题",
        "nipple_order: 乳头/身体玩弄的直接指令+羞辱",
        "outfit_tease: 黑丝/手套/高跟细节，你描述并命令他盯着硬",
        "cuck_scene_push: 你主动讲一小段绿帽/旁观画面并下令他怎么跪着听，禁止让他填空写作",
        "posture_snap: 突然换姿势（跪/撅/手背后）并立刻接下一句侮辱",
        "reward_stroke: 因听话短暂允许多撸几下当赏赐，同时骂他下贱",
        "public_mock: 公开/被听见幻想一句+命令，不要问卷",
    )

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        self._prompt_path = Path(settings.prompt_path)
        self._system_prompt = self._prompt_path.read_text(encoding="utf-8")

    def build_messages(
        self,
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
        active_task: Task | None,
        task_window_ready: bool,
        photo_task_window_ready: bool = False,
        local_media_summary: dict[str, int],
        task_followup_kind: TaskFollowupKind = TaskFollowupKind.NONE,
        resolved_task: Task | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        media_turn_hints: MediaTurnHints | None = None,
        video_categories_context: str | None = None,
    ) -> list[dict[str, str]]:
        runtime_context = self._build_runtime_context(
            profile=profile,
            user_text=user_text,
            active_task=active_task,
            task_window_ready=task_window_ready,
            photo_task_window_ready=photo_task_window_ready,
            local_media_summary=local_media_summary,
            recent_messages=recent_messages,
            task_followup_kind=task_followup_kind,
            resolved_task=resolved_task,
            settings=self._settings,
            recalled_memories=recalled_memories or [],
            media_turn_hints=media_turn_hints,
            video_categories_context=video_categories_context,
        )

        messages: list[dict[str, str]] = [{"role": "system", "content": f"{self._system_prompt}\n\n{runtime_context}"}]
        for item in recent_messages:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    @staticmethod
    def _build_runtime_context(
        *,
        profile: UserProfile,
        active_task: Task | None,
        task_window_ready: bool,
        photo_task_window_ready: bool,
        local_media_summary: dict[str, int],
        recent_messages: list[dict[str, Any]] | None = None,
        task_followup_kind: TaskFollowupKind = TaskFollowupKind.NONE,
        resolved_task: Task | None = None,
        settings: Any | None = None,
        recalled_memories: list[dict[str, Any]] | None = None,
        media_turn_hints: MediaTurnHints | None = None,
        video_categories_context: str | None = None,
        user_text: str = "",
    ) -> str:
        task_summary = "none"
        if active_task is not None:
            task_summary = f"{active_task.title} | {active_task.instructions} | status={active_task.status}"

        resolved_task_summary = "none"
        if resolved_task is not None and task_followup_kind != TaskFollowupKind.NONE:
            resolved_task_summary = (
                f"{resolved_task.title} | {resolved_task.instructions} | status={resolved_task.status}"
            )

        dislikes = ", ".join(profile.dislikes) if profile.dislikes else "none"
        hard_limits = ", ".join(profile.hard_limits) if profile.hard_limits else "none"
        notes = ", ".join(profile.notes[-5:]) if profile.notes else "none"
        recent_mood = "neutral"
        if recent_messages:
            recent_mood = "positive" if any("好" in item.get("content", "") for item in recent_messages[-3:]) else "neutral"

        task_followup_guidance = ContextBuilder._task_followup_guidance(task_followup_kind)
        task_window_guidance = ContextBuilder._task_window_guidance(
            task_window_ready=task_window_ready,
            active_task=active_task,
        )
        photo_task_window_guidance = ContextBuilder._photo_task_window_guidance(
            photo_task_window_ready=photo_task_window_ready,
            active_task=active_task,
        )
        memory_summary = ContextBuilder._format_recalled_memories(recalled_memories or [])
        video_media_guidance = ContextBuilder._video_media_guidance(
            media_turn_hints=media_turn_hints,
            local_media_summary=local_media_summary,
        )
        anti_template_guidance = ContextBuilder._anti_template_guidance(
            user_text=user_text,
            recent_messages=recent_messages or [],
        )
        x_search_recommended = ContextBuilder._x_humiliation_search_recommended(
            profile=profile,
            user_text=user_text,
            recent_messages=recent_messages or [],
            settings=settings,
        )
        x_search_guidance = ContextBuilder._x_humiliation_search_guidance(recommended=x_search_recommended)
        sm_play_search_recommended = ContextBuilder._sm_play_search_recommended(
            profile=profile,
            user_text=user_text,
            recent_messages=recent_messages or [],
            settings=settings,
        )
        sm_play_search_guidance = ContextBuilder._sm_play_search_guidance(
            recommended=sm_play_search_recommended,
            enabled=bool(getattr(settings, "enable_sm_play_web_search", True)),
        )
        play_axis_guidance = ContextBuilder._play_axis_guidance(
            profile=profile,
            user_text=user_text,
            recent_messages=recent_messages or [],
        )
        scene_image_guidance = ContextBuilder._scene_image_guidance(
            profile=profile,
            user_text=user_text,
            recent_messages=recent_messages or [],
        )

        return (
            "Runtime context (important for decision making):\n"
            f"- user_display_name: {profile.display_name}\n"
            f"- current_state: {profile.state}\n"
            f"- conversation_count: {profile.conversation_count}\n"
            f"- compliance_score: {profile.compliance_score}\n"
            "- randomness_tool_available: Call roll_random_twist when the last replies felt samey, openings/endings are blocked, or you need a fresh domination angle. Prefer this over dumping keyword lists.\n"
            f"- intense_enter_compliance_score: {int(getattr(settings, 'intense_enter_compliance_score', 8))}\n"
            f"- intense_exit_compliance_score: {int(getattr(settings, 'intense_exit_compliance_score', 3))}\n"
            f"- task_window_ready: {str(task_window_ready).lower()}\n"
            f"- photo_task_window_ready: {str(photo_task_window_ready).lower()}\n"
            f"- x_humiliation_search_recommended: {str(x_search_recommended).lower()}\n"
            f"- sm_play_search_recommended: {str(sm_play_search_recommended).lower()}\n"
            f"- next_task_turn: {profile.next_task_turn}\n"
            f"- next_photo_task_turn: {profile.next_photo_task_turn}\n"
            f"- next_video_turn: {profile.next_video_turn}\n"
            f"- active_task: {task_summary}\n"
            f"- last_task_followup: {task_followup_kind}\n"
            f"- last_resolved_task: {resolved_task_summary}\n"
            f"- dislikes: {dislikes}\n"
            f"- hard_limits: {hard_limits}\n"
            f"- notes: {notes}\n"
            f"- recalled_long_term_memories:\n{memory_summary}\n"
            f"- recent_mood: {recent_mood}\n"
            f"- local_images_available: {local_media_summary.get('images', 0)}\n"
            f"- local_videos_available: {local_media_summary.get('videos', 0)}\n"
            f"- video_categories_available: {video_categories_context or 'none'}\n"
            "\nOperational reminders (must follow):\n"
            f"{anti_template_guidance}"
            f"{play_axis_guidance}"
            f"{scene_image_guidance}"
            f"{sm_play_search_guidance}"
            f"{x_search_guidance}"
            f"{task_window_guidance}"
            f"{photo_task_window_guidance}"
            "- Task frequency must stay low: normal state every 10-18 turns, intense state every 6-12 turns. Never issue formal tasks in aftercare or paused state.\n"
            "- Tasks must feel human, creative and varied — do NOT default to simple repetition of keywords (e.g. 'repeat this 10 times'). Use the current humiliation theme, fetishes, X content or memory to invent fresh, immersive tasks (writing confessions, specific physical training with detailed reporting, fantasy roleplay, service instructions, etc.). Vary the style and creativity every single time.\n"
            "- Photo verification tasks are even rarer: normal every 22-36 turns, intense every 16-26 turns. Do not casually assign photo tasks.\n"
            "- Default reply mode: verbal domination and humiliation. Do NOT turn every message into a homework-style task.\n"
            "- If last_task_followup is ignored, do not mention that skipped task again.\n"
            "- If last_task_followup is completed or photo_submitted, briefly acknowledge obedience, then continue dominant control.\n"
            "- If last_task_followup is refused or failed, humiliate or punish as appropriate without re-issuing the same task.\n"
            f"{task_followup_guidance}"
            "- Use recalled_long_term_memories naturally. Never claim ignorance of facts listed there.\n"
            "- When user expresses dislike ('不喜欢', '讨厌', '不要'), immediately record it using update_user_profile.\n"
            "- Media usage: Prefer local/X library; call generate_scene_image only for a specific pose/outfit beat or when the user asks to see you — not every turn. Always write a full spoken reply; never replace dialogue with an image-only turn. Never call it for extremely graphic, scat, fluid-heavy, or ultra-degrading content — such requests are automatically intercepted before hitting xAI moderation.\n- NEVER write text claiming you 'generated', '为你生成', '生成女王的形象', or '我生成了' any image or the Queen's visual. If an image is attached by the system, just guide the user to look at it directly (e.g. '看着这张……'). If no image is attached, do not mention generation or image creation at all.\n- Special rule for Queen's image: If user asks '生成形象', '女王的样子', '你的形象' etc., you MUST call generate_scene_image with a CLEAN prompt focused ONLY on the Queen's appearance (use the visual description). This bypasses heavy content blocks so you can show your look even during explicit play.\n"
            "- When creating a formal task, make it feel alive and personal. Vary task types heavily (never repeat the same 'repeat words N times' pattern). Tie it to the exact current conversation, user's mentioned fetishes, or at most one fresh X image/action. Use roll_random_twist if you need fresh inspiration for the task flavor.\n"
            f"{video_media_guidance}"
            "- Safety first: Respect all hard limits and safewords strictly.\n"
            "- Tone: Commanding, stern, lewd and humiliating when appropriate — but natural spoken Chinese, not keyword collage.\n"
        )

    @staticmethod
    def _anti_template_guidance(*, user_text: str, recent_messages: list[dict[str, Any]]) -> str:
        assistant_replies = [
            str(item.get("content") or "").strip()
            for item in recent_messages
            if item.get("role") == "assistant" and str(item.get("content") or "").strip()
        ]
        recent_assistant = assistant_replies[-3:]

        banned_openings: list[str] = []
        banned_endings: list[str] = []
        for reply in recent_assistant:
            opening = ContextBuilder._extract_opening(reply)
            ending = ContextBuilder._extract_ending(reply)
            if opening:
                banned_openings.append(opening)
            if ending:
                banned_endings.append(ending)
            for prefix in ContextBuilder.TEMPLATE_OPENING_PREFIXES:
                if reply.startswith(prefix) and prefix not in banned_openings:
                    banned_openings.append(prefix)
            for phrase in ContextBuilder.TEMPLATE_ENDING_PHRASES:
                if phrase in reply and phrase not in banned_endings:
                    banned_endings.append(phrase)

        # Keep unique, short list
        banned_openings = ContextBuilder._unique_keep_order(banned_openings)[:6]
        banned_endings = ContextBuilder._unique_keep_order(banned_endings)[:6]

        overused = ContextBuilder._find_overused_phrases(assistant_replies[-4:])
        length_mode = ContextBuilder._length_mode_for_user_text(user_text)
        if length_mode == "short":
            length_line = (
                "- length_mode: short. User message is short/reactive. Reply in 1-3 tight sentences. "
                "No long monologue, no multi-theme dump.\n"
                "- 用户这条很短：先点名回应（射了/求/爽/继续等），再给一个短命令。禁止写成长段关键词羞辱文。\n"
            )
        elif length_mode == "medium":
            length_line = (
                "- length_mode: medium. A normal spoken reply is fine; still prefer natural dialogue over essay.\n"
                "- 中等长度即可，保持口语，不要堆标签。\n"
            )
        else:
            length_line = (
                "- length_mode: long_ok. User wrote more detail; you may expand the scene, still max 1-2 fetish themes.\n"
                "- 用户写得较详：可以展开场景，但仍只抓 1～2 个主轴。\n"
            )

        openings_text = " / ".join(banned_openings) if banned_openings else "none from recent turns"
        endings_text = " / ".join(banned_endings) if banned_endings else "none from recent turns"
        overused_text = " / ".join(overused) if overused else "none detected"

        style_hint = ContextBuilder._style_hint_for_turn(
            conversation_fingerprint=len(assistant_replies) + len(user_text),
            length_mode=length_mode,
        )

        joi_loop = ContextBuilder._detect_joi_loop(recent_assistant)
        confession_loop = ContextBuilder._detect_confession_loop(recent_assistant)
        denial_loop = ContextBuilder._detect_denial_only_loop(recent_assistant)
        loop_lines = ""
        if joi_loop:
            loop_lines += (
                "- JOI-loop detected（盯屏幕/握紧/毁掉快感）。本轮禁止同一套路，必须换玩法轴。\n"
            )
        if confession_loop:
            loop_lines += (
                "- Confession-loop detected（讲清楚/原话/脑子里画面/详细说给我听）。"
                "本轮禁止再逼用户写作文式幻想；改为你下命令+戏弄羞辱，最多一句短确认。\n"
            )
        if denial_loop:
            loop_lines += (
                "- Denial-only loop detected（停手/不准碰连续出现）。"
                "本轮必须允许碰/慢撸/按节奏至少一种，用戏弄而不是纯禁止。\n"
            )

        return (
            "Anti-template (MUST obey this turn):\n"
            f"{length_line}"
            f"- Do NOT start this reply with any of: {openings_text}\n"
            f"- Do NOT end this reply with any of: {endings_text}\n"
            f"- Avoid reusing these high-frequency phrases this turn: {overused_text}\n"
            f"{loop_lines}"
            "- Max 1-2 fetish themes this turn. No hashtag-list / keyword-collage style.\n"
            "- React to the user's latest message first, then dominate. Never ignore what they just said.\n"
            "- If the user asked for feet/service/oral fantasy/nipples/etc., follow THAT beat; do not yank them back to screen-staring JOI.\n"
            "- Prefer imperative commands over questions. At most one question this turn; zero is better. End on an order or insult, not an interview.\n"
            "- 本轮结尾优先：直接命令/任务/羞辱判决。禁止连续用「讲清楚？」「是什么姿势？」收尾。\n"
            f"- Style hint for this turn: {style_hint}\n"
            "- If recent assistant replies felt copy-pasted, call roll_random_twist once, then write a fresh angle.\n"
            "- 禁止固定首尾：不要「跪好，废物贱狗…」开头；不要「——继续，别停」或审讯式提问收尾。\n"
        )

    @staticmethod
    def _detect_joi_loop(recent_assistant: list[str]) -> bool:
        if len(recent_assistant) < 2:
            return False
        markers = (
            "对着屏幕",
            "毁掉快感",
            "毁掉每一次",
            "握紧那根",
            "握紧",
            "只准神着",
            "只准在脑子里",
            "盯着屏幕",
            "继续握",
        )
        hits = 0
        for reply in recent_assistant[-3:]:
            if any(marker in reply for marker in markers):
                hits += 1
        return hits >= 2

    @staticmethod
    def _detect_confession_loop(recent_assistant: list[str]) -> bool:
        """User forced to narrate fantasies every turn."""
        if len(recent_assistant) < 2:
            return False
        markers = (
            "讲清楚",
            "原话",
            "详细说",
            "说给我听",
            "在脑子里",
            "画面",
            "描述",
            "告诉我",
            "是什么姿势",
            "怎么想",
            "幻想",
        )
        hits = 0
        for reply in recent_assistant[-3:]:
            # Heavy question + confession demand
            qmarks = reply.count("？") + reply.count("?")
            if any(marker in reply for marker in markers) or qmarks >= 2:
                hits += 1
        return hits >= 2

    @staticmethod
    def _detect_denial_only_loop(recent_assistant: list[str]) -> bool:
        if len(recent_assistant) < 2:
            return False
        markers = (
            "停手",
            "不准碰",
            "不许碰",
            "只准看",
            "手拿开",
            "手先停",
            "不许动",
            "不准射",
            "不许射",
        )
        hits = 0
        for reply in recent_assistant[-3:]:
            if any(marker in reply for marker in markers):
                hits += 1
        return hits >= 2

    @staticmethod
    def _play_axis_guidance(
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
    ) -> str:
        text = (user_text or "").strip()
        assistant_replies = [
            str(item.get("content") or "")
            for item in recent_messages
            if item.get("role") == "assistant"
        ][-3:]
        confession_loop = ContextBuilder._detect_confession_loop(assistant_replies)
        denial_loop = ContextBuilder._detect_denial_only_loop(assistant_replies)
        joi_loop = ContextBuilder._detect_joi_loop(assistant_replies)

        # User-steered axes take priority so the model does not ignore pivots.
        if any(k in text for k in ("脚", "脚底", "脚趾", "鞋", "丝足", "美腿")):
            forced = "foot_command: 用户提脚——足控命令+你方羞辱叙述，禁止拧回审讯作文"
        elif any(k in text for k in ("口交", "舔", "侍奉", "舌头", "含", "交合")):
            forced = "service_tease: 用户提侍奉/交合——你口述画面并下令，允许或边缘撸，少提问"
        elif any(k in text for k in ("乳", "奶", "乳头")):
            forced = "nipple_order: 用户提乳头——直接玩弄指令+羞辱"
        elif any(k in text for k in ("绿帽", "别人", "旁观", "女奴", "cuck", "NTR", "ntr")):
            forced = "cuck_scene_push: 用户提绿帽/女奴——你推进旁观画面+命令，不要让他填空"
        elif any(k in text for k in ("女装", "伪娘", "裙子", "丝袜装")):
            forced = "outfit_tease: 用户提女装——穿戴戏弄+命令"
        elif any(k in text for k in ("太舒服", "不想停", "硬到", "想射", "继续撸", "想碰")):
            forced = "allow_stroke_edge: 用户求继续/太爽——先允许慢撸或边缘戏弄，再突然收紧，禁止纯停手审讯"
        elif confession_loop:
            forced = "reward_stroke: 刚连续逼描述——本轮改成赏赐式慢撸+侮辱，禁止再要长段幻想作文"
        elif denial_loop:
            forced = "allow_stroke_edge: 刚连续停手——本轮必须给撸/节奏命令"
        elif joi_loop:
            forced = "cuck_scene_push: JOI 复读——切换绿帽/旁观画面命令或足控"
        else:
            idx = int(getattr(profile, "conversation_count", 0) or 0) % len(ContextBuilder.PLAY_AXIS_HINTS)
            forced = ContextBuilder.PLAY_AXIS_HINTS[idx]

        must_rotate = confession_loop or denial_loop or joi_loop or (
            int(getattr(profile, "conversation_count", 0) or 0) % 3 == 0
        )
        if must_rotate:
            rotate_line = (
                "- MUST proactively switch/advance play this turn (user should not need to ask for 换花样). "
                "Use recommended_play_axis or roll_random_twist / search_sm_play_ideas.\n"
                "- 本轮必须主动换打法或加码，不要等用户提醒。\n"
            )
        else:
            rotate_line = (
                "- Proactively drive the scene; do not wait for the user to request variety.\n"
                "- 主动推进；用命令而不是提问推进。\n"
            )
        return (
            "Play variety (MUST obey this turn):\n"
            f"- recommended_play_axis: {forced}\n"
            f"{rotate_line}"
            "- Keep 1 main axis. Spoken Chinese. Prefer teasing insults + direct orders over questionnaires.\n"
            "- 本轮：直接给指示/即时任务；你自己把羞辱画面讲出来；少逼用户描述。\n"
            "- Allow stroking/edging on some turns — denial is a tool, not the only mode.\n"
        )

    @staticmethod
    def _scene_image_guidance(
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
    ) -> str:
        text = (user_text or "").strip()
        visual_hooks = (
            "黑丝", "丝袜", "脚", "鞋", "高跟", "乳胶", "跪", "形象", "样子",
            "腿", "手套", "看着", "盯着", "特写",
        )
        user_wants_visual = any(h in text for h in visual_hooks) or any(
            h in text.casefold() for h in ("image", "pic", "photo", "生成")
        )
        # Encourage a still every few turns, stronger when user is visual.
        turn = int(getattr(profile, "conversation_count", 0) or 0)
        cadence_hit = turn > 0 and turn % 3 == 0
        if user_wants_visual:
            return (
                "Scene image (this turn):\n"
                "- scene_image_recommended: soft. User mentioned a visual hook — you MAY call generate_scene_image "
                "with a CLEAN pose/outfit prompt, but still write full spoken orders. Prefer not every consecutive turn.\n"
                "- 用户提到画面点时可出图，但仍必须有完整文字命令；禁止只发图不说话。\n"
            )
        if cadence_hit:
            return (
                "Scene image (this turn):\n"
                "- scene_image_recommended: optional. System may attach local media; only generate if a fresh pose is worth it.\n"
                "- 本轮可选出图；优先靠对话与本地素材，不要为了出图而省略文字。\n"
            )
        return (
            "Scene image (this turn):\n"
            "- scene_image_recommended: false. Skip generate_scene_image unless the user asks to see you.\n"
            "- 本轮默认不要主动 generate_scene_image。\n"
        )

    @staticmethod
    def _sm_play_search_recommended(
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
        settings: Any | None,
    ) -> bool:
        """When True, Queen may call search_sm_play_ideas once for live technique research."""
        if settings is not None and not bool(getattr(settings, "enable_sm_play_web_search", True)):
            return False
        interval = int(getattr(settings, "sm_play_search_interval_turns", 5) or 5)
        interval = max(3, interval)
        turn = int(getattr(profile, "conversation_count", 0) or 0)
        text = (user_text or "").strip()

        trigger_keywords = (
            "换花样",
            "换一个",
            "新玩法",
            "别的玩法",
            "无聊",
            "重复",
            "没意思",
            "单调",
            "再狠",
            "更狠",
            "来点新",
            "教我",
            "训练我",
            "怎么玩",
            "玩法",
        )
        if any(keyword in text for keyword in trigger_keywords):
            return True

        assistant = [
            str(item.get("content") or "")
            for item in recent_messages
            if item.get("role") == "assistant"
        ][-3:]
        if ContextBuilder._detect_joi_loop(assistant):
            return True

        if turn > 0 and turn % interval == 0:
            return True
        return False

    @staticmethod
    def _sm_play_search_guidance(*, recommended: bool, enabled: bool = True) -> str:
        if not enabled:
            return (
                "- sm_play_web_search is disabled. Do NOT call search_sm_play_ideas. "
                "Use roll_random_twist / recommended_play_axis instead.\n"
            )
        if recommended:
            return (
                "- sm_play_search_recommended is true. You MAY call search_sm_play_ideas ONCE "
                "(topic aligned with recommended_play_axis or the user's pivot). "
                "Pick ONE idea from the digest and rewrite as natural spoken orders. "
                "Never mention research/web/X/links/authors.\n"
                "- 本轮可以联网搜一次调教玩法：只取 1 个点子，用你自己的口吻下令，禁止贴检索原文。\n"
            )
        return (
            "- sm_play_search_recommended is false. Do NOT call search_sm_play_ideas this turn "
            "unless the user explicitly asks for 新玩法/换花样.\n"
            "- 本轮默认不联网搜玩法；用对话、推荐玩法轴或 roll_random_twist 推进。\n"
        )

    @staticmethod
    def _x_humiliation_search_recommended(
        *,
        profile: UserProfile,
        user_text: str,
        recent_messages: list[dict[str, Any]],
        settings: Any | None,
    ) -> bool:
        interval = int(getattr(settings, "x_humiliation_search_interval_turns", 4) or 4)
        interval = max(2, interval)
        turn = int(getattr(profile, "conversation_count", 0) or 0)

        text = (user_text or "").strip()
        lower = text.casefold()
        # Explicit user asks for fresher / harder material
        trigger_keywords = (
            "再狠",
            "更狠",
            "换花样",
            "换一个",
            "无聊",
            "重复",
            "没意思",
            "新鲜",
            "来点新",
            "刺激一点",
            "素材",
        )
        if any(keyword in text for keyword in trigger_keywords):
            return True

        # Short reactive turns almost never need external dump
        if ContextBuilder._length_mode_for_user_text(text) == "short":
            # Only on interval boundary
            return turn > 0 and turn % interval == 0

        # Default cadence: every N turns
        if turn > 0 and turn % interval == 0:
            return True

        # If last two assistant replies are both very long, prefer twist over more X
        assistant = [
            str(item.get("content") or "")
            for item in recent_messages
            if item.get("role") == "assistant"
        ][-2:]
        if len(assistant) == 2 and all(len(item) > 180 for item in assistant):
            return False

        # Mild extra chance mid-interval for longer user messages
        if len(text) >= 40 and turn % interval == (interval // 2):
            return True

        # "all" / lower unused but kept for future locale triggers
        _ = lower
        return False

    @staticmethod
    def _x_humiliation_search_guidance(*, recommended: bool) -> str:
        if recommended:
            return (
                "- x_humiliation_search_recommended is true. You MAY call search_x_humiliation ONCE for a single fresh image/action. "
                "Prefer online; fall back to fetch_local_x_humiliation only if online is empty. "
                "Digest into ONE concrete detail in your own words — no hashtags, no keyword dump, never mention sources/X/authors.\n"
                "- 本轮可以拉一次 X 素材：只提炼一个画面/动作，用口语写进羞辱，不要标签堆叠。\n"
            )
        return (
            "- x_humiliation_search_recommended is false. Do NOT call search_x_humiliation or fetch_local_x_humiliation this turn. "
            "Advance the scene with dialogue, memory, and the user's latest message. Use roll_random_twist if you need variety.\n"
            "- 本轮禁止调用 X 羞辱搜索。靠对话本身推进，不要拼素材库关键词。\n"
        )

    @staticmethod
    def _length_mode_for_user_text(user_text: str) -> str:
        text = (user_text or "").strip()
        if not text:
            return "short"
        # Photo / system submissions may be long wrappers
        if text.startswith("[") and len(text) > 80:
            return "long_ok"
        # Chinese reactive one-liners are often 1–10 chars; keep threshold low.
        # Examples short: 射了 / 好 / 继续 / 啊啊啊射了
        # Examples medium: 主人可以今天训练我寸止吗
        if len(text) <= 10:
            return "short"
        if len(text) <= 80:
            return "medium"
        return "long_ok"

    @staticmethod
    def _extract_opening(text: str, *, max_chars: int = 12) -> str:
        cleaned = re.sub(r"\s+", "", (text or "").strip())
        if not cleaned:
            return ""
        # First clause up to punctuation if short, else prefix
        for sep in ("，", ",", "。", "！", "!", "？", "?", "…", "——", "—"):
            if sep in cleaned[: max_chars + 4]:
                part = cleaned.split(sep, 1)[0]
                if 2 <= len(part) <= max_chars + 2:
                    return part
        return cleaned[:max_chars]

    @staticmethod
    def _extract_ending(text: str, *, max_chars: int = 16) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        # Prefer last sentence fragment
        for sep in ("。", "！", "!", "？", "?", "\n"):
            if sep in cleaned:
                tail = cleaned.rsplit(sep, 1)[-1].strip()
                if tail:
                    cleaned = tail
                    break
        compact = re.sub(r"\s+", "", cleaned)
        # Strip leading dash-like openers in endings
        compact = compact.lstrip("—-–―")
        if len(compact) > max_chars:
            compact = compact[-max_chars:]
        return compact

    @staticmethod
    def _find_overused_phrases(replies: list[str]) -> list[str]:
        if not replies:
            return []
        blob = "\n".join(replies)
        found: list[str] = []
        for phrase in ContextBuilder.HIGH_FREQUENCY_PHRASE_CANDIDATES:
            if blob.count(phrase) >= 1 and phrase not in found:
                # If it appeared in more than one recent reply, or twice in one long reply
                appearances = sum(1 for reply in replies if phrase in reply)
                if appearances >= 2 or blob.count(phrase) >= 2:
                    found.append(phrase)
                elif appearances == 1 and len(replies) <= 2:
                    # Still ban sticky template phrases from the last reply
                    if phrase in ContextBuilder.HIGH_FREQUENCY_PHRASE_CANDIDATES[:6]:
                        found.append(phrase)
        return found[:8]

    @staticmethod
    def _style_hint_for_turn(*, conversation_fingerprint: int, length_mode: str) -> str:
        short_styles = (
            "cold mock of the user's latest reaction only",
            "one sharp command, zero questions",
            "allow slow strokes for two sentences of teasing",
            "sudden focus shift to feet/legs with one cruel order",
            "one degrading nickname + one physical order",
            "reward a few strokes then snatch control back",
        )
        medium_styles = (
            "react first, then one new order — no interview",
            "upgrade last beat without repeating wording",
            "quiet contempt + allow edging rhythm",
            "you narrate the dirty scene; he only obeys",
            "service tease: dictate worship, mock him while he strokes",
            "posture snap then insult, no questions",
        )
        long_styles = (
            "scene push with one fetish theme, commands not questions",
            "memory callback + new teasing pressure",
            "you describe cuck/foot/service detail; he follows orders",
            "contrast humiliation with concrete sensory mockery",
            "outfit tease then a degrading action order",
            "public-fantasy mockery closed with a command",
        )
        pool = short_styles if length_mode == "short" else medium_styles if length_mode == "medium" else long_styles
        return pool[conversation_fingerprint % len(pool)]

    @staticmethod
    def _unique_keep_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(key)
        return result

    @staticmethod
    def _video_media_guidance(
        *,
        media_turn_hints: MediaTurnHints | None,
        local_media_summary: dict[str, int],
    ) -> str:
        videos_available = int(local_media_summary.get("videos", 0))
        if videos_available <= 0:
            return "- local_videos_available is 0. Do not tease or promise videos this turn.\n"

        if media_turn_hints is None:
            return (
                "- Videos are rare and never random. Only tease a video if the user explicitly asked for one.\n"
                "- 视频不会随机塞入对话；只有用户明确要视频时才可自然引导。\n"
            )

        if media_turn_hints.user_wants_video:
            return (
                "- The user explicitly asked for a video. Lead in dominantly; if a matching local video exists, it may attach after your reply.\n"
                "- 用户要视频：用强势口吻自然铺垫（句式每轮换，不要死记「跪好」模板），不要干巴巴只发文件。\n"
            )

        if media_turn_hints.video_window_ready:
            return (
                "- video_window_ready is true. You MAY tease sending a video if it truly fits the scene.\n"
                "- A video attaches ONLY when your reply clearly sets it up with RICH, creative, dominant foreshadowing; system sends text first, then the clip.\n"
                "- Video folders are coarse (usually sm/ vs pov/). sm = spectator SM training; pov = first-person humiliation + JOI/edging talk.\n"
                "- 「撸/寸止/不许射」are verbal orders YOU speak this turn — not extra folders. Pair pov/ clips with stroke/edging commands; pair sm/ with watch-and-learn humiliation.\n"
                "- Vary the setup every time (do not copy prior openings). Use sensory detail and one clear command. Never just say '看这个视频'.\n"
                "- Match folder hints in video_categories_available. Do not mention video every turn.\n"
                "- 此刻可偶尔发视频：铺垫要新鲜，避免同一套开头结尾。\n"
            )

        return (
            "- video_window_ready is false. Do not tease or promise videos this turn unless the user explicitly asks.\n"
            "- 本条不要主动提视频；继续言语支配即可。\n"
        )

    @staticmethod
    def _photo_task_window_guidance(*, photo_task_window_ready: bool, active_task: Task | None) -> str:
        if active_task is not None:
            return ""
        if not photo_task_window_ready:
            return (
                "- photo_task_window_ready is false. Do NOT create photo verification tasks in this reply.\n"
                "- 本条禁止发拍照验证类正式任务。口头提一句可以，但不要布置拍照作业。\n"
            )
        return (
            "- photo_task_window_ready is true. You MAY create a photo verification task only if truly warranted — default is still to skip.\n"
            "- 只有此刻可以发拍照验证任务，且应极少使用。\n"
        )

    @staticmethod
    def _format_recalled_memories(memories: list[dict[str, Any]]) -> str:
        if not memories:
            return "  none"
        lines: list[str] = []
        for index, memory in enumerate(memories, start=1):
            category = memory.get("category", "unknown")
            text = memory.get("text", "")
            lines.append(f"  {index}. [{category}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _task_window_guidance(*, task_window_ready: bool, active_task: Task | None) -> str:
        if active_task is not None:
            return (
                "- An open task already exists. Do NOT call create_task. Do not assign another formal task in this reply.\n"
                "- 已有进行中的任务：禁止再发正式任务，本条专注羞辱、命令或推进场景。\n"
            )
        if not task_window_ready:
            return (
                "- task_window_ready is false. Do NOT call create_task. No formal task in this reply.\n"
                "- Use verbal domination only. Do not assign numbered or homework-style tasks.\n"
                "- 本条禁止发正式任务。不要把每句话都变成「去做XX」的任务布置。\n"
            )
        return (
            "- task_window_ready is true. You MAY call create_task once if the scene truly warrants it — skipping is still the better default.\n"
            "- Even when allowed, most replies should NOT include a formal task. Continue with dialogue and control first.\n"
            "- 只有此刻可以发正式任务，且不是每条都要发；拿不准就不发。\n"
        )

    @staticmethod
    def _task_followup_guidance(task_followup_kind: TaskFollowupKind) -> str:
        guidance = {
            TaskFollowupKind.COMPLETED: (
                "- The user just completed the previous task. Reward compliance briefly, then push forward.\n"
            ),
            TaskFollowupKind.REFUSED: (
                "- The user just refused the previous task. Escalate verbal control and shame the refusal.\n"
            ),
            TaskFollowupKind.FAILED: (
                "- The user admitted failure on the previous task. Mock the failure and tighten control.\n"
            ),
            TaskFollowupKind.IGNORED: (
                "- The user ignored the previous task. Move on without mentioning that task.\n"
            ),
            TaskFollowupKind.PHOTO_SUBMITTED: (
                "- The user just submitted a verification photo for the previous task. Judge, humiliate, or reward based on the photo description.\n"
            ),
        }
        return guidance.get(task_followup_kind, "")
