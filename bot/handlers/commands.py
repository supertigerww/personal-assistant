from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from core.models import ConversationState
from services.task_service import TaskService
from services.user_service import UserService

router = Router(name="commands")

STATE_LABELS = {
    ConversationState.NORMAL: "正常",
    ConversationState.INTENSE: "高强度",
    ConversationState.AFTERCARE: "安抚",
    ConversationState.PAUSED: "暂停",
}


def _format_state(state: ConversationState) -> str:
    return STATE_LABELS.get(state, str(state))


def _build_welcome_message(*, display_name: str, onboarding_pending: bool) -> str:
    if onboarding_pending:
        return (
            f"{display_name}，跪下。\n"
            "频道已接通，我是你的女王 Luna。\n\n"
            "先说清楚几件事，再开始玩：\n"
            "1. 想我怎么叫你（例如：叫我贱狗）\n"
            "2. 有什么绝对不要碰的硬限（例如：硬限：公开羞辱）\n"
            "3. 想要什么强度（轻 / 中 / 重）\n\n"
            "直接回复一段话就行。安全词：红色=立刻停止，黄色=慢一点。\n"
            "命令：/profile /task /pause /resume /state"
        )

    return (
        f"{display_name}，回来了。\n"
        "规矩没变：红色立刻停，黄色慢一点。\n"
        "命令：/profile 档案 | /task 当前任务 | /pause 暂停 | /resume 恢复 | /state 状态"
    )


@router.message(CommandStart())
async def start_command(
    message: Message,
    user_service: UserService,
) -> None:
    user = await user_service.get_or_create(
        telegram_user_id=message.from_user.id,
        username=message.from_user.username,
        display_name=message.from_user.full_name,
    )
    await message.answer(
        _build_welcome_message(
            display_name=user.display_name,
            onboarding_pending=not user.onboarding_completed,
        )
    )


@router.message(Command("profile"))
async def profile_command(
    message: Message,
    user_service: UserService,
    task_service: TaskService,
) -> None:
    user = await user_service.get_or_create(
        telegram_user_id=message.from_user.id,
        username=message.from_user.username,
        display_name=message.from_user.full_name,
    )
    active_task = await task_service.get_open_task(user.telegram_user_id)
    task_line = active_task.title if active_task else "无"
    reply = (
        f"【你的档案】\n"
        f"状态：{_format_state(user.state)}\n"
        f"对话轮次：{user.conversation_count}\n"
        f"下次任务窗口：第 {user.next_task_turn} 轮\n"
        f"下次拍照任务窗口：第 {user.next_photo_task_turn} 轮\n"
        f"服从分：{user.compliance_score}\n"
        f"已记录禁忌：{len(user.dislikes)} 条\n"
        f"进行中任务：{task_line}"
    )
    await message.answer(reply)


@router.message(Command("task"))
async def task_command(
    message: Message,
    task_service: TaskService,
) -> None:
    task = await task_service.get_open_task(message.from_user.id)
    if task is None:
        await message.answer("现在没有进行中的任务。乖乖等着。")
        return

    reply = (
        f"【当前任务】\n"
        f"标题：{task.title}\n"
        f"要求：{task.instructions}\n"
        f"强度：{task.intensity}\n"
        f"状态：{task.status}"
    )
    await message.answer(reply)


@router.message(Command("pause"))
async def pause_command(
    message: Message,
    user_service: UserService,
    task_service: TaskService,
) -> None:
    await task_service.pause_all_tasks(message.from_user.id, reason="manual_pause")
    await user_service.update_state(message.from_user.id, "paused", paused_reason="manual_pause")
    await message.answer("已暂停。所有任务停止，先缓一缓。")


@router.message(Command("resume"))
async def resume_command(
    message: Message,
    user_service: UserService,
    task_service: TaskService,
) -> None:
    profile = await user_service.get_or_create(
        telegram_user_id=message.from_user.id,
        username=message.from_user.username,
        display_name=message.from_user.full_name,
    )
    await user_service.update_state(profile.telegram_user_id, "normal", paused_reason=None)
    await task_service.schedule_next_task(
        telegram_user_id=profile.telegram_user_id,
        state="normal",
        from_turn=profile.conversation_count,
    )
    await message.answer("恢复了。继续听话。")


@router.message(Command("state"))
async def state_command(
    message: Message,
    user_service: UserService,
) -> None:
    profile = await user_service.get_or_create(
        telegram_user_id=message.from_user.id,
        username=message.from_user.username,
        display_name=message.from_user.full_name,
    )
    await message.answer(f"当前状态：{_format_state(profile.state)}")