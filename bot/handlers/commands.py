from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from services.task_service import TaskService
from services.user_service import UserService

router = Router(name="commands")


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
    reply = (
        f"{user.display_name}, the channel is live.\n"
        "Commands: /profile, /task, /pause, /resume, /state.\n"
        "Safewords are intercepted immediately at the code layer."
    )
    await message.answer(reply)


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
    task_line = active_task.title if active_task else "none"
    reply = (
        f"State: {user.state}\n"
        f"Conversation turns: {user.conversation_count}\n"
        f"Next task window: turn {user.next_task_turn}\n"
        f"Compliance score: {user.compliance_score}\n"
        f"Tracked dislikes: {len(user.dislikes)}\n"
        f"Active task: {task_line}"
    )
    await message.answer(reply)


@router.message(Command("task"))
async def task_command(
    message: Message,
    task_service: TaskService,
) -> None:
    task = await task_service.get_open_task(message.from_user.id)
    if task is None:
        await message.answer("No active task is currently open.")
        return

    reply = (
        f"Task: {task.title}\n"
        f"Instructions: {task.instructions}\n"
        f"Intensity: {task.intensity}\n"
        f"Status: {task.status}"
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
    await message.answer("Tasks are paused. The state is now 'paused'.")


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
    await message.answer("State returned to 'normal'. Task scheduling is active again.")


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
    await message.answer(f"Current state: {profile.state}")

