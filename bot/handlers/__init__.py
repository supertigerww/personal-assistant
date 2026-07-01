from aiogram import Router

from .commands import router as commands_router
from .messages import router as messages_router


def build_router() -> Router:
    root_router = Router()
    root_router.include_router(commands_router)
    root_router.include_router(messages_router)
    return root_router

