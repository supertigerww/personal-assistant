from aiogram import Router


def build_router() -> Router:
    from .callbacks import router as callbacks_router
    from .commands import router as commands_router
    from .messages import router as messages_router

    root_router = Router()
    root_router.include_router(commands_router)
    root_router.include_router(callbacks_router)
    root_router.include_router(messages_router)
    return root_router