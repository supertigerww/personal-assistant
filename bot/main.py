from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from dotenv import load_dotenv

from bot.handlers import build_router
from core.config import Settings, get_settings
from core.context_builder import ContextBuilder
from core.grok_client import GrokClient
from core.queen_engine import QueenEngine
from db.database import Database
from services.media_service import MediaService
from services.memory_service import MemoryService
from services.safety_service import SafetyService
from services.task_service import TaskService
from services.user_service import UserService

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main() -> None:
    bot: Bot | None = None
    try:
        load_dotenv()
        settings = get_settings()
        configure_logging(settings)
        logger.info("Initializing Queen Bot services.")

        database = Database(settings.sqlite_path)
        await database.initialize()
        logger.info("Database initialized at %s", settings.sqlite_path)

        user_service = UserService(database=database, settings=settings)
        task_service = TaskService(database=database, settings=settings, user_service=user_service)
        memory_service = MemoryService(database=database)
        grok_client = GrokClient(settings=settings)

        # MediaService accepts an optional user_service so it can make state-aware media decisions.
        media_service = MediaService(settings=settings, grok_client=grok_client, user_service=user_service)
        safety_service = SafetyService(
            settings=settings,
            user_service=user_service,
            task_service=task_service,
        )
        context_builder = ContextBuilder(settings=settings)
        engine = QueenEngine(
            settings=settings,
            grok_client=grok_client,
            user_service=user_service,
            task_service=task_service,
            memory_service=memory_service,
            media_service=media_service,
            safety_service=safety_service,
            context_builder=context_builder,
        )
        logger.info("Core services initialized successfully.")

        session = AiohttpSession(api=TelegramAPIServer.from_base(settings.bot_endpoint))
        bot = Bot(token=settings.bot_token, session=session)
        dispatcher = Dispatcher()
        root_router = build_router()
        dispatcher.include_router(root_router)
        logger.info("Router tree initialized with all registered handlers.")

        dispatcher.workflow_data.update(
            {
                "settings": settings,
                "user_service": user_service,
                "task_service": task_service,
                "engine": engine,
            }
        )
        logger.info("Dispatcher workflow data initialized.")
        logger.info("Queen Bot started successfully.")

        await dispatcher.start_polling(bot)
    except Exception as exc:
        logger.exception("Fatal error in main: %s", exc)
        raise
    finally:
        if bot is not None:
            await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
