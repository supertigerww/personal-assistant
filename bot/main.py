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
from db.chroma_client import create_chroma_client
from db.database import Database
from services.media_service import MediaService
from services.memory_service import MemoryService
from services.onboarding_service import OnboardingService
from services.safety_service import SafetyService
from services.task_service import TaskService
from services.user_photo_service import UserPhotoService
from services.processing_gate import ProcessingGate
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
        user_photo_service = UserPhotoService(settings=settings)
        chroma_client = create_chroma_client(settings)
        memory_service = MemoryService(
            database=database,
            chroma_client=chroma_client,
            settings=settings,
        )
        if chroma_client.enabled:
            logger.info("Long-term Chroma memory enabled at %s", settings.chroma_path)
        else:
            logger.info("Long-term Chroma memory disabled.")
        grok_client = GrokClient(settings=settings)

        # MediaService accepts optional user_service and database for state-aware media decisions and delivery tracking.
        media_service = MediaService(
            settings=settings,
            grok_client=grok_client,
            user_service=user_service,
            database=database,
        )
        safety_service = SafetyService(
            settings=settings,
            user_service=user_service,
            task_service=task_service,
        )
        context_builder = ContextBuilder(settings=settings)
        onboarding_service = OnboardingService(
            user_service=user_service,
            memory_service=memory_service,
        )
        processing_gate = ProcessingGate()
        engine = QueenEngine(
            settings=settings,
            grok_client=grok_client,
            user_service=user_service,
            task_service=task_service,
            memory_service=memory_service,
            media_service=media_service,
            safety_service=safety_service,
            context_builder=context_builder,
            onboarding_service=onboarding_service,
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
                "user_photo_service": user_photo_service,
                "media_service": media_service,
                "memory_service": memory_service,
                "processing_gate": processing_gate,
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
