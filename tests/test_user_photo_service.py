from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.user_photo_service import UserPhotoService


@pytest.mark.asyncio
async def test_save_telegram_photo_downloads_largest_variant(settings, tmp_path, monkeypatch):
    service = UserPhotoService(settings=settings)
    destination_holder: dict[str, str] = {}

    async def fake_download(file_path: str, destination: str) -> None:
        destination_holder["path"] = str(destination)
        Path = __import__("pathlib").Path
        Path(destination).write_bytes(b"fake-image")

    bot = SimpleNamespace(
        get_file=AsyncMock(
            return_value=SimpleNamespace(file_path="photos/file_1.jpg"),
        ),
        download_file=AsyncMock(side_effect=fake_download),
    )
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=99),
        photo=[
            SimpleNamespace(file_id="small", file_size=100),
            SimpleNamespace(file_id="large", file_size=900),
        ],
        caption="验证照",
    )

    saved = await service.save_telegram_photo(bot=bot, message=message)

    assert saved.telegram_user_id == 99
    assert saved.file_id == "large"
    assert saved.caption == "验证照"
    assert saved.path.endswith(".jpg")
    assert destination_holder["path"] == saved.path