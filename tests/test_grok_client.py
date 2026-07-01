from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.grok_client import GrokClient


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0JcAAAAASUVORK5CYII="
)


class StubResponsesAPI:
    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class StubImagesAPI:
    def __init__(self, actions):
        self._actions = list(actions)
        self.calls = 0
        self.last_kwargs = None

    async def generate(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        action = self._actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class StubOpenAIClient:
    def __init__(self, *, response_actions, image_actions):
        self.responses = StubResponsesAPI(response_actions)
        self.images = StubImagesAPI(image_actions)


class FakeAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def build_settings(**overrides):
    base = {
        "xai_api_key": "test-key",
        "xai_base_url": "https://api.x.ai/v1",
        "xai_model": "grok-test",
        "xai_image_model": "grok-image-test",
        "xai_max_retries": 1,
        "xai_retry_delay_seconds": 0.0,
        "enable_image_generation": True,
        "generated_images_path": "data/generated-images-test",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_create_response_retries_and_normalizes_input():
    settings = build_settings()
    client = GrokClient(settings)
    stub_client = StubOpenAIClient(
        response_actions=[RuntimeError("temporary failure"), SimpleNamespace(id="resp-1")],
        image_actions=[],
    )
    client._client = stub_client

    response = await client.create_response(
        input_items=[{"role": "user", "content": "\u4f60\u597d\x00 world"}],
        tools=[],
    )

    assert response.id == "resp-1"
    assert stub_client.responses.calls == 2
    assert stub_client.responses.last_kwargs["input"][0]["content"] == "\u4f60\u597d world"


@pytest.mark.asyncio
async def test_generate_image_downloads_remote_url_to_local_file(tmp_path):
    settings = build_settings(generated_images_path=str(tmp_path))
    client = GrokClient(settings)
    stub_client = StubOpenAIClient(
        response_actions=[],
        image_actions=[SimpleNamespace(data=[SimpleNamespace(url="https://example.com/a.png")])],
    )
    client._client = stub_client

    async def fake_download(url: str) -> tuple[bytes, str | None]:
        assert url == "https://example.com/a.png"
        return PNG_BYTES, "image/png"

    client._download_generated_image = fake_download  # type: ignore[method-assign]

    sources = await client.generate_image(prompt="\u4e2d\u6587\u63d0\u793a\u8bcd", count=1)

    assert len(sources) == 1
    output_path = Path(sources[0])
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.read_bytes() == PNG_BYTES
    assert stub_client.images.calls == 1
    assert stub_client.images.last_kwargs["prompt"] == "\u4e2d\u6587\u63d0\u793a\u8bcd"


@pytest.mark.asyncio
async def test_generate_image_supports_b64_payloads(tmp_path):
    settings = build_settings(generated_images_path=str(tmp_path))
    client = GrokClient(settings)
    stub_client = StubOpenAIClient(
        response_actions=[],
        image_actions=[
            SimpleNamespace(
                data=[SimpleNamespace(b64_json=base64.b64encode(PNG_BYTES).decode("ascii"))]
            )
        ],
    )
    client._client = stub_client

    sources = await client.generate_image(prompt="\u4e2d\u6587\u63d0\u793a\u8bcd", count=1)

    assert len(sources) == 1
    output_path = Path(sources[0])
    assert output_path.exists()
    assert output_path.suffix == ".png"
    assert output_path.read_bytes() == PNG_BYTES


@pytest.mark.asyncio
async def test_non_retryable_api_error_stops_immediately():
    settings = build_settings()
    client = GrokClient(settings)
    stub_client = StubOpenAIClient(
        response_actions=[FakeAPIError("model not found", status_code=400)],
        image_actions=[],
    )
    client._client = stub_client

    with pytest.raises(FakeAPIError, match="model not found"):
        await client.create_response(input_items=[{"role": "user", "content": "hi"}], tools=[])

    assert stub_client.responses.calls == 1


def test_init_requires_basic_xai_settings():
    with pytest.raises(RuntimeError, match="xai_api_key"):
        GrokClient(build_settings(xai_api_key=None))
