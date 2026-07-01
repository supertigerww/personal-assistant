from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.grok_client import GrokClient


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


def build_settings(**overrides):
    base = {
        "xai_api_key": "test-key",
        "xai_base_url": "https://api.x.ai/v1",
        "xai_model": "grok-test",
        "xai_image_model": "grok-image-test",
        "xai_max_retries": 1,
        "xai_retry_delay_seconds": 0.0,
        "enable_image_generation": True,
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
        input_items=[{"role": "user", "content": "你好\x00 world"}],
        tools=[],
    )

    assert response.id == "resp-1"
    assert stub_client.responses.calls == 2
    assert stub_client.responses.last_kwargs["input"][0]["content"] == "你好 world"


@pytest.mark.asyncio
async def test_generate_image_retries_and_preserves_chinese_prompt():
    settings = build_settings()
    client = GrokClient(settings)
    stub_client = StubOpenAIClient(
        response_actions=[],
        image_actions=[
            RuntimeError("temporary failure"),
            SimpleNamespace(data=[SimpleNamespace(url="https://example.com/a.png")]),
        ],
    )
    client._client = stub_client

    urls = await client.generate_image(prompt="中文提示词", count=1)

    assert urls == ["https://example.com/a.png"]
    assert stub_client.images.calls == 2
    assert stub_client.images.last_kwargs["prompt"] == "中文提示词"


def test_init_requires_basic_xai_settings():
    with pytest.raises(RuntimeError, match="xai_api_key"):
        GrokClient(build_settings(xai_api_key=None))
