import asyncio

import pytest

from services.processing_gate import ProcessingGate


@pytest.mark.asyncio
async def test_processing_gate_serializes_same_user():
    gate = ProcessingGate()
    order: list[str] = []

    async def job(label: str, delay: float) -> None:
        async with gate.acquire(7):
            order.append(f"{label}-start")
            await asyncio.sleep(delay)
            order.append(f"{label}-end")

    await asyncio.gather(
        job("first", 0.05),
        job("second", 0.01),
    )

    assert order == ["first-start", "first-end", "second-start", "second-end"]


@pytest.mark.asyncio
async def test_processing_gate_allows_different_users_in_parallel():
    gate = ProcessingGate()
    order: list[str] = []

    async def job(user_id: int, label: str, delay: float) -> None:
        async with gate.acquire(user_id):
            order.append(f"{label}-start")
            await asyncio.sleep(delay)
            order.append(f"{label}-end")

    await asyncio.gather(
        job(1, "u1", 0.05),
        job(2, "u2", 0.01),
    )

    assert order.index("u2-end") < order.index("u1-end")