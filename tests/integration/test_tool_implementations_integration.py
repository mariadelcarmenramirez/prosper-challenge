"""Integration tests for tool_implementations against a live EHR API.

Skips automatically if the API is not reachable, so the unit suite stays green
without infrastructure. Run the stack first:
    docker compose up -d
    cd ehr-api && uv run uvicorn app.main:app --port 8000
"""

import os
import uuid

import httpx
import pytest
import pytest_asyncio

from voice_agent.tools import implementations as tool_implementations

BASE = os.environ.get("EHR_BASE_URL", "http://localhost:8000")


async def _is_up() -> bool:
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=2.0) as c:
            return (await c.get("/health")).status_code == 200
    except httpx.HTTPError:
        return False


@pytest_asyncio.fixture(autouse=True)
async def require_api():
    if not await _is_up():
        pytest.skip(f"EHR API not running at {BASE}")


async def test_find_unknown_patient_returns_none():
    result = await tool_implementations.find_patient("No Such Person", "1970-01-01", "+34000000000")
    assert result is None


async def test_create_then_find_returns_same_id():
    name = f"Test {uuid.uuid4().hex[:8]}"
    created = await tool_implementations.create_patient(name, "1991-03-03", "+34622222222")
    assert "id" in created
    found = await tool_implementations.find_patient(name, "1991-03-03", "+34622222222")
    assert found is not None
    assert found["id"] == created["id"]
