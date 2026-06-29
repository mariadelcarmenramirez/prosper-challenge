"""Integration tests for tool_implementations against a live EHR API.

The ``ehr_test_api`` fixture in ``conftest.py`` provisions an isolated EHR API
backed by the dedicated ``ehr_test`` database and points the tool client at it
for the whole session, so these tests never touch dev data. The only thing they
need is Postgres up (``docker compose up -d``); if it isn't reachable the suite
skips automatically, keeping ``uv run pytest`` green without infrastructure.
"""

import uuid

from voice_agent.tools import implementations as tool_implementations


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
