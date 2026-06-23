"""Test fixtures: an in-process client over the real FastAPI app + Postgres.

Each test gets a clean schema (tables truncated). We talk to the app via
httpx ASGITransport, so no server needs to be running — but a Postgres instance
does (``docker compose up -d``).
"""

import os
from pathlib import Path

# Default to the docker-compose database so the suite runs out of the box.
os.environ.setdefault("DATABASE_URL", "postgresql://ehr:ehr@localhost:5432/ehr")

import pytest_asyncio
from app import database
from app.main import app
from httpx import ASGITransport, AsyncClient

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"


@pytest_asyncio.fixture
async def client():
    await database.init_pool()
    async with database.acquire() as conn:
        await conn.execute(MIGRATION.read_text())
        await conn.execute("TRUNCATE appointments, patients RESTART IDENTITY CASCADE")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await database.close_pool()
