"""Test fixtures: an in-process client over the real FastAPI app + a DEDICATED
test Postgres database.

The suite runs against its own database (``ehr_test`` by default) so it can never
truncate dev/seed data — and we refuse to start unless the target db name ends in
``_test`` as a safety net. The database is created automatically if missing, so
the suite still runs out of the box once Postgres is up (``docker compose up -d``).

Each test starts from a clean schema (tables truncated). The ``docket`` fixture
additionally loads a canonical, known dataset (see ``fixtures.py``) so a test can
rely on, say, "Jane's cancellable appointment" always being present.

We talk to the app via httpx ASGITransport, so no server needs to be running —
but a Postgres instance does.
"""

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg

# Talk to a dedicated test database, never the dev one. Override with TEST_DATABASE_URL.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://ehr:ehr@localhost:5432/ehr_test"
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL

import pytest_asyncio
from app import database
from app.main import app
from httpx import ASGITransport, AsyncClient
from tests.fixtures import Docket, load_docket

MIGRATION = Path(__file__).resolve().parent.parent / "migrations" / "001_initial.sql"


def _db_name(dsn: str) -> str:
    return urlsplit(dsn).path.lstrip("/")


async def _ensure_test_database(dsn: str) -> None:
    """Create the test database if it doesn't exist, after guarding its name.

    Connects to the always-present ``postgres`` maintenance database to issue the
    ``CREATE DATABASE`` (which cannot run inside a transaction).
    """
    name = _db_name(dsn)
    if not name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to run tests against database {name!r}: the test database name "
            "must end in '_test' so the suite can never truncate dev data. "
            "Set TEST_DATABASE_URL to a *_test database."
        )
    admin_dsn = urlunsplit(urlsplit(dsn)._replace(path="/postgres"))
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def client():
    await _ensure_test_database(TEST_DATABASE_URL)
    await database.init_pool()
    async with database.acquire() as conn:
        await conn.execute(MIGRATION.read_text())
        await conn.execute("TRUNCATE appointments, patients RESTART IDENTITY CASCADE")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await database.close_pool()


@pytest_asyncio.fixture
async def docket(client) -> Docket:
    """Load the canonical dataset into the (already-truncated) test DB and return
    handles to the seeded rows. Built on ``client`` so the data is fresh per test."""
    async with database.acquire() as conn:
        return await load_docket(conn)
