"""Spin up an isolated, ``ehr_test``-backed EHR API for the integration suite.

The integration tests exercise ``voice_agent``'s HTTP client against a *real*
running EHR API (not the in-process ASGI app the ehr-api suite uses). To keep
them from ever touching dev data, this fixture launches its **own** API process
pointed at the dedicated ``ehr_test`` database — created if missing, guarded so
the name must end in ``_test``, and truncated to a clean slate first — then
points the tool client at it for the whole session and tears it down after.

Same safety rule as the eval and the ehr-api suite: we refuse any database whose
name doesn't end in ``_test``. Requires Postgres up (``docker compose up -d``);
if it isn't reachable the whole suite skips, so ``uv run pytest`` stays green
without infrastructure.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import httpx
import pytest

from voice_agent.tools import implementations as tool_implementations

ROOT = Path(__file__).resolve().parents[2]
EHR_API_DIR = ROOT / "ehr-api"
MIGRATION = EHR_API_DIR / "migrations" / "001_initial.sql"

# Dedicated test database, never dev. Override with TEST_DATABASE_URL.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://ehr:ehr@localhost:5432/ehr_test"
)
# A port of our own so we never collide with the dev API (:8000) or the eval (:8011).
API_PORT = int(os.environ.get("INTEGRATION_API_PORT", "8012"))
BASE_URL = f"http://localhost:{API_PORT}"


def _db_name(dsn: str) -> str:
    return urlsplit(dsn).path.lstrip("/")


def _guard_test_dsn(dsn: str) -> None:
    if not _db_name(dsn).endswith("_test"):
        raise RuntimeError(
            f"Refusing to run integration tests against {_db_name(dsn)!r}: the database "
            "name must end in '_test'. Set TEST_DATABASE_URL to a *_test database."
        )


async def _ensure_database(dsn: str) -> None:
    """Create the test database if it doesn't exist yet (after guarding its name)."""
    _guard_test_dsn(dsn)
    name = _db_name(dsn)
    admin_dsn = urlunsplit(urlsplit(dsn)._replace(path="/postgres"))
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def _reset(dsn: str) -> None:
    """Apply the schema and truncate to a clean slate before the suite runs."""
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(MIGRATION.read_text())
        await conn.execute("TRUNCATE appointments, patients RESTART IDENTITY CASCADE")
    finally:
        await conn.close()


def _api_healthy() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session", autouse=True)
def ehr_test_api():
    """Launch an ``ehr_test``-backed EHR API for the whole integration session."""
    try:
        asyncio.run(_ensure_database(TEST_DATABASE_URL))
        asyncio.run(_reset(TEST_DATABASE_URL))
    except (OSError, asyncpg.PostgresError) as exc:
        pytest.skip(f"Postgres not available for integration tests: {exc}")

    env = {**os.environ, "DATABASE_URL": TEST_DATABASE_URL}
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--port", str(API_PORT)],
        cwd=EHR_API_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            if _api_healthy():
                break
            if proc.poll() is not None:
                raise RuntimeError("EHR API process exited before becoming healthy.")
            time.sleep(0.5)
        else:
            raise RuntimeError("EHR API did not become healthy in time.")

        # The tool client reads this module-global at call time, so point it (and the
        # env var, for anything that reads it later) at our isolated test API.
        tool_implementations.EHR_BASE_URL = BASE_URL
        os.environ["EHR_BASE_URL"] = BASE_URL
        yield BASE_URL
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
