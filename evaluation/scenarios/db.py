"""Direct access to the dedicated test database for scenario setup + inspection.

The eval talks to the EHR through its real HTTP API, but it owns the *ground truth*
by connecting straight to Postgres: it seeds each scenario's known starting state
before a call and inspects the final rows after it to decide pass/fail. Same safety
rule as the EHR test suite — we refuse any database whose name doesn't end in
``_test`` so a run can never touch dev data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg

TZ = ZoneInfo("Europe/Madrid")  # clinic wall-clock, matches the EHR
DEFAULT_TEST_DSN = "postgresql://ehr:ehr@localhost:5432/ehr_test"
MIGRATION = (
    Path(__file__).resolve().parents[2] / "ehr-api" / "migrations" / "001_initial.sql"
)


def _db_name(dsn: str) -> str:
    return urlsplit(dsn).path.lstrip("/")


def guard_test_dsn(dsn: str) -> None:
    if not _db_name(dsn).endswith("_test"):
        raise RuntimeError(
            f"Refusing to run the eval against {_db_name(dsn)!r}: the database name must "
            "end in '_test'. Set EVAL_DATABASE_URL to a *_test database."
        )


async def ensure_database(dsn: str) -> None:
    """Create the test database if it doesn't exist yet (after guarding its name)."""
    guard_test_dsn(dsn)
    name = _db_name(dsn)
    admin_dsn = urlunsplit(urlsplit(dsn)._replace(path="/postgres"))
    conn = await asyncpg.connect(admin_dsn)
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def reset(conn: asyncpg.Connection) -> None:
    """Apply the schema and truncate to a clean slate before seeding a scenario."""
    await conn.execute(MIGRATION.read_text())
    await conn.execute("TRUNCATE appointments, patients RESTART IDENTITY CASCADE")


# --- clinic clock helpers (mirror the EHR's date logic) ---------------------


def next_weekday_at(hour: int, days_ahead: int = 1) -> datetime:
    """First weekday at least ``days_ahead`` out, at ``hour`` clinic-local."""
    day = datetime.now(TZ).date() + timedelta(days=days_ahead)
    while day.weekday() >= 5:  # skip Sat/Sun
        day += timedelta(days=1)
    return datetime.combine(day, datetime.min.time(), tzinfo=TZ).replace(hour=hour)


def next_saturday() -> datetime:
    day = datetime.now(TZ).date() + timedelta(days=1)
    while day.weekday() != 5:
        day += timedelta(days=1)
    return datetime.combine(day, datetime.min.time(), tzinfo=TZ).replace(hour=10)


def human(dt: datetime) -> str:
    """A natural, unambiguous phrasing of a slot for the caller persona to speak."""
    h = dt.hour
    suffix = "AM" if h < 12 else "PM"
    h12 = h if 1 <= h <= 12 else abs(h - 12)
    return dt.strftime(f"%A %-d %B at {h12}:00 {suffix}")


# --- seeding ----------------------------------------------------------------


async def insert_patient(
    conn: asyncpg.Connection, full_name: str, dob: str, phone: str
) -> UUID:
    return await conn.fetchval(
        "INSERT INTO patients (full_name, date_of_birth, phone) VALUES ($1, $2, $3) RETURNING id",
        full_name,
        datetime.strptime(dob, "%Y-%m-%d").date(),
        phone,
    )


async def insert_appointment(
    conn: asyncpg.Connection,
    patient_id: UUID,
    starts_at: datetime,
    status: str = "scheduled",
    held_minutes: int | None = None,
) -> UUID:
    held_until = (
        datetime.now(TZ) + timedelta(minutes=held_minutes) if held_minutes is not None else None
    )
    return await conn.fetchval(
        """
        INSERT INTO appointments (patient_id, starts_at, ends_at, status, held_until)
        VALUES ($1, $2, $3, $4, $5) RETURNING id
        """,
        patient_id,
        starts_at,
        starts_at + timedelta(hours=1),
        status,
        held_until,
    )


# --- inspection -------------------------------------------------------------


async def appointments_for(conn: asyncpg.Connection, patient_id: UUID) -> list[dict]:
    rows = await conn.fetch(
        "SELECT id, patient_id, starts_at, status FROM appointments WHERE patient_id = $1",
        patient_id,
    )
    return [dict(r) for r in rows]


async def scheduled_at(conn: asyncpg.Connection, starts_at: datetime) -> list[dict]:
    rows = await conn.fetch(
        "SELECT id, patient_id, status FROM appointments "
        "WHERE starts_at = $1 AND status = 'scheduled'",
        starts_at,
    )
    return [dict(r) for r in rows]


async def all_scheduled(conn: asyncpg.Connection) -> list[dict]:
    rows = await conn.fetch(
        "SELECT id, patient_id, starts_at, status FROM appointments WHERE status = 'scheduled'"
    )
    return [dict(r) for r in rows]


async def patient_count(conn: asyncpg.Connection, full_name: str, dob: str) -> int:
    return await conn.fetchval(
        "SELECT count(*) FROM patients WHERE full_name = $1 AND date_of_birth = $2",
        full_name,
        datetime.strptime(dob, "%Y-%m-%d").date(),
    )


async def appointment_status(conn: asyncpg.Connection, appt_id: UUID) -> str | None:
    return await conn.fetchval("SELECT status FROM appointments WHERE id = $1", appt_id)
