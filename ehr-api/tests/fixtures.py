"""Canonical test "docket": a fixed, known dataset loaded fresh before each test
that asks for it.

The point is determinism. A test that cancels Jane's appointment can rely on it
being present at the start of the test, regardless of what previous tests (or
previous runs) did to it — the data comes from here, not from hoping the row
survived. Tear-down is the next test's ``TRUNCATE`` in ``conftest.py``.

Appointment times are computed relative to ``now()`` because the EHR filters on
the database ``now()`` (e.g. ``starts_at > now()``). A row pinned to a hardcoded
date like ``2026-07-06`` would silently fall into the past as the calendar moves;
relative slots stay genuinely "upcoming" forever.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg

TZ = ZoneInfo("Europe/Madrid")  # clinic wall-clock, matches the app

# Canonical identities. Phones differ from those minted by other test modules so
# the docket never collides with data a test creates via the API.
JANE = {"full_name": "Jane Doe", "date_of_birth": date(1990, 1, 1), "phone": "+34600000001"}
JOHN = {"full_name": "John Smith", "date_of_birth": date(1985, 5, 20), "phone": "+34600000002"}


def next_weekday_at(hour: int, days_ahead: int = 1) -> datetime:
    """First weekday at least ``days_ahead`` out, at ``hour`` clinic-local time."""
    day = datetime.now(TZ).date() + timedelta(days=days_ahead)
    while day.weekday() >= 5:  # skip Saturday/Sunday
        day += timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


@dataclass(frozen=True)
class Docket:
    """Handles to the seeded rows so tests can reference them without re-querying."""

    jane_id: UUID
    john_id: UUID
    # The day all the appointments below fall on (handy for availability assertions).
    day: date
    # Jane's upcoming, still-scheduled appointment — the canonical "cancellable" one.
    jane_scheduled_id: UUID
    jane_scheduled_at: datetime
    # A cancelled appointment in Jane's history — must NOT count as cancellable, and
    # its slot must be free again in availability.
    jane_cancelled_id: UUID
    jane_cancelled_at: datetime
    # A live (non-expired) hold owned by John — must be excluded from availability,
    # but must NOT count as a cancellable appointment.
    john_held_id: UUID
    john_held_at: datetime


async def _insert_patient(conn: asyncpg.Connection, p: dict) -> UUID:
    return await conn.fetchval(
        "INSERT INTO patients (full_name, date_of_birth, phone) VALUES ($1, $2, $3) RETURNING id",
        p["full_name"],
        p["date_of_birth"],
        p["phone"],
    )


async def _insert_appointment(
    conn: asyncpg.Connection,
    patient_id: UUID,
    starts_at: datetime,
    status: str,
    held_until: datetime | None = None,
) -> UUID:
    return await conn.fetchval(
        """
        INSERT INTO appointments (patient_id, starts_at, ends_at, status, held_until)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        """,
        patient_id,
        starts_at,
        starts_at + timedelta(hours=1),
        status,
        held_until,
    )


async def load_docket(conn: asyncpg.Connection) -> Docket:
    """Insert the canonical dataset into an (already-truncated) test DB.

    Three appointments on the same upcoming weekday exercise all three states:
    09:00 closed-over by nothing, 10:00 scheduled, 11:00 cancelled (free again),
    12:00 held.
    """
    jane_id = await _insert_patient(conn, JANE)
    john_id = await _insert_patient(conn, JOHN)

    scheduled_at = next_weekday_at(10)
    cancelled_at = next_weekday_at(11)
    held_at = next_weekday_at(12)
    # A generous hold window so the fixture's hold is reliably "active" for the test.
    held_until = datetime.now(TZ) + timedelta(hours=1)

    jane_scheduled_id = await _insert_appointment(conn, jane_id, scheduled_at, "scheduled")
    jane_cancelled_id = await _insert_appointment(conn, jane_id, cancelled_at, "cancelled")
    john_held_id = await _insert_appointment(conn, john_id, held_at, "held", held_until)

    return Docket(
        jane_id=jane_id,
        john_id=john_id,
        day=scheduled_at.date(),
        jane_scheduled_id=jane_scheduled_id,
        jane_scheduled_at=scheduled_at,
        jane_cancelled_id=jane_cancelled_id,
        jane_cancelled_at=cancelled_at,
        john_held_id=john_held_id,
        john_held_at=held_at,
    )
