"""Integration tests for availability + the booking lifecycle."""

from datetime import datetime, time, timedelta
from uuid import UUID, uuid4

import pytest_asyncio
from app import database
from app.routers.appointments import TZ, now

PATIENT = {"full_name": "Booker One", "date_of_birth": "1980-02-02", "phone": "+34611111111"}


def future_slot(hour: int = 10, offset_days: int = 2) -> datetime:
    """A future clinic-hour slot start (next weekday >= offset_days out)."""
    day = now().date() + timedelta(days=offset_days)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


def next_weekend_slot(hour: int = 10) -> datetime:
    """A future Saturday at a clinic hour: a valid time on a closed day."""
    day = now().date() + timedelta(days=1)
    while day.weekday() != 5:  # Saturday
        day += timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


def past_slot(hour: int = 10) -> datetime:
    """A clinic-hour slot on the most recent past weekday: in the past but valid on
    every other axis, so validate_bookable reaches the past guard (its last check)
    instead of short-circuiting on weekend/out-of-hours like now()'s wall clock would."""
    day = now().date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


@pytest_asyncio.fixture
async def patient(client):
    return (await client.post("/patients", json=PATIENT)).json()


async def _hold(client, patient_id, starts_at):
    return await client.post(
        "/appointments", json={"patient_id": patient_id, "starts_at": starts_at.isoformat()}
    )


async def _expire_hold(appt_id: str) -> None:
    """Force a hold's TTL into the past, simulating its 5-minute window lapsing.
    The API never mints an expired hold, so we age it directly in the DB."""
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE appointments SET held_until = now() - interval '1 minute' WHERE id = $1",
            UUID(appt_id),
        )


async def _backdate_start(appt_id: str) -> None:
    """Move a still-live hold's start into the past so confirm reaches the
    past-slot guard rather than the expiry one (the API won't book a past slot)."""
    async with database.acquire() as conn:
        await conn.execute(
            "UPDATE appointments SET starts_at = now() - interval '1 hour', ends_at = now() "
            "WHERE id = $1",
            UUID(appt_id),
        )


async def test_list_slots_returns_available(client):
    slot = future_slot()
    r = await client.get("/slots", params={"date": slot.date().isoformat()})
    assert r.status_code == 200
    starts = {datetime.fromisoformat(s["starts_at"]) for s in r.json()}
    assert slot in starts


async def test_create_appointment_creates_hold(client, patient):
    slot = future_slot()
    r = await _hold(client, patient["id"], slot)
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "held"
    assert body["held_until"] is not None


async def test_held_slot_excluded_from_availability(client, patient):
    slot = future_slot(hour=11)
    await _hold(client, patient["id"], slot)
    r = await client.get("/slots", params={"date": slot.date().isoformat()})
    starts = {datetime.fromisoformat(s["starts_at"]) for s in r.json()}
    assert slot not in starts


async def test_confirm_books_slot(client, patient):
    slot = future_slot(hour=12)
    appt = (await _hold(client, patient["id"], slot)).json()
    r = await client.post(f"/appointments/{appt['id']}/confirm")
    assert r.status_code == 200
    assert r.json()["status"] == "scheduled"


async def test_double_book_returns_409(client, patient):
    slot = future_slot(hour=13)
    first = (await _hold(client, patient["id"], slot)).json()
    await client.post(f"/appointments/{first['id']}/confirm")
    r = await _hold(client, patient["id"], slot)
    assert r.status_code == 409


async def test_cancel_frees_slot(client, patient):
    slot = future_slot(hour=14)
    appt = (await _hold(client, patient["id"], slot)).json()
    await client.post(f"/appointments/{appt['id']}/confirm")

    r = await client.delete(f"/appointments/{appt['id']}")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    slots = (await client.get("/slots", params={"date": slot.date().isoformat()})).json()
    assert slot in {datetime.fromisoformat(s["starts_at"]) for s in slots}


async def test_appointment_responses_use_clinic_local_time(client, patient):
    # Regression: read-side endpoints must report the same wall-clock hour as
    # /slots. asyncpg returns TIMESTAMPTZ in UTC, so without normalization a
    # 16:00 booking reads back as 14:00 and the agent shows the wrong time.
    slot = future_slot(hour=16)
    created = (await _hold(client, patient["id"], slot)).json()
    confirmed = (await client.post(f"/appointments/{created['id']}/confirm")).json()
    listed = (await client.get("/appointments", params={"patient_id": patient["id"]})).json()[0]

    for body in (created, confirmed, listed):
        start = datetime.fromisoformat(body["starts_at"])
        assert start == slot                          # same instant
        assert start.astimezone(TZ).hour == 16        # same wall-clock hour as booked
        assert start.utcoffset() == slot.utcoffset()  # clinic-local offset, not UTC


async def test_create_rejects_past_returns_422(client, patient):
    r = await _hold(client, patient["id"], past_slot())
    assert r.status_code == 422
    assert "past" in r.json()["detail"]  # pin the past guard, not weekend/out-of-hours


async def test_create_rejects_out_of_hours_returns_422(client, patient):
    slot = future_slot(hour=20)  # after 18:00
    r = await _hold(client, patient["id"], slot)
    assert r.status_code == 422
    assert "clinic hours" in r.json()["detail"]


async def test_list_patient_appointments_future_scheduled_only(client, patient):
    booked = future_slot(hour=15)
    held = future_slot(hour=16)
    appt = (await _hold(client, patient["id"], booked)).json()
    await client.post(f"/appointments/{appt['id']}/confirm")
    await _hold(client, patient["id"], held)  # stays 'held'

    r = await client.get("/appointments", params={"patient_id": patient["id"]})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert datetime.fromisoformat(rows[0]["starts_at"]) == booked


# --- Error branches ---------------------------------------------------------


async def test_slots_end_before_start_returns_422(client):
    day = future_slot().date()
    r = await client.get(
        "/slots",
        params={"start": day.isoformat(), "end": (day - timedelta(days=1)).isoformat()},
    )
    assert r.status_code == 422
    assert "before" in r.json()["detail"]


async def test_create_rejects_off_the_hour_returns_422(client, patient):
    slot = future_slot(hour=10).replace(minute=30)
    r = await _hold(client, patient["id"], slot)
    assert r.status_code == 422
    assert "on the hour" in r.json()["detail"]


async def test_create_rejects_weekend_returns_422(client, patient):
    r = await _hold(client, patient["id"], next_weekend_slot())
    assert r.status_code == 422
    assert "weekend" in r.json()["detail"]


async def test_confirm_unknown_returns_404(client):
    r = await client.post(f"/appointments/{uuid4()}/confirm")
    assert r.status_code == 404


async def test_confirm_cancelled_returns_409(client, patient):
    appt = (await _hold(client, patient["id"], future_slot(hour=9))).json()
    await client.delete(f"/appointments/{appt['id']}")
    r = await client.post(f"/appointments/{appt['id']}/confirm")
    assert r.status_code == 409
    assert "cancelled" in r.json()["detail"]


async def test_confirm_expired_hold_returns_409(client, patient):
    appt = (await _hold(client, patient["id"], future_slot(hour=9))).json()
    await _expire_hold(appt["id"])
    r = await client.post(f"/appointments/{appt['id']}/confirm")
    assert r.status_code == 409
    assert "expired" in r.json()["detail"]


async def test_confirm_past_slot_returns_409(client, patient):
    # Hold is still live (not expired), but its start has slipped into the past.
    appt = (await _hold(client, patient["id"], future_slot(hour=9))).json()
    await _backdate_start(appt["id"])
    r = await client.post(f"/appointments/{appt['id']}/confirm")
    assert r.status_code == 409
    assert "past" in r.json()["detail"]  # the past guard, not the expiry one above


async def test_cancel_unknown_returns_404(client):
    r = await client.delete(f"/appointments/{uuid4()}")
    assert r.status_code == 404
