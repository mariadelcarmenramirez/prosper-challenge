"""Integration tests for availability + the booking lifecycle."""

from datetime import datetime, time, timedelta

import pytest_asyncio
from app.routers.appointments import TZ, now

PATIENT = {"full_name": "Booker One", "date_of_birth": "1980-02-02", "phone": "+34611111111"}


def future_slot(hour: int = 10, offset_days: int = 2) -> datetime:
    """A future clinic-hour slot start (next weekday >= offset_days out)."""
    day = now().date() + timedelta(days=offset_days)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


@pytest_asyncio.fixture
async def patient(client):
    return (await client.post("/patients", json=PATIENT)).json()


async def _hold(client, patient_id, starts_at):
    return await client.post(
        "/appointments", json={"patient_id": patient_id, "starts_at": starts_at.isoformat()}
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
    past = (now() - timedelta(days=7)).replace(minute=0, second=0, microsecond=0)
    r = await _hold(client, patient["id"], past)
    assert r.status_code == 422


async def test_create_rejects_out_of_hours_returns_422(client, patient):
    slot = future_slot(hour=20)  # after 18:00
    r = await _hold(client, patient["id"], slot)
    assert r.status_code == 422


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
