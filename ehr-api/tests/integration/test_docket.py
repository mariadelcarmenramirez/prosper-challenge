"""Tests that exercise the canonical docket fixture.

These double as the demonstration of the stable-environment property: a test can
cancel Jane's appointment and the *next* test still finds it there, because the
data is reloaded fresh (truncate + ``load_docket``) before each test.
"""

from datetime import datetime, time, timedelta

from tests.fixtures import TZ


async def test_docket_jane_has_one_cancellable_appointment(client, docket):
    r = await client.get("/appointments", params={"patient_id": str(docket.jane_id)})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == str(docket.jane_scheduled_id)
    assert datetime.fromisoformat(rows[0]["starts_at"]) == docket.jane_scheduled_at


async def test_docket_john_has_no_cancellable_appointments(client, docket):
    """John's only appointment is a hold, which is not a cancellable booking."""
    r = await client.get("/appointments", params={"patient_id": str(docket.john_id)})
    assert r.status_code == 200
    assert r.json() == []


async def test_docket_availability_reflects_all_three_states(client, docket):
    slots = (await client.get("/slots", params={"date": docket.day.isoformat()})).json()
    starts = {datetime.fromisoformat(s["starts_at"]) for s in slots}

    at = lambda hour: datetime.combine(docket.day, time(hour), tzinfo=TZ)  # noqa: E731
    assert at(10) not in starts, "scheduled slot must be taken"
    assert at(12) not in starts, "held slot must be taken"
    assert at(11) in starts, "a cancelled-only slot must be free again"


async def _cancel_janes_appointment(client, docket):
    """The 'always there' appointment exists, gets cancelled, and is then gone —
    within a single test. Two tests run this independently to prove isolation."""
    before = (await client.get("/appointments", params={"patient_id": str(docket.jane_id)})).json()
    assert len(before) == 1  # present at the start, every time

    r = await client.delete(f"/appointments/{docket.jane_scheduled_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"

    after = (await client.get("/appointments", params={"patient_id": str(docket.jane_id)})).json()
    assert after == []


async def test_cancel_janes_appointment_first_run(client, docket):
    await _cancel_janes_appointment(client, docket)


async def test_cancel_janes_appointment_again_is_unaffected(client, docket):
    # Even though the previous test cancelled it, it's back: the fixture reloads.
    await _cancel_janes_appointment(client, docket)
