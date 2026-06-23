"""Seed a little demo data so availability visibly has a taken slot.

Run once (with Postgres up and the schema applied):
    cd ehr-api && uv run python -m app.seed
"""

import asyncio
from datetime import datetime, time, timedelta

from app.database import acquire, close_pool, init_pool
from app.routers.appointments import TZ

DEMO_PATIENTS = [
    ("Jane Doe", "1990-01-01", "+34600000001", "jane@example.com"),
    ("John Smith", "1985-05-20", "+34600000002", None),
]


def next_business_day_at(hour: int) -> datetime:
    """Next weekday (after today) at the given clinic-local hour."""
    day = datetime.now(TZ).date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime.combine(day, time(hour), tzinfo=TZ)


async def main() -> None:
    await init_pool()
    try:
        async with acquire() as conn:
            for full_name, dob, phone, email in DEMO_PATIENTS:
                await conn.execute(
                    """
                    INSERT INTO patients (full_name, date_of_birth, phone, email)
                    VALUES ($1, $2::date, $3, $4)
                    ON CONFLICT (full_name, date_of_birth, phone) DO NOTHING
                    """,
                    full_name,
                    dob,
                    phone,
                    email,
                )
            patient_id = await conn.fetchval(
                "SELECT id FROM patients WHERE full_name = $1", DEMO_PATIENTS[0][0]
            )
            starts_at = next_business_day_at(10)
            await conn.execute(
                """
                INSERT INTO appointments (patient_id, starts_at, ends_at, status)
                VALUES ($1, $2, $3, 'scheduled')
                ON CONFLICT (starts_at) WHERE status = 'scheduled' DO NOTHING
                """,
                patient_id,
                starts_at,
                starts_at + timedelta(hours=1),
            )
            print(f"Seeded {len(DEMO_PATIENTS)} patients; booked {DEMO_PATIENTS[0][0]} at {starts_at:%Y-%m-%d %H:%M %Z}")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
