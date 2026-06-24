"""Appointment endpoints + clinic-hours / timezone helpers.

Availability is derived (no slots table): for any day/range we enumerate the
clinic's hourly slots and remove the ones already taken by a scheduled or a
non-expired held appointment, plus anything in the past.

Booking is two-phase:
  POST /appointments              -> creates a HELD row (reserves the slot)
  POST /appointments/{id}/confirm -> promotes held -> scheduled
  DELETE /appointments/{id}       -> soft delete (-> cancelled)
"""

from datetime import date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg
from fastapi import APIRouter, HTTPException, Query, status

from app.database import acquire
from app.schemas import (
    AppointmentResponse,
    CancelAppointmentResponse,
    CreateAppointmentRequest,
    SlotResponse,
)

router = APIRouter(tags=["appointments"])

# --- Clinic configuration ---------------------------------------------------
TZ = ZoneInfo("Europe/Madrid")          # clinic wall-clock timezone
CLINIC_OPEN_HOUR = 9                     # first bookable start
CLINIC_LAST_START = 17                   # last bookable start (17:00-18:00)
APPOINTMENT_HOURS = 1                    # appointments last 1hour
HOLD_TTL_MINUTES = 5                     # how long a proposed slot stays reserved


def now() -> datetime:
    """Current clinic-local time (timezone-aware)."""
    return datetime.now(TZ)


def normalize(dt: datetime) -> datetime:
    """Interpret a naive datetime as clinic-local; leave aware ones untouched."""
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt


def clinic_slots(start_day: date, end_day: date) -> list[datetime]:
    """All clinic-hour slot starts (aware) for [start_day, end_day] inclusive."""
    slots: list[datetime] = []
    day = start_day
    while day <= end_day:
        if day.weekday() < 5:  # Mon-Fri
            for hour in range(CLINIC_OPEN_HOUR, CLINIC_LAST_START + 1):
                slots.append(datetime.combine(day, time(hour), tzinfo=TZ))
        day += timedelta(days=1)
    return slots


def validate_bookable(starts_at: datetime) -> datetime:
    """Raise 422 unless starts_at is a future, on-the-hour, in-clinic-hours slot."""
    starts_at = normalize(starts_at)
    local = starts_at.astimezone(TZ)
    if (local.minute, local.second, local.microsecond) != (0, 0, 0):
        raise HTTPException(422, "Appointments start on the hour.")
    if local.weekday() >= 5:
        raise HTTPException(422, "The clinic is closed at weekends.")
    if not (CLINIC_OPEN_HOUR <= local.hour <= CLINIC_LAST_START):
        raise HTTPException(422, "Outside clinic hours (Mon-Fri 09:00-18:00).")
    if starts_at <= now():
        raise HTTPException(422, "Cannot book a slot in the past.")
    return starts_at


def _appt(row: asyncpg.Record) -> AppointmentResponse:
    return AppointmentResponse(**dict(row))


# --- Availability -----------------------------------------------------------


@router.get("/slots", response_model=list[SlotResponse])
async def list_availability_slots(
    date: date | None = Query(None, description="Single day (YYYY-MM-DD)"),
    start: date | None = Query(None, description="Range start (YYYY-MM-DD)"),
    end: date | None = Query(None, description="Range end (YYYY-MM-DD)"),
) -> list[SlotResponse]:
    """Free, bookable slots for a day or range. Defaults to today if nothing given.

    A slot is excluded when it is in the past, or when an appointment that is
    `scheduled` or `held` (and not yet expired) already occupies it.
    """
    if date is not None:
        start_day = end_day = date
    else:
        start_day = start or now().date()
        end_day = end or start_day
    if end_day < start_day:
        raise HTTPException(422, "`end` must not be before `start`.")

    range_start = datetime.combine(start_day, time(0), tzinfo=TZ)
    range_end = datetime.combine(end_day + timedelta(days=1), time(0), tzinfo=TZ)

    async with acquire() as conn:
        taken_rows = await conn.fetch(
            """
            SELECT starts_at FROM appointments
            WHERE starts_at >= $1 AND starts_at < $2
              AND (status = 'scheduled'
                   OR (status = 'held' AND held_until > now()))
            """,
            range_start,
            range_end,
        )
    taken = {r["starts_at"].astimezone(TZ) for r in taken_rows}

    current = now()
    free = [
        s
        for s in clinic_slots(start_day, end_day)
        if s > current and s not in taken
    ]
    return [SlotResponse(starts_at=s, ends_at=s + timedelta(hours=APPOINTMENT_HOURS)) for s in free]


# --- Patient's appointments (drives the cancel flow) ------------------------


@router.get("/appointments", response_model=list[AppointmentResponse])
async def list_patient_appointments(patient_id: UUID = Query(...)) -> list[AppointmentResponse]:
    """A patient's cancellable appointments: scheduled and in the future only."""
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, patient_id, starts_at, ends_at, status, held_until
            FROM appointments
            WHERE patient_id = $1 AND status = 'scheduled' AND starts_at > now()
            ORDER BY starts_at
            """,
            patient_id,
        )
    return [_appt(r) for r in rows]


# --- Booking lifecycle ------------------------------------------------------


@router.post(
    "/appointments", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED
)
async def create_appointment(body: CreateAppointmentRequest) -> AppointmentResponse:
    """Reserve a slot for a patient: insert a HELD row that expires in HOLD_TTL.

    The agent calls this the moment it proposes a slot, so nobody else can take
    it during confirmation. 409 if the slot is already taken.
    """
    starts_at = validate_bookable(body.starts_at)
    ends_at = starts_at + timedelta(hours=APPOINTMENT_HOURS)
    held_until = now() + timedelta(minutes=HOLD_TTL_MINUTES)

    async with acquire() as conn:
        async with conn.transaction():
            clash = await conn.fetchval(
                """
                SELECT 1 FROM appointments
                WHERE starts_at = $1
                  AND (status = 'scheduled'
                       OR (status = 'held' AND held_until > now()))
                LIMIT 1
                """,
                starts_at,
            )
            if clash:
                raise HTTPException(status.HTTP_409_CONFLICT, "That slot is no longer available.")
            row = await conn.fetchrow(
                """
                INSERT INTO appointments (patient_id, starts_at, ends_at, status, held_until)
                VALUES ($1, $2, $3, 'held', $4)
                RETURNING id, patient_id, starts_at, ends_at, status, held_until
                """,
                body.patient_id,
                starts_at,
                ends_at,
                held_until,
            )
    return _appt(row)


@router.post("/appointments/{appointment_id}/confirm", response_model=AppointmentResponse)
async def confirm_appointment(appointment_id: UUID) -> AppointmentResponse:
    """Finalize a hold: held -> scheduled, re-checking it hasn't expired, isn't
    in the past, and the slot is still free."""
    async with acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                SELECT id, patient_id, starts_at, ends_at, status, held_until
                FROM appointments WHERE id = $1 FOR UPDATE
                """,
                appointment_id,
            )
            if row is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found.")
            if row["status"] == "scheduled":
                return _appt(row)  # idempotent
            if row["status"] == "cancelled":
                raise HTTPException(status.HTTP_409_CONFLICT, "Appointment was cancelled.")
            if row["held_until"] is None or row["held_until"] <= now():
                raise HTTPException(status.HTTP_409_CONFLICT, "The hold on that slot expired.")
            if row["starts_at"] <= now():
                raise HTTPException(status.HTTP_409_CONFLICT, "That slot is in the past.")
            try:
                confirmed = await conn.fetchrow(
                    """
                    UPDATE appointments
                    SET status = 'scheduled', held_until = NULL
                    WHERE id = $1
                    RETURNING id, patient_id, starts_at, ends_at, status, held_until
                    """,
                    appointment_id,
                )
            except asyncpg.UniqueViolationError:
                raise HTTPException(status.HTTP_409_CONFLICT, "That slot was just booked.")
    return _appt(confirmed)


@router.delete("/appointments/{appointment_id}", response_model=CancelAppointmentResponse)
async def cancel_appointment(appointment_id: UUID) -> CancelAppointmentResponse:
    """Soft delete: flip status to cancelled (releases the slot). Works for both
    a held reservation and a confirmed booking; idempotent if already cancelled."""
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            UPDATE appointments SET status = 'cancelled'
            WHERE id = $1 AND status IN ('held', 'scheduled')
            RETURNING id, status
            """,
            appointment_id,
        )
        if row is None:
            existing = await conn.fetchval(
                "SELECT status FROM appointments WHERE id = $1", appointment_id
            )
            if existing is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "Appointment not found.")
            return CancelAppointmentResponse(id=appointment_id, status=existing)
    return CancelAppointmentResponse(**dict(row))
