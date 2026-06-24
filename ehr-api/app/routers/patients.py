"""Patient endpoints: validate identity, register, and look up."""

import re
from datetime import date, datetime

import asyncpg
from fastapi import APIRouter, HTTPException, Query, status

from app.database import acquire
from app.routers.appointments import TZ  # clinic timezone, single source of truth
from app.schemas import (
    ConfirmPatientDataRequest,
    ConfirmPatientDataResponse,
    CreatePatientRequest,
    PatientResponse,
)

router = APIRouter(tags=["patients"])

# Identity-validation bounds. A patient is matched on the exact (name, DOB, phone)
# triple, so normalizing here is what keeps "+34 600 000 000" and "+34600000000"
# (or "  Jane  Doe ") from missing the real record or minting a duplicate.
MIN_DOB_YEAR = 1900
PHONE_MIN_DIGITS = 7
PHONE_MAX_DIGITS = 15  # E.164 maximum


def normalize_name(raw: str) -> str:
    """Trim and collapse internal whitespace so the same person resolves to the
    same stored row (``"  Jane   Doe "`` -> ``"Jane Doe"``)."""
    return " ".join(raw.split())


def normalize_phone(raw: str) -> str:
    """Keep an optional leading ``+`` and drop every other non-digit, so spoken
    forms like ``"+34 600 000 000"`` collapse to one canonical string."""
    raw = raw.strip()
    digits = re.sub(r"\D", "", raw)
    return f"+{digits}" if raw.startswith("+") else digits


def _parse_dob(raw: str) -> date | None:
    """Parse an ISO ``YYYY-MM-DD`` date, or ``None`` if it isn't a valid one."""
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


@router.post("/patients/validate", response_model=ConfirmPatientDataResponse)
async def confirm_patient_data(body: ConfirmPatientDataRequest) -> ConfirmPatientDataResponse:
    """Validate and normalize a caller's identity *before* any lookup or insert.

    This is the agent's safety check: it confirms the name, date of birth and
    phone are well-formed and returns canonical values, so find/create never run
    on a typo'd date or a differently-formatted phone — which would otherwise
    miss the real patient or create a duplicate. It never touches the database,
    and always returns 200 (bad caller data is a normal outcome, not an error):
    the agent branches on ``valid``.
    """
    issues: list[str] = []

    full_name = normalize_name(body.full_name)
    if len(full_name) < 2:
        issues.append("The full name is missing or too short.")

    phone = normalize_phone(body.phone)
    if not (PHONE_MIN_DIGITS <= len(phone.lstrip("+")) <= PHONE_MAX_DIGITS):
        issues.append("The phone number doesn't look complete.")

    dob = _parse_dob(body.date_of_birth)
    dob_iso: str | None = None
    if dob is None:
        issues.append("The date of birth isn't a valid calendar date.")
    elif dob > datetime.now(TZ).date():
        issues.append("The date of birth is in the future.")
    elif dob.year < MIN_DOB_YEAR:
        issues.append("The date of birth is too far in the past.")
    else:
        dob_iso = dob.isoformat()

    valid = not issues
    return ConfirmPatientDataResponse(
        valid=valid,
        full_name=full_name if valid else None,
        date_of_birth=dob_iso if valid else None,
        phone=phone if valid else None,
        issues=issues,
    )


@router.post("/patients", response_model=PatientResponse, status_code=status.HTTP_201_CREATED)
async def create_patient(body: CreatePatientRequest) -> PatientResponse:
    """Register a new patient. The DB generates the patient_id (UUID).

    A patient is uniquely identified by (full_name, date_of_birth, phone); a
    duplicate is rejected with 409 so the agent re-uses the existing record.
    """
    async with acquire() as conn:
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO patients (full_name, date_of_birth, phone)
                VALUES ($1, $2, $3)
                RETURNING id, full_name, date_of_birth, phone
                """,
                body.full_name,
                body.date_of_birth,
                body.phone,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A patient with that name, date of birth and phone already exists.",
            )
    return PatientResponse(**dict(row))


@router.get("/patients/find", response_model=PatientResponse)
async def find_patient(
    full_name: str = Query(...),
    date_of_birth: date = Query(...),
    phone: str = Query(...),
) -> PatientResponse:
    """Look up a patient by their full identity (name + DOB + phone).

    Returns 404 if not found so the agent knows to register them.
    """
    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, full_name, date_of_birth, phone
            FROM patients
            WHERE full_name = $1 AND date_of_birth = $2 AND phone = $3
            """,
            full_name,
            date_of_birth,
            phone,
        )
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    return PatientResponse(**dict(row))
