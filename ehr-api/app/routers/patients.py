"""Patient endpoints: register and look up."""

from datetime import date

import asyncpg
from fastapi import APIRouter, HTTPException, Query, status

from app.database import acquire
from app.schemas import CreatePatientRequest, PatientResponse

router = APIRouter(tags=["patients"])


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
                INSERT INTO patients (full_name, date_of_birth, phone, email)
                VALUES ($1, $2, $3, $4)
                RETURNING id, full_name, date_of_birth, phone, email
                """,
                body.full_name,
                body.date_of_birth,
                body.phone,
                body.email,
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
            SELECT id, full_name, date_of_birth, phone, email
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
