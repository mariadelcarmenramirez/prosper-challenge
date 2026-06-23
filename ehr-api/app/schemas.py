"""Pydantic request/response models for the EHR API.

The request/response shape is designed to read like a real integration: patients
and appointments are identified by UUID, datetimes are timezone-aware ISO-8601.
"""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

# --- Patients ---------------------------------------------------------------


class CreatePatientRequest(BaseModel):
    # A patient is fully identified by name + date of birth + phone; all three
    # are required so the same person can always be found again.
    full_name: str
    date_of_birth: date
    phone: str
    email: str | None = None


class PatientResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    full_name: str
    date_of_birth: date
    phone: str
    email: str | None = None


# --- Appointments -----------------------------------------------------------


class CreateAppointmentRequest(BaseModel):
    """Book a slot for a patient. ``starts_at`` is interpreted as clinic-local
    (Europe/Madrid) time when no timezone offset is supplied."""

    patient_id: UUID
    starts_at: datetime


class AppointmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    patient_id: UUID
    starts_at: datetime
    ends_at: datetime
    status: str
    held_until: datetime | None = None


class SlotResponse(BaseModel):
    """A free, bookable slot. Only available slots are ever returned, so there
    is no ``is_booked`` field to keep in sync."""

    starts_at: datetime
    ends_at: datetime


class CancelAppointmentResponse(BaseModel):
    id: UUID
    status: str
