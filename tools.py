"""LLM tool (function-calling) definitions.

One ``FunctionSchema`` per EHR operation. Descriptions tell the model exactly
when to call each tool and what argument formats to use, since the model is the
one resolving the caller's words ("this Friday at 3pm") into concrete values.
"""

from pipecat.adapters.schemas.function_schema import FunctionSchema

_DATE = "Calendar date as YYYY-MM-DD."
_DATETIME = (
    "Appointment start as YYYY-MM-DDTHH:MM:SS in 24-hour clinic-local time "
    "(e.g. 2026-07-06T15:00:00 for 3pm). Always on the hour."
)

find_patient = FunctionSchema(
    name="find_patient",
    description=(
        "Look up an existing patient by their full identity. Call this first, "
        "after collecting the caller's full name, date of birth and phone number."
    ),
    properties={
        "full_name": {"type": "string", "description": "Caller's full name."},
        "date_of_birth": {"type": "string", "description": _DATE},
        "phone": {"type": "string", "description": "Caller's phone number."},
    },
    required=["full_name", "date_of_birth", "phone"],
)

create_patient = FunctionSchema(
    name="create_patient",
    description=(
        "Register a new patient when find_patient did not find them. Uses the "
        "name, date of birth and phone you already collected."
    ),
    properties={
        "full_name": {"type": "string", "description": "Caller's full name."},
        "date_of_birth": {"type": "string", "description": _DATE},
        "phone": {"type": "string", "description": "Caller's phone number."},
        "email": {"type": "string", "description": "Optional email address."},
    },
    required=["full_name", "date_of_birth", "phone"],
)

list_availability_slots = FunctionSchema(
    name="list_availability_slots",
    description=(
        "Get the clinic's free appointment slots for a single day or a date "
        "range. Use this to check which of the caller's preferred times are "
        "actually available before proposing one."
    ),
    properties={
        "date": {"type": "string", "description": f"Single day. {_DATE}"},
        "start": {"type": "string", "description": f"Range start. {_DATE}"},
        "end": {"type": "string", "description": f"Range end. {_DATE}"},
    },
    required=[],
)

list_patient_appointments = FunctionSchema(
    name="list_patient_appointments",
    description=(
        "List a patient's upcoming, still-scheduled appointments. Call this in "
        "the cancel flow to find which appointment the caller means."
    ),
    properties={
        "patient_id": {"type": "string", "description": "The patient's id from find/create."},
    },
    required=["patient_id"],
)

create_appointment = FunctionSchema(
    name="create_appointment",
    description=(
        "Reserve (hold) a slot for the patient the moment you propose it. This "
        "does NOT finalize the booking — call confirm_appointment once the "
        "caller agrees. Returns an appointment id with status 'held'."
    ),
    properties={
        "patient_id": {"type": "string", "description": "The patient's id."},
        "starts_at": {"type": "string", "description": _DATETIME},
    },
    required=["patient_id", "starts_at"],
)

confirm_appointment = FunctionSchema(
    name="confirm_appointment",
    description=(
        "Finalize a held appointment once the caller confirms the proposed "
        "slot. Promotes the appointment from 'held' to 'scheduled'."
    ),
    properties={
        "appointment_id": {"type": "string", "description": "Id returned by create_appointment."},
    },
    required=["appointment_id"],
)

cancel_appointment = FunctionSchema(
    name="cancel_appointment",
    description=(
        "Cancel an appointment by id (soft delete). Use it to cancel a real "
        "booking the caller asks to cancel, or to release a held slot the "
        "caller rejected."
    ),
    properties={
        "appointment_id": {"type": "string", "description": "The appointment id to cancel."},
    },
    required=["appointment_id"],
)

TOOL_SCHEMAS: list[FunctionSchema] = [
    find_patient,
    create_patient,
    list_availability_slots,
    list_patient_appointments,
    create_appointment,
    confirm_appointment,
    cancel_appointment,
]
