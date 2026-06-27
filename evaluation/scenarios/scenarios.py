"""The edge-case scenario suite.

Each :class:`Scenario` is a self-contained, controlled experiment: it seeds a known
starting state, hands the simulated caller a persona + goal, and then judges the
finished call with a deterministic oracle. The set covers the cases called out for
this challenge — book the exact day/hour, cancel something that doesn't exist, two
patients with the same name + date of birth but different phones, the empty-
availability loop terminating via the call guard — plus the other failure modes a
scheduling agent must get right (confirm-before-mutate, invalid-time rejection,
slot already taken, misheard details).

The same suite runs against both architectures and all models, so a single
table can compare accuracy across the matrix.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import asyncpg

from evaluation.harness.trace import ConversationTrace

from . import db
from .oracle import (
    OracleResult,
    called_before,
    count_tool,
    results_of,
)

SetupFn = Callable[[asyncpg.Connection], Awaitable[dict]]
PersonaFn = Callable[[dict], str]
OracleFn = Callable[[asyncpg.Connection, ConversationTrace, dict], Awaitable[OracleResult]]


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    setup: SetupFn
    persona: PersonaFn
    oracle: OracleFn


# A couple of identities reused across scenarios.
JANE = ("Jane Doe", "1990-01-01", "+34600000001")
JOHN = ("John Smith", "1985-05-20", "+34600000002")


def _identity_block(name: str, dob: str, phone: str) -> str:
    # DOB spoken naturally; the agent re-asks it as YYYY-MM-DD internally.
    return (
        f"Your name is {name}. Your date of birth is {dob}. Your phone number is {phone}. "
        "Give these clearly when asked."
    )


# ---------------------------------------------------------------------------
# 1. New patient registers and books an exact slot.
# ---------------------------------------------------------------------------


async def _setup_empty(conn: asyncpg.Connection) -> dict:
    slot = db.next_weekday_at(15, days_ahead=2)  # 3 PM, a clear weekday
    return {"slot": slot}


def _persona_book_new(ctx: dict) -> str:
    name, dob, phone = ("Maria Garcia", "1992-04-15", "+34611111111")
    return (
        f"You are a NEW patient who has never called before. {_identity_block(name, dob, phone)} "
        f"GOAL: book an appointment for {db.human(ctx['slot'])}. Accept the first slot the "
        "assistant offers at that time and confirm it clearly with a 'yes'."
    )


async def _oracle_book_new(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    name, dob = "Maria Garcia", "1992-04-15"
    booked = await db.scheduled_at(conn, ctx["slot"])
    r.add("booked_exact_slot", bool(booked), detail=f"scheduled rows at slot: {len(booked)}")
    r.add("patient_registered_once", await db.patient_count(conn, name, dob) == 1)
    r.add(
        "validated_before_lookup",
        called_before(trace, "confirm_patient_data", "find_patient"),
    )
    r.add("registered_new", count_tool(trace, "create_patient") >= 1, hard=False)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 2. Existing patient books (find path, no duplicate created).
# ---------------------------------------------------------------------------


async def _setup_jane(conn: asyncpg.Connection) -> dict:
    jane_id = await db.insert_patient(conn, *JANE)
    slot = db.next_weekday_at(14, days_ahead=2)
    return {"jane_id": jane_id, "slot": slot}


def _persona_book_existing(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: book an appointment for {db.human(ctx['slot'])}. Accept that slot and confirm."
    )


async def _oracle_book_existing(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    booked = await db.scheduled_at(conn, ctx["slot"])
    r.add("booked_exact_slot", bool(booked))
    if booked:
        r.add("booked_under_existing_patient", booked[0]["patient_id"] == ctx["jane_id"])
    r.add("no_duplicate_patient", await db.patient_count(conn, JANE[0], JANE[1]) == 1)
    r.add("did_not_register_existing", count_tool(trace, "create_patient") == 0, hard=False)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 3. Cancel an existing, scheduled appointment (only after explicit yes).
# ---------------------------------------------------------------------------


async def _setup_jane_with_appt(conn: asyncpg.Connection) -> dict:
    jane_id = await db.insert_patient(conn, *JANE)
    slot = db.next_weekday_at(10, days_ahead=2)
    appt_id = await db.insert_appointment(conn, jane_id, slot, "scheduled")
    return {"jane_id": jane_id, "appt_id": appt_id, "slot": slot}


def _persona_cancel_existing(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: cancel your appointment on {db.human(ctx['slot'])}. When the assistant reads it "
        "back and asks to confirm cancellation, say yes."
    )


async def _oracle_cancel_existing(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    status = await db.appointment_status(conn, ctx["appt_id"])
    r.add("appointment_cancelled", status == "cancelled", detail=f"status={status}")
    r.add("listed_before_cancel", called_before(trace, "list_patient_appointments", "cancel_appointment"))
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 4. Cancel an appointment that does not exist.
# ---------------------------------------------------------------------------


def _persona_cancel_nonexistent(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        "GOAL: cancel 'your appointment' — but you actually have none booked. Insist you think "
        "you have one. Accept it politely if told there's nothing to cancel."
    )


async def _oracle_cancel_nonexistent(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    scheduled = await db.all_scheduled(conn)
    r.add("nothing_cancelled_or_created", len(scheduled) == 0, detail=f"scheduled={len(scheduled)}")
    # With no appointments on file, a well-behaved agent never calls cancel at all.
    r.add("no_blind_cancel", count_tool(trace, "cancel_appointment") == 0)
    r.add("graceful_end", trace.error is None and trace.end_reason != "max_turns", hard=False)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 5. Two patients: same name + DOB, different phone -> disambiguate by phone.
# ---------------------------------------------------------------------------


async def _setup_twins(conn: asyncpg.Connection) -> dict:
    name, dob = "Carlos Ruiz", "1980-02-02"
    a_id = await db.insert_patient(conn, name, dob, "+34620000001")
    b_id = await db.insert_patient(conn, name, dob, "+34620000002")  # the caller
    slot = db.next_weekday_at(16, days_ahead=2)
    return {"name": name, "dob": dob, "a_id": a_id, "b_id": b_id, "phone_b": "+34620000002", "slot": slot}


def _persona_twins(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(ctx['name'], ctx['dob'], ctx['phone_b'])} "
        f"There is another patient with your exact name and birth date, but your phone is the way "
        f"to tell you apart. GOAL: book an appointment for {db.human(ctx['slot'])}. Confirm it."
    )


async def _oracle_twins(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    booked = await db.scheduled_at(conn, ctx["slot"])
    r.add("booked_exact_slot", bool(booked))
    if booked:
        r.add("booked_under_correct_patient", booked[0]["patient_id"] == ctx["b_id"],
              detail="must be patient B (matching phone), not the namesake")
    r.add("no_duplicate_patient", await db.patient_count(conn, ctx["name"], ctx["dob"]) == 2)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 6. Empty-availability loop -> guard ends the call after the cap.
# ---------------------------------------------------------------------------


def _persona_empty_avail(ctx: dict) -> str:
    sat = db.human(db.next_saturday())
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: book an appointment, but you can ONLY come on weekends — start by asking for "
        f"{sat}. The clinic is closed weekends, so whenever told there's nothing, insist on "
        "another Saturday or Sunday. Never accept a weekday. Keep trying."
    )


async def _oracle_empty_avail(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    r.add("no_booking_made", len(await db.all_scheduled(conn)) == 0)
    r.add("ended_without_error", trace.error is None and trace.end_reason != "max_turns")
    # Soft: did the loop-guard's no_availability stop actually fire?
    r.add("guard_no_availability_fired", trace.end_reason == "stop:no_availability", hard=False,
          detail=f"end_reason={trace.end_reason}")
    return r


# ---------------------------------------------------------------------------
# 7. Reject every offer -> guard ends after MAX_REJECTED_OFFERS.
# ---------------------------------------------------------------------------


def _persona_reject_all(ctx: dict) -> str:
    day = db.human(db.next_weekday_at(9, days_ahead=2))
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: you want something on {day} but you are extremely picky: whatever specific time "
        "the assistant offers, say no and ask for a different time, every single time. Never "
        "accept any offer. Keep rejecting until the assistant gives up."
    )


async def _oracle_reject_all(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    r.add("no_booking_made", len(await db.all_scheduled(conn)) == 0)
    r.add("ended_without_error", trace.error is None and trace.end_reason != "max_turns")
    r.add("guard_too_many_rejections_fired", trace.end_reason == "stop:too_many_rejections",
          hard=False, detail=f"end_reason={trace.end_reason}")
    return r


# ---------------------------------------------------------------------------
# 8. Invalid time (weekend/past) rejected by server, then a valid booking.
# ---------------------------------------------------------------------------


def _persona_invalid_then_valid(ctx: dict) -> str:
    bad = db.human(db.next_saturday())
    good = db.human(ctx["slot"])
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: first ask to book {bad}. If told that's not possible, then ask for {good} and "
        "accept that, confirming clearly."
    )


async def _oracle_invalid_then_valid(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    booked = await db.scheduled_at(conn, ctx["slot"])
    r.add("booked_valid_slot", bool(booked))
    # No scheduled appointment should ever land on a weekend.
    weekend = [a for a in await db.all_scheduled(conn) if a["starts_at"].astimezone(db.TZ).weekday() >= 5]
    r.add("no_weekend_booking", len(weekend) == 0)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 9. Requested slot already taken -> agent offers an alternative.
# ---------------------------------------------------------------------------


async def _setup_taken_slot(conn: asyncpg.Connection) -> dict:
    jane_id = await db.insert_patient(conn, *JANE)
    john_id = await db.insert_patient(conn, *JOHN)
    taken = db.next_weekday_at(10, days_ahead=2)
    await db.insert_appointment(conn, john_id, taken, "scheduled")  # John owns 10:00
    alt = db.next_weekday_at(11, days_ahead=2)
    return {"jane_id": jane_id, "john_id": john_id, "taken": taken, "alt": alt}


def _persona_taken_slot(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: you want {db.human(ctx['taken'])}. If that exact time is unavailable, you are "
        f"happy to take {db.human(ctx['alt'])} instead. Confirm whatever free slot you're offered."
    )


async def _oracle_taken_slot(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    taken_rows = await db.scheduled_at(conn, ctx["taken"])
    r.add("no_double_booking", len(taken_rows) == 1 and taken_rows[0]["patient_id"] == ctx["john_id"])
    jane_appts = [a for a in await db.appointments_for(conn, ctx["jane_id"]) if a["status"] == "scheduled"]
    r.add("jane_booked_alternative", bool(jane_appts))
    if jane_appts:
        r.add("alternative_not_taken_slot",
              all(a["starts_at"].astimezone(db.TZ) != ctx["taken"] for a in jane_appts))
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 10. Confirm-before-mutate: caller changes their mind, nothing is cancelled.
# ---------------------------------------------------------------------------


def _persona_change_mind(ctx: dict) -> str:
    return (
        f"You are an existing patient. {_identity_block(*JANE)} "
        f"GOAL: you call thinking about cancelling your {db.human(ctx['slot'])} appointment, but "
        "when the assistant asks you to confirm the cancellation, you CHANGE YOUR MIND and say "
        "no, keep it. Then say goodbye."
    )


async def _oracle_change_mind(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    status = await db.appointment_status(conn, ctx["appt_id"])
    r.add("appointment_still_scheduled", status == "scheduled", detail=f"status={status}")
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------
# 11. Misheard date of birth -> validation fails, caller corrects, then books.
# ---------------------------------------------------------------------------


def _persona_misheard_dob(ctx: dict) -> str:
    name, good_dob, phone = ("Lucia Fernandez", "1995-07-09", "+34633333333")
    return (
        f"You are a NEW patient. Your name is {name}, your phone is {phone}. "
        f"When first asked for your date of birth, say it WRONG as 'the 30th of February 1995' "
        f"(an impossible date). Only after the assistant says that date isn't valid, correct "
        f"yourself to the 9th of July 1995 (your real date of birth is {good_dob}). "
        f"GOAL: then book {db.human(ctx['slot'])} and confirm."
    )


async def _oracle_misheard_dob(conn, trace, ctx) -> OracleResult:
    r = OracleResult()
    name, dob = "Lucia Fernandez", "1995-07-09"
    r.add("registered_with_correct_dob", await db.patient_count(conn, name, dob) == 1)
    validations = results_of(trace, "confirm_patient_data")
    had_invalid = any(isinstance(v, dict) and v.get("valid") is False for v in validations)
    had_valid = any(isinstance(v, dict) and v.get("valid") is True for v in validations)
    r.add("revalidated_after_correction", had_invalid and had_valid,
          detail=f"validations={len(validations)}")
    r.add("booked", bool(await db.scheduled_at(conn, ctx["slot"])), hard=False)
    r.add("no_error", trace.error is None)
    return r


# ---------------------------------------------------------------------------


SCENARIOS: list[Scenario] = [
    Scenario("book_new_patient", "New patient registers and books an exact slot",
             _setup_empty, _persona_book_new, _oracle_book_new),
    Scenario("book_existing_patient", "Existing patient books; no duplicate created",
             _setup_jane, _persona_book_existing, _oracle_book_existing),
    Scenario("cancel_existing", "Cancel a real scheduled appointment after confirmation",
             _setup_jane_with_appt, _persona_cancel_existing, _oracle_cancel_existing),
    Scenario("cancel_nonexistent", "Ask to cancel when nothing is booked",
             _setup_jane, _persona_cancel_nonexistent, _oracle_cancel_nonexistent),
    Scenario("same_name_diff_phone", "Disambiguate two same-name/DOB patients by phone",
             _setup_twins, _persona_twins, _oracle_twins),
    Scenario("empty_availability_loop", "No availability for weeks -> guard ends the call",
             _setup_jane, _persona_empty_avail, _oracle_empty_avail),
    Scenario("reject_offers_loop", "Caller rejects every offer -> guard ends the call",
             _setup_jane, _persona_reject_all, _oracle_reject_all),
    Scenario("invalid_then_valid_time", "Weekend request rejected, then a valid booking",
             _setup_jane, _persona_invalid_then_valid, _oracle_invalid_then_valid),
    Scenario("slot_already_taken", "Requested slot taken -> agent offers an alternative",
             _setup_taken_slot, _persona_taken_slot, _oracle_taken_slot),
    Scenario("confirm_before_cancel", "Caller changes mind -> nothing is cancelled",
             _setup_jane_with_appt, _persona_change_mind, _oracle_change_mind),
    Scenario("misheard_dob", "Invalid DOB rejected, corrected, then booked",
             _setup_empty, _persona_misheard_dob, _oracle_misheard_dob),
]
