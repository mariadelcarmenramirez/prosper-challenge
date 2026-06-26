"""Phased-specialist ("sequential handoff") variant of the scheduling brain.

An alternative wiring to the single-context agent (``architectures/single.py``). The three are
A/B-tested side by side: ``bot.py`` picks between them with the ``AGENT_ARCH`` flag.
This module shares only the kernel (``guard`` + ``runtime``) and the prompt header
(``prompts``) with the other architectures; it never imports a sibling, so none of
the three depend on each other.

The idea is the idiomatic voice pattern: exactly one LLM is active at a time, and
the conversation moves through three phases — IDENTIFY -> (BOOK | CANCEL). Each
phase is a *specialist*: a focused system prompt plus a small subset of the tools.
A ``transfer_to_*`` tool is what advances the conversation: when the model calls
it, the handler swaps the context's system prompt and offered tool subset, so the
next inference runs as the new specialist. There is still one ``OpenAILLMService``
and one shared message history, so the patient_id discovered while identifying is
carried forward for free; for robustness we also capture it deterministically from
the EHR result into a per-call ``CallState`` and inject that into the next prompt,
rather than trusting the model to copy the id into its transfer arguments.

Phases and their tools:
* IDENTIFY  — confirm_patient_data, find_patient, create_patient, and the two
              transfer tools. Validates identity, finds-or-registers the caller,
              then asks intent and hands off.
* BOOK      — list_availability_slots, create_appointment, confirm_appointment,
              cancel_appointment. Runs the booking loop.
* CANCEL    — list_patient_appointments, cancel_appointment. Runs the cancel flow.

The two ``transfer_to_*`` tools are defined here, not in ``tool_schemas.py``:
they are specialist-only routing tools with no EHR implementation (they call no
endpoint — they only swap phase state), so keeping the whole second architecture
in one self-contained file makes it trivial to A/B against the single agent.

Loop-safety is identical to the single agent: we reuse ``CallGuard`` and the same
``make_handler`` from the shared ``guard``/``runtime`` kernel, so empty-availability
/ rejected-offer streaks and the global tool-call ceiling behave the same here. Transfer tools are bounded
by construction (once you leave IDENTIFY the transfer tools are no longer offered),
so they bypass the guard rather than risk aborting mid-handoff.

Known trade-off of the strict subsets: a caller who reaches CANCEL but has no
appointments cannot be booked in the same call, because the Canceller has no
booking tools and no transfer tool. That is the price of fully isolating phases;
the single-context agent (``architectures/single.py``) does not have this limitation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

# Build on the shared kernel (guard + runtime) and the shared prompt header, never
# on a sibling architecture, so this phase machine shares identical handler
# behaviour (find_patient None-normalizing, streak signals, the global circuit
# breaker) without coupling to a sibling architecture. ``build_llm`` is re-exported so bot.py can
# call this module exactly like it calls the other architectures.
from ..core.guard import CallGuard
from ..core.prompts import always_on_rules
from ..core.runtime import (  # noqa: F401  (build_llm re-exported)
    TOOL_HANDLERS,
    build_llm,
    make_handler,
)
from ..tools.schemas import (
    cancel_appointment,
    confirm_appointment,
    confirm_patient_data,
    create_appointment,
    create_patient,
    find_patient,
    list_availability_slots,
    list_patient_appointments,
)

# --- Transfer (handoff) tool schemas ---------------------------------------
#
# The only tools that move the conversation between phases. They still require the
# patient_id as a gate (the identifier cannot hand off before it believes it has
# one) and as a fallback, but the value actually carried forward is the one
# captured deterministically from the EHR result, not whatever the model passes.

transfer_to_booking = FunctionSchema(
    name="transfer_to_booking",
    description=(
        "Hand the call off to the booking specialist once the caller is "
        "identified AND has said they want to BOOK a new appointment. Pass the "
        "confirmed patient_id. Do not announce the transfer to the caller; it is "
        "seamless."
    ),
    properties={
        "patient_id": {"type": "string", "description": "The confirmed patient's id."},
    },
    required=["patient_id"],
)

transfer_to_cancellation = FunctionSchema(
    name="transfer_to_cancellation",
    description=(
        "Hand the call off to the cancellation specialist once the caller is "
        "identified AND has said they want to CANCEL an existing appointment. "
        "Pass the confirmed patient_id. Do not announce the transfer to the "
        "caller; it is seamless."
    ),
    properties={
        "patient_id": {"type": "string", "description": "The confirmed patient's id."},
    },
    required=["patient_id"],
)


# --- Phase prompts ----------------------------------------------------------


def build_identifier_prompt(
    now: datetime | None = None,
    patient_id: str | None = None,
    clinic_name: str = "Prosper Health",
) -> str:
    """IDENTIFY phase: validate identity, find-or-register, then route."""
    return always_on_rules(now, clinic_name) + """
YOUR ROLE RIGHT NOW — identify the caller, then route them. You can only identify and hand off; you \
have no booking or cancellation tools, so never promise to book or cancel yourself.

IDENTIFY THE CALLER
- Greet them, introduce yourself as the Prosper Health scheduling assistant, then collect three \
things: full name, date of birth, and phone number. Ask for whatever is missing.
- BEFORE any lookup, call confirm_patient_data(full_name, date_of_birth, phone) to validate and \
clean them up.
  - If it returns "valid: false", you misheard something: apologise lightly and re-ask only for the \
field named in "issues", then call confirm_patient_data again.
  - When it returns "valid: true", read the NORMALIZED name, date of birth and phone back to the \
caller in words and wait for an explicit yes. If they correct anything, call confirm_patient_data \
again.
- Only after they confirm, call find_patient(full_name, date_of_birth, phone) using the NORMALIZED \
values.
  - If it returns a patient, greet them by name.
  - If it returns "found: false", say "I don't have you in our system yet, let me register you," \
then call create_patient(full_name, date_of_birth, phone). Use the patient id it returns.

ROUTE THE CALLER
- Once you have a patient_id, ask whether they would like to BOOK a new appointment or CANCEL an \
existing one.
- If they want to book: call transfer_to_booking(patient_id) with that id.
- If they want to cancel: call transfer_to_cancellation(patient_id) with that id.
- The handoff is seamless — do not tell the caller you are transferring them; just keep talking \
naturally once the next step takes over.
"""


def build_booker_prompt(
    now: datetime | None = None,
    patient_id: str | None = None,
    clinic_name: str = "Prosper Health",
) -> str:
    """BOOK phase: the booking loop for an already-identified caller."""
    who = f"patient_id {patient_id}" if patient_id else "the identified caller"
    return always_on_rules(now, clinic_name) + f"""
YOUR ROLE RIGHT NOW — book a new appointment for an already-identified caller ({who}). Use that \
patient_id for every booking tool call. The caller is already verified; do not re-identify them.

BOOK FLOW
- Ask when they'd like to come in. They may give a specific day, a range of days, or a range of \
times — accept any of these.
- For EVERY time the caller mentions, repeat it back as an exact date to remove ambiguity \
("so Monday the 6th of July at 3 in the afternoon?") before treating it as a candidate.
- Call list_availability_slots for the relevant day or range to see what is actually free (it \
already excludes booked, held, past, and out-of-hours slots).
- Keep only the caller's candidate times that appear in the available list, earliest first.
  - If NONE are available: tell them those times are taken and ask for their availability AFTER \
the last time they gave you. Take the new times and check again. Do this for at most 4 rounds; if \
there is still nothing, apologise politely that there's no availability right now and end the call.
  - If at least one is available: take the EARLIEST and call create_appointment(patient_id, \
starts_at) to hold it. Then say "I can offer you {{date}} at {{time}}. Shall I book it?" and wait.
    - If they say yes: call confirm_appointment(appointment_id), then confirm the booking with the \
date and time.
    - If they say no: call cancel_appointment(appointment_id) to release the hold, then offer the \
next available candidate. If you run out of candidates, ask for new availability.
- After about 4 rejected offers or 4 rounds of new availability, apologise politely and end the \
call so you never loop forever.
"""


def build_canceller_prompt(
    now: datetime | None = None,
    patient_id: str | None = None,
    clinic_name: str = "Prosper Health",
) -> str:
    """CANCEL phase: the cancellation flow for an already-identified caller."""
    who = f"patient_id {patient_id}" if patient_id else "the identified caller"
    return always_on_rules(now, clinic_name) + f"""
YOUR ROLE RIGHT NOW — cancel an existing appointment for an already-identified caller ({who}). Use \
that patient_id. The caller is already verified; do not re-identify them.

CANCEL FLOW
- Call list_patient_appointments(patient_id). It returns only upcoming, still-scheduled \
appointments.
  - If empty: gently tell them they have no upcoming appointments to cancel, and say goodbye. (You \
cannot book in this step.)
  - If one: "You have one on {{date}} at {{time}}. Would you like to cancel that?" Wait for a yes.
  - If several: list them briefly and ask which one they mean.
- Only after an explicit yes, call cancel_appointment(appointment_id) and confirm it's cancelled.
"""


# --- Phase registry ---------------------------------------------------------


@dataclass(frozen=True)
class Phase:
    """One specialist: its name, the tools it may use, and how to build its prompt."""

    name: str
    tools: tuple[FunctionSchema, ...]
    build_prompt: Callable[..., str]


IDENTIFY = Phase(
    name="identify",
    tools=(
        confirm_patient_data,
        find_patient,
        create_patient,
        transfer_to_booking,
        transfer_to_cancellation,
    ),
    build_prompt=build_identifier_prompt,
)

BOOK = Phase(
    name="book",
    tools=(
        list_availability_slots,
        create_appointment,
        confirm_appointment,
        cancel_appointment,
    ),
    build_prompt=build_booker_prompt,
)

CANCEL = Phase(
    name="cancel",
    tools=(
        list_patient_appointments,
        cancel_appointment,
    ),
    build_prompt=build_canceller_prompt,
)

# Which destination phase each transfer tool routes to.
TRANSFERS: dict[str, Phase] = {
    "transfer_to_booking": BOOK,
    "transfer_to_cancellation": CANCEL,
}


# --- Per-call state ---------------------------------------------------------


@dataclass
class CallState:
    """Facts captured once per call and reused when a phase swaps.

    ``now`` is the call's "today" anchor: the identifier prompt is built with it,
    so the booker / canceller prompt must reuse the SAME value (rather than
    silently falling back to wall-clock) or relative dates like "this Wednesday"
    would resolve against a different day after the handoff.

    ``patient_id`` is folded in deterministically from the real find_patient /
    create_patient result (see ``_make_capturing_coro``), so the transfer no
    longer has to trust the model to copy the id into its tool arguments.
    """

    now: datetime | None = None
    patient_id: str | None = None


def _make_capturing_coro(name: str, coro, state: CallState):
    """Wrap an EHR coroutine so a successful identify result records the patient_id.

    Wrapping the *coroutine* (not the Pipecat handler) keeps all loop-safety in
    the shared ``make_handler`` untouched; we only observe the result and fold the
    id into ``state`` before returning it unchanged.
    """

    async def wrapped(**kwargs):
        result = await coro(**kwargs)
        if isinstance(result, dict) and result.get("id"):
            state.patient_id = result["id"]
        return result

    return wrapped


# --- Phase swapping ---------------------------------------------------------


def _set_system_prompt(context, content: str) -> None:
    """Replace the first system message's content; insert one if there is none.

    Only the first system message is touched. ``bot.py`` may have appended a later
    "kick off the call" system message; leaving it in place is harmless history.
    """
    new_messages = []
    replaced = False
    for message in context.messages:
        if not replaced and isinstance(message, dict) and message.get("role") == "system":
            new_messages.append({**message, "content": content})
            replaced = True
        else:
            new_messages.append(message)
    if not replaced:
        new_messages.insert(0, {"role": "system", "content": content})
    context.set_messages(new_messages)


def _apply_phase(
    context, phase: Phase, patient_id: str | None = None, now: datetime | None = None
) -> None:
    """Make ``phase`` the active specialist: swap its prompt and its tool subset.

    The OpenAI LLM service reads ``context.tools`` and the system prompt fresh on
    every inference, so swapping them inside a tool handler takes effect on the
    very next turn. ``now`` is threaded through so the new phase keeps the call's
    original "today" anchor instead of resetting to wall-clock.
    """
    _set_system_prompt(context, phase.build_prompt(now=now, patient_id=patient_id))
    context.set_tools(ToolsSchema(standard_tools=list(phase.tools)))


def _make_transfer_handler(phase: Phase, state: CallState | None = None):
    """Handler for a ``transfer_to_*`` tool: advance to ``phase`` and report back.

    The patient_id is taken from ``state`` — captured deterministically from the
    real find_patient / create_patient result — and only falls back to the
    model-supplied argument if nothing was captured. ``state.now`` is reused so the
    next phase keeps the same "today" the identifier used.
    """

    async def handler(params: FunctionCallParams) -> None:
        captured = state.patient_id if state else None
        patient_id = captured or params.arguments.get("patient_id")
        now = state.now if state else None
        _apply_phase(params.context, phase, patient_id=patient_id, now=now)
        await params.result_callback({"status": "transferred", "phase": phase.name})

    return handler


# --- Public surface (mirrors agent.py so bot.py can swap modules) -----------


def register_tools(
    llm: OpenAILLMService, guard: CallGuard | None = None, now: datetime | None = None
) -> CallGuard:
    """Register every handler on the LLM service: EHR tools + transfer tools.

    All handlers are registered up front; which ones the model may actually call
    is controlled per phase by the offered tool subset (``get_initial_tools_schema``
    and ``_apply_phase``). A fresh per-call ``CallGuard`` (shared by the EHR
    handlers) provides the same loop-safety as the single agent; transfer handlers
    are unguarded because the conversation can only pass through them a bounded
    number of times.

    A per-call ``CallState`` is shared between the identify handlers and the
    transfer handlers: the identify handlers fold the real patient_id into it, the
    transfer handlers read it back (so the handoff id never depends on the model)
    and reuse ``now`` as the call's "today". Pass the same ``now`` used to build the
    initial system prompt so a phase swap keeps that date anchor.
    """
    guard = guard or CallGuard()
    state = CallState(now=now)
    for name, coro in TOOL_HANDLERS.items():
        # Only the identify tools yield a patient_id worth capturing.
        if name in ("find_patient", "create_patient"):
            coro = _make_capturing_coro(name, coro, state)
        llm.register_function(name, make_handler(name, coro, guard))
    for name, phase in TRANSFERS.items():
        llm.register_function(name, _make_transfer_handler(phase, state))
    return guard


def get_initial_system_prompt(now: datetime | None = None) -> str:
    """The starting (IDENTIFY) system prompt for a fresh call."""
    return IDENTIFY.build_prompt(now=now)


def get_initial_tools_schema() -> ToolsSchema:
    """The starting (IDENTIFY) tool subset to attach to the LLM context."""
    return ToolsSchema(standard_tools=list(IDENTIFY.tools))
