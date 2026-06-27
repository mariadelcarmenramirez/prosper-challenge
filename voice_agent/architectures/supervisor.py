import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

# Build on the shared kernel (guard + runtime) and the shared prompt header, never
# on a sibling architecture, so the supervisor reuses identical loop-safety and the
# same clinic preamble without importing a sibling architecture.
from ..core.guard import MAX_TOTAL_TOOL_CALLS, CallGuard, with_stop
from ..core.prompts import always_on_rules
from ..core.runtime import TOOL_HANDLERS, build_llm, end_call

# Aliased to avoid colliding with the supervisor's own ``cancel_appointment``
# delegation tool defined below: this is the EHR tool the workers actually call.
from ..tools.schemas import cancel_appointment as _ehr_cancel_appointment
from ..tools.schemas import (
    confirm_appointment,
    confirm_patient_data,
    create_appointment,
    create_patient,
    find_patient,
    list_availability_slots,
    list_patient_appointments,
)

WORKER_MODEL = "gpt-4.1"
WORKER_MAX_STEPS = 6  # hard cap on a single worker's internal tool-calling loop


# --- Supervisor delegation tools (the supervisor's ONLY tools) --------------

identify_caller = FunctionSchema(
    name="identify_caller",
    description=(
        "Delegate to the identification worker to validate and find-or-register "
        "the caller. Call this first, once you have their full name, date of birth "
        "and phone. The worker reports whether they were identified (or which field "
        "to re-ask). You do not get or need a patient id — the workers track it."
    ),
    properties={
        "full_name": {"type": "string", "description": "Caller's full name as heard."},
        "date_of_birth": {"type": "string", "description": "Date of birth as YYYY-MM-DD."},
        "phone": {"type": "string", "description": "Caller's phone number as heard."},
    },
    required=["full_name", "date_of_birth", "phone"],
)

book_appointment = FunctionSchema(
    name="book_appointment",
    description=(
        "Delegate to the booking worker. Pass a plain-words note: either the "
        "caller's availability (with exact dates), or their confirmation/rejection "
        "of a slot the worker offered. The worker checks availability, holds or "
        "confirms a slot, and reports the offer or outcome. Only call after "
        "identify_caller has identified the caller."
    ),
    properties={
        "request": {
            "type": "string",
            "description": (
                "What to tell the booking worker, in plain words — e.g. 'The caller "
                "wants Monday the 6th of July at 3pm, or any time Tuesday morning' or "
                "'The caller confirmed the offered slot' or 'The caller declined; "
                "offer the next option'."
            ),
        },
    },
    required=["request"],
)

cancel_appointment = FunctionSchema(
    name="cancel_appointment",
    description=(
        "Delegate to the cancellation worker. Pass a plain-words note: either 'list "
        "the caller's upcoming appointments', or which appointment the caller has "
        "confirmed they want to cancel. The worker lists or cancels and reports back. "
        "Only call after identify_caller has identified the caller."
    ),
    properties={
        "request": {
            "type": "string",
            "description": (
                "What to tell the cancellation worker, in plain words — e.g. 'List "
                "the caller's upcoming appointments' or 'The caller confirmed "
                "cancelling the Monday the 6th of July at 3pm appointment'."
            ),
        },
    },
    required=["request"],
)

SUPERVISOR_TOOLS: list[FunctionSchema] = [identify_caller, book_appointment, cancel_appointment]


# --- Shared per-call state --------------------------------------------------


@dataclass
class SessionState:
    """Per-call facts the supervisor never has to handle itself.

    Updated by observing worker EHR results, then auto-injected into the booking
    and cancellation worker tasks.
    """

    patient_id: str | None = None
    patient_name: str | None = None
    held_id: str | None = None
    held_slot: str | None = None


def _update_state(state: SessionState, name: str, result: Any) -> None:
    """Fold one EHR tool result into the shared state."""
    if not isinstance(result, dict):
        return
    if name in ("find_patient", "create_patient") and result.get("id"):
        state.patient_id = result["id"]
        state.patient_name = result.get("full_name") or state.patient_name
    elif name == "create_appointment" and result.get("status") == "held" and result.get("id"):
        state.held_id = result["id"]
        state.held_slot = result.get("starts_at")
    elif name == "confirm_appointment" and result.get("status") == "scheduled":
        state.held_id = None
        state.held_slot = None
    elif name == "cancel_appointment" and result.get("id") == state.held_id:
        state.held_id = None
        state.held_slot = None


# --- Workers ----------------------------------------------------------------


def _build_worker_preamble(now: datetime | None, role: str) -> str:
    """Shared worker header: the clinic preamble plus 'you don't talk to the caller'."""
    return always_on_rules(now, "Prosper Health") + f"""
YOU ARE THE {role} WORKER. You do NOT talk to the caller — a supervisor relays your results, so \
write a short report (1-2 sentences) for the supervisor, not a spoken line. Use your tools to do \
the work. If any tool result contains "stop": true, do not call more tools — just report the \
situation and stop.
"""


def build_identifier_worker_prompt(now: datetime | None = None) -> str:
    return _build_worker_preamble(now, "IDENTIFICATION") + """
TASK: validate the caller's details, then find or register them.
- Call confirm_patient_data(full_name, date_of_birth, phone). If it returns "valid: false", do NOT
  look anyone up — report which field in "issues" looks wrong so the supervisor can re-ask, and stop.
- If valid, call find_patient with the NORMALIZED values. If found, report that the caller is
  identified and their name. If "found: false", call create_patient with those values and report the
  caller has been registered. End your report with the caller's name.
"""


def build_booker_worker_prompt(now: datetime | None = None) -> str:
    return _build_worker_preamble(now, "BOOKING") + """
TASK: act on the supervisor's note about the caller's availability or their confirmation/rejection.
You are given the caller's patient_id and any slot currently held for them.
- If the note gives availability: call list_availability_slots for the relevant day/range, then take
  the EARLIEST free slot that matches and call create_appointment(patient_id, starts_at) to hold it.
  Report the offered date and time (in words) and that it is held, awaiting the caller's yes.
- If the note says the caller CONFIRMED: call confirm_appointment on the held appointment, then
  report it is booked, with the date and time.
- If the note says the caller DECLINED: call cancel_appointment to release the held slot; then, if
  new times were given, hold the next match and report the new offer; otherwise report that you need
  other times.
- If nothing is available: report that and that the supervisor should ask for other times.
Pass appointment times as YYYY-MM-DDTHH:MM:SS. Never read ids aloud; the supervisor tracks the held
slot for you, so you do not need to mention any id in your report.
"""


def build_canceller_worker_prompt(now: datetime | None = None) -> str:
    return _build_worker_preamble(now, "CANCELLATION") + """
TASK: act on the supervisor's note. You are given the caller's patient_id.
- Always start by calling list_patient_appointments(patient_id); it returns only upcoming,
  still-scheduled appointments.
  - If empty: report the caller has no upcoming appointments to cancel.
  - If the note only asks to list them: report each upcoming appointment's date and time (in words)
    so the supervisor can ask which one.
  - If the note says the caller CONFIRMED cancelling a specific one: call cancel_appointment on the
    matching appointment and report it is cancelled, with the date and time.
"""


@dataclass(frozen=True)
class Worker:
    """An LLM sub-agent: its EHR tool subset, its prompt, and how to brief it."""

    name: str
    schemas: tuple[FunctionSchema, ...]
    build_prompt: Callable[..., str]
    build_task: Callable[[dict, SessionState], str]

    def openai_tools(self) -> list[dict]:
        return [_to_openai_tool(schema) for schema in self.schemas]


def _identifier_task(args: dict, state: SessionState) -> str:
    return (
        f"The caller provided name='{args.get('full_name')}', "
        f"date_of_birth='{args.get('date_of_birth')}', phone='{args.get('phone')}'. "
        "Validate, then find or register them."
    )


def _booker_task(args: dict, state: SessionState) -> str:
    held = state.held_id or "none"
    held_at = f" at {state.held_slot}" if state.held_slot else ""
    return (
        f"Caller patient_id={state.patient_id}. Currently held appointment: {held}{held_at}. "
        f"Supervisor relay: {args.get('request', '')}"
    )


def _canceller_task(args: dict, state: SessionState) -> str:
    return f"Caller patient_id={state.patient_id}. Supervisor relay: {args.get('request', '')}"


IDENTIFIER_WORKER = Worker(
    name="identifier",
    schemas=(confirm_patient_data, find_patient, create_patient),
    build_prompt=build_identifier_worker_prompt,
    build_task=_identifier_task,
)

BOOKER_WORKER = Worker(
    name="booker",
    schemas=(
        list_availability_slots,
        create_appointment,
        confirm_appointment,
        _ehr_cancel_appointment,
    ),
    build_prompt=build_booker_worker_prompt,
    build_task=_booker_task,
)

CANCELLER_WORKER = Worker(
    name="canceller",
    schemas=(list_patient_appointments, _ehr_cancel_appointment),
    build_prompt=build_canceller_worker_prompt,
    build_task=_canceller_task,
)


# --- Worker execution loop --------------------------------------------------


def _to_openai_tool(schema: FunctionSchema) -> dict:
    """Convert a pipecat FunctionSchema to an OpenAI chat-completions tool dict."""
    return {
        "type": "function",
        "function": {
            "name": schema.name,
            "description": schema.description,
            "parameters": {
                "type": "object",
                "properties": schema.properties,
                "required": schema.required,
            },
        },
    }


def _parse_args(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _serialize_tool_call(tool_call) -> dict:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


async def _execute_tool(
    name: str,
    arguments: dict,
    guard: CallGuard,
    state: SessionState,
    handlers: dict[str, Callable[..., Awaitable[Any]]],
) -> Any:
    """Run one EHR tool for a worker, with the same loop-safety as the single agent."""
    signal = guard.record_call()
    if signal is not None:
        return signal  # global ceiling: hand the worker the stop signal
    result = await handlers[name](**arguments)
    if result is None:  # find_patient signals "unknown" with None
        result = {"found": False}
    _update_state(state, name, result)
    streak = guard.update(name, result)
    if streak is not None:
        result = with_stop(result, streak)
    return result


async def _run_agent_loop(
    *,
    system_prompt: str,
    tools: list[dict],
    task: str,
    client,
    guard: CallGuard,
    state: SessionState,
    handlers: dict[str, Callable[..., Awaitable[Any]]] | None = None,
    model: str = WORKER_MODEL,
    max_steps: int = WORKER_MAX_STEPS,
) -> str:
    """Run a worker's bounded tool-calling loop and return its report for the supervisor."""
    handlers = handlers if handlers is not None else TOOL_HANDLERS
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    for _ in range(max_steps):
        response = await client.chat.completions.create(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or ""
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [_serialize_tool_call(tc) for tc in tool_calls],
            }
        )
        for tool_call in tool_calls:
            result = await _execute_tool(
                tool_call.function.name,
                _parse_args(tool_call.function.arguments),
                guard,
                state,
                handlers,
            )
            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
            )
    return "I couldn't finish that step; please ask the caller to try again."


# --- Supervisor prompt ------------------------------------------------------


def build_supervisor_prompt(now: datetime | None = None, clinic_name: str = "Prosper Health") -> str:
    return always_on_rules(now, clinic_name) + """
YOU ARE THE SUPERVISOR. You are the only one who speaks with the caller, and you own the whole call. \
You have NO EHR tools; instead you orchestrate three specialist workers and relay what they report, \
in your own warm words:
- identify_caller(full_name, date_of_birth, phone): validates and finds-or-registers the caller.
- book_appointment(request): hands the booking specialist a plain-words note about the caller's \
availability, or their confirmation/rejection of an offer.
- cancel_appointment(request): hands the cancellation specialist a plain-words note about which \
appointment to cancel, or 'list the caller's upcoming appointments'.
You never track a patient id yourself — the workers already know who the caller is once identified.

FLOW
1. Greet, introduce yourself as the Prosper Health scheduling assistant, and collect the caller's \
full name, date of birth and phone number. Then call identify_caller with them.
   - If the report says a field looks wrong, apologise lightly, re-ask only that field, and call \
identify_caller again.
   - Once it reports the caller is identified, greet them by name.
2. Ask whether they want to BOOK a new appointment or CANCEL an existing one.

BOOK
- Ask when they'd like to come in. For EVERY time they mention, repeat it back as an exact date \
("so Monday the 6th of July at 3 in the afternoon?") to remove ambiguity.
- Call book_appointment with their availability in plain words, using exact dates.
- The specialist will either OFFER a held slot or say there is no availability.
  - On an offer: say "I can offer you {date} at {time}. Shall I book it?" and wait.
    - If yes: call book_appointment(request="The caller confirmed the offered slot."), then confirm \
the booking out loud with the date and time.
    - If no: call book_appointment(request="The caller declined; offer the next option.") — or pass \
their new times if they gave any.
  - On no availability: ask for other times and call book_appointment again.

CANCEL
- Call cancel_appointment(request="List the caller's upcoming appointments.").
- Relay what it reports: if none, say so; if one, "You have one on {date} at {time}. Cancel that?"; \
if several, list them and ask which one they mean.
- After an explicit yes, call cancel_appointment(request="The caller confirmed cancelling the \
{date} at {time} appointment.") and confirm it's cancelled.
"""


# --- Supervisor (per-call orchestrator) -------------------------------------


class Supervisor:
    """Per-call object that wires the supervisor LLM to its three worker sub-agents.

    One instance per call (created in ``bot.py``). Mirrors the public surface of
    ``single`` so ``bot.py`` can treat both the same.
    """

    def __init__(self, now: datetime | None = None) -> None:
        self.state = SessionState()
        self.guard = CallGuard()
        self.handlers = TOOL_HANDLERS
        self._now = now
        self._client_obj = None  # lazy: only built when a worker actually runs
        self._workers: dict[str, Worker] = {
            "identify_caller": IDENTIFIER_WORKER,
            "book_appointment": BOOKER_WORKER,
            "cancel_appointment": CANCELLER_WORKER,
        }

    # -- public surface ------------------------------------------------------

    def get_initial_system_prompt(self) -> str:
        return build_supervisor_prompt(self._now)

    def get_initial_tools_schema(self) -> ToolsSchema:
        return ToolsSchema(standard_tools=list(SUPERVISOR_TOOLS))

    def register_tools(self, llm: OpenAILLMService) -> CallGuard:
        """Register the three delegation tools on the supervisor LLM."""
        for tool_name, worker in self._workers.items():
            llm.register_function(tool_name, self._make_delegation_handler(worker))
        return self.guard

    # -- internals -----------------------------------------------------------

    def _client(self):
        if self._client_obj is None:
            from openai import AsyncOpenAI

            self._client_obj = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._client_obj

    def _make_delegation_handler(self, worker: Worker):
        async def handler(params: FunctionCallParams) -> None:
            report = await _run_agent_loop(
                system_prompt=worker.build_prompt(self._now),
                tools=worker.openai_tools(),
                task=worker.build_task(params.arguments, self.state),
                client=self._client(),
                guard=self.guard,
                state=self.state,
                handlers=self.handlers,
            )
            # Global circuit breaker: a worker pushed us over the ceiling — hand the
            # supervisor the report plus a stop signal and end the call, just like
            # the single agent does.
            if self.guard.total_calls >= MAX_TOTAL_TOOL_CALLS:
                await params.result_callback(
                    {"report": report, "stop": True, "reason": "tool_call_limit"}
                )
                await end_call(params)
                return
            await params.result_callback({"report": report})

        return handler
