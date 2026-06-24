"""The audio-free "brain": binds LLM tools to their EHR implementations.

Imported by both ``bot.py`` (audio pipeline) and the test/eval harness so they
build the LLM and register the same handlers identically. Nothing here depends
on the transport or audio — only on the LLM service abstraction.

Loop safety lives here too. The system prompt asks the model to give up politely
after a few empty-availability rounds or rejected offers, but a prompt is only a
request. ``CallGuard`` turns those caps into a hard guarantee: it tracks
per-call counters, attaches a ``{"stop": true, "reason": ...}`` signal to the
tool result when a streak threshold is crossed (which the prompt is instructed
to obey), and ends the call programmatically once a global tool-call ceiling is
hit so a runaway loop is aborted no matter what the model does.
"""

import os
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

import tool_implementations
from tool_schemas import TOOL_SCHEMAS

TOOL_HANDLERS: dict[str, Callable[..., Awaitable[Any]]] = {
    "confirm_patient_data": tool_implementations.confirm_patient_data,
    "find_patient": tool_implementations.find_patient,
    "create_patient": tool_implementations.create_patient,
    "list_availability_slots": tool_implementations.list_availability_slots,
    "list_patient_appointments": tool_implementations.list_patient_appointments,
    "create_appointment": tool_implementations.create_appointment,
    "confirm_appointment": tool_implementations.confirm_appointment,
    "cancel_appointment": tool_implementations.cancel_appointment,
}

# Loop-safety thresholds. The prompt mirrors the first two as polite UX ("after
# about 4 rounds, apologise and end"); these are the values the code enforces.
MAX_EMPTY_AVAILABILITY_ROUNDS = 4  # consecutive list_availability_slots with no free slots
MAX_REJECTED_OFFERS = 4            # consecutive held offers the caller turned down
MAX_TOTAL_TOOL_CALLS = 40          # global circuit breaker: any runaway loop, not just slots


def _stop(reason: str) -> dict[str, Any]:
    """The hard signal handed back to the model (and obeyed by the prompt)."""
    return {"stop": True, "reason": reason}


def _with_stop(result: Any, signal: dict[str, Any]) -> dict[str, Any]:
    """Attach the stop signal to a tool result without dropping the payload."""
    if isinstance(result, dict):
        return {**result, **signal}
    return {"result": result, **signal}


class CallGuard:
    """Per-call counters that turn the prompt's soft caps into a hard guarantee.

    One instance lives for the duration of a single call (created in
    ``register_tools``), so every counter is naturally per-session. It watches
    the stream of tool calls and results and reports when a limit is crossed:

    * ``record_call`` counts every tool call and fires the global circuit
      breaker once ``MAX_TOTAL_TOOL_CALLS`` is reached.
    * ``update`` inspects a tool's result to track the two domain streaks —
      consecutive empty availability rounds and consecutive rejected offers —
      and fires once either threshold is reached.

    Both report by returning a ``{"stop": true, "reason": ...}`` dict (or
    ``None`` when nothing is wrong); the handler decides what to do with it.
    """

    def __init__(self) -> None:
        self.total_calls = 0
        self.empty_availability_rounds = 0
        self.rejected_offers = 0
        # Appointment ids we are holding but have not confirmed yet. Cancelling
        # one of these means the caller rejected the offer; cancelling anything
        # else is a genuine booking cancellation and must not count.
        self._held_ids: set[str] = set()

    def record_call(self) -> dict[str, Any] | None:
        """Count one tool call; return a stop signal if the global ceiling is hit."""
        self.total_calls += 1
        if self.total_calls >= MAX_TOTAL_TOOL_CALLS:
            return _stop("tool_call_limit")
        return None

    def update(self, name: str, result: Any) -> dict[str, Any] | None:
        """Inspect one tool result; return a stop signal if a streak threshold is hit."""
        if name == "list_availability_slots":
            return self._on_availability(result)
        if name == "create_appointment":
            self._on_create(result)
        elif name == "confirm_appointment":
            self._on_confirm(result)
        elif name == "cancel_appointment":
            return self._on_cancel(result)
        return None

    def _on_availability(self, result: Any) -> dict[str, Any] | None:
        # /slots returns a list; an error returns a dict — only a successful,
        # empty list counts as "no availability this round".
        if not isinstance(result, list):
            return None
        if result:
            self.empty_availability_rounds = 0
            return None
        self.empty_availability_rounds += 1
        if self.empty_availability_rounds >= MAX_EMPTY_AVAILABILITY_ROUNDS:
            return _stop("no_availability")
        return None

    def _on_create(self, result: Any) -> None:
        if isinstance(result, dict) and result.get("status") == "held" and result.get("id"):
            self._held_ids.add(result["id"])

    def _on_confirm(self, result: Any) -> None:
        # The caller accepted an offer: the rejection streak resets.
        if isinstance(result, dict) and result.get("status") == "scheduled":
            self._held_ids.discard(result.get("id", ""))
            self.rejected_offers = 0

    def _on_cancel(self, result: Any) -> dict[str, Any] | None:
        appt_id = result.get("id") if isinstance(result, dict) else None
        if appt_id not in self._held_ids:
            return None  # cancelling a real booking, not a rejected hold
        self._held_ids.discard(appt_id)
        self.rejected_offers += 1
        if self.rejected_offers >= MAX_REJECTED_OFFERS:
            return _stop("too_many_rejections")
        return None


async def _end_call(params: FunctionCallParams) -> None:
    """Hard stop: ask the pipeline task to shut down gracefully (flushes queued speech)."""
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


def _make_handler(name: str, coro: Callable[..., Awaitable[Any]], guard: CallGuard):
    async def handler(params: FunctionCallParams) -> None:
        # Circuit breaker first: if we are over the global ceiling, do not even
        # run the tool — return the stop signal and end the call programmatically.
        signal = guard.record_call()
        if signal is not None:
            await params.result_callback(signal)
            await _end_call(params)
            return

        result = await coro(**params.arguments)
        # find_patient returns None, a JSON-serializable signal for unknown patients.
        if result is None:
            result = {"found": False}

        # Streak thresholds: hand the model the stop signal so the prompt can end
        # the call politely. The global ceiling above is the backstop if it doesn't.
        signal = guard.update(name, result)
        if signal is not None:
            result = _with_stop(result, signal)

        await params.result_callback(result)

    return handler


def register_tools(llm: OpenAILLMService, guard: CallGuard | None = None) -> CallGuard:
    """Register every tool handler on the LLM service.

    A fresh ``CallGuard`` is created per call (one per ``register_tools``) so its
    counters are scoped to a single conversation. The guard is returned so the
    caller (or a test/eval harness) can inspect or pre-seed the per-call state.
    """
    guard = guard or CallGuard()
    for name, coro in TOOL_HANDLERS.items():
        llm.register_function(name, _make_handler(name, coro, guard))
    return guard


def get_tools_schema() -> ToolsSchema:
    """Tool schemas to attach to the LLM context."""
    return ToolsSchema(standard_tools=TOOL_SCHEMAS)


def build_llm(model: str = "gpt-4.1") -> OpenAILLMService:
    """Construct the LLM service identically for the bot and the eval harness."""
    return OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model=model)
