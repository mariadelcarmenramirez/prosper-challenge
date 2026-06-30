"""Pipecat-bound runtime kernel, shared by every agent architecture.

This is the transport-facing half of the kernel (the pure-logic half lives in
``guard.py``). It owns the one tool registry that maps each LLM tool name to its
EHR implementation, the factory that wraps an implementation in a Pipecat handler
with loop-safety attached, the graceful call-end frame, and the LLM constructor.

Every architecture (single-context ``agent`` and ``supervisor_agent``) builds on
this module and on ``guard``; none of them depends on another, so a change to one
flow cannot break the others.
"""

import os
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

from ..tools import implementations as tool_implementations
from .guard import CallGuard, with_stop

# The single source of truth mapping each LLM tool name to its EHR coroutine.
# Architectures register some or all of these; the supervisor's workers call them
# directly. Keeping it here (not inside any one architecture) is what lets all
# three share identical tool behaviour without importing each other.
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


async def end_call(params: FunctionCallParams) -> None:
    """Hard stop: ask the pipeline task to shut down gracefully (flushes queued speech)."""
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


# Short spoken acknowledgements played the instant a backend lookup starts, so the
# caller hears that we got their info instead of dead air while the EHR + LLM work.
ACK_PHRASES = (
    "Okay, perfect — give me just a moment.",
    "Got it, let me check that for you.",
    "Thank you, one moment please.",
    "Alright, bear with me one second.",
)
# One user turn can fire a short burst of tool calls (e.g. find then create). Debounce
# so we acknowledge once per turn, not once per tool — a genuinely new turn is always
# separated by the caller speaking again, which is far longer than this window.
ACK_DEBOUNCE_SECS = 4.0


async def speak_ack(params: FunctionCallParams, ack_state: dict[str, float]) -> None:
    """Speak a brief filler the moment a tool runs, debounced across a tool burst."""
    now = time.monotonic()
    if now - ack_state["last"] < ACK_DEBOUNCE_SECS:
        return
    ack_state["last"] = now
    await params.llm.push_frame(
        TTSSpeakFrame(random.choice(ACK_PHRASES)), FrameDirection.DOWNSTREAM
    )


def make_handler(
    name: str,
    coro: Callable[..., Awaitable[Any]],
    guard: CallGuard,
    ack_state: dict[str, float] | None = None,
):
    """Wrap an EHR coroutine in a Pipecat function handler with loop-safety attached.

    When ``ack_state`` is provided, a short spoken acknowledgement is played as soon
    as the tool starts so the caller is not left in silence during the lookup.
    """

    async def handler(params: FunctionCallParams) -> None:
        # Circuit breaker first: if we are over the global ceiling, do not even
        # run the tool — return the stop signal and end the call programmatically.
        signal = guard.record_call()
        if signal is not None:
            await params.result_callback(signal)
            await end_call(params)
            return

        # We have the caller's info and are about to hit the EHR: let them know.
        if ack_state is not None:
            await speak_ack(params, ack_state)

        result = await coro(**params.arguments)
        # find_patient returns None, a JSON-serializable signal for unknown patients.
        if result is None:
            result = {"found": False}

        # Streak thresholds: hand the model the stop signal so the prompt can end
        # the call politely. The global ceiling above is the backstop if it doesn't.
        signal = guard.update(name, result)
        if signal is not None:
            result = with_stop(result, signal)

        await params.result_callback(result)

    return handler


def build_llm(model: str = "gpt-4.1") -> OpenAILLMService:
    """Construct the LLM service identically for the bot and the eval harness."""
    return OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model=model)
