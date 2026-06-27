"""Per-call loop safety, shared by every agent architecture.

This is the pure-logic half of the runtime kernel: the streak/circuit-breaker
state machine plus the small stop-signal helpers. It has no Pipecat, audio, or
network dependency, so it is trivially unit-testable and can be imported by the
single-context agent and the supervisor alike without any of them depending on
each other.

The system prompt asks the model to give up politely after a few empty-availability
rounds or rejected offers, but a prompt is only a request. ``CallGuard`` turns those
soft caps into a hard guarantee: it tracks per-call counters and reports a
``{"stop": true, "reason": ...}`` signal when a streak threshold or the global
tool-call ceiling is crossed. The handler that owns the guard decides what to do
with that signal (attach it to the result, end the call, or both).
"""

import os
from typing import Any

MAX_EMPTY_AVAILABILITY_ROUNDS = int(os.getenv("MAX_EMPTY_AVAILABILITY_ROUNDS"))
MAX_REJECTED_OFFERS = int(os.getenv("MAX_REJECTED_OFFERS"))
MAX_TOTAL_TOOL_CALLS = int(os.getenv("MAX_TOTAL_TOOL_CALLS"))


def _stop(reason: str) -> dict[str, Any]:
    """The hard signal handed back to the model (and obeyed by the prompt)."""
    return {"stop": True, "reason": reason}


def with_stop(result: Any, signal: dict[str, Any]) -> dict[str, Any]:
    """Attach the stop signal to a tool result without dropping the payload."""
    if isinstance(result, dict):
        return {**result, **signal}
    return {"result": result, **signal}


class CallGuard:
    """Per-call counters that turn the prompt's soft caps into a hard guarantee.

    One instance lives for the duration of a single call, so every counter is
    naturally per-session. It watches the stream of tool calls and results and
    reports when a limit is crossed:

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
