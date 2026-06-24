"""Unit tests for the per-call loop-safety guard in agent.py.

Two layers are exercised: the ``CallGuard`` state machine directly, and the
handler wiring (``_make_handler``) that attaches the stop signal to a tool
result and ends the call programmatically when the global ceiling is hit.
"""

from types import SimpleNamespace

import agent
from agent import (
    MAX_EMPTY_AVAILABILITY_ROUNDS,
    MAX_REJECTED_OFFERS,
    MAX_TOTAL_TOOL_CALLS,
    CallGuard,
)

HELD = {"id": "appt-1", "status": "held"}
SCHEDULED = {"id": "appt-1", "status": "scheduled"}
CANCELLED = {"id": "appt-1", "status": "cancelled"}
SLOT = {"starts_at": "2026-07-06T10:00:00", "ends_at": "2026-07-06T11:00:00"}


# --- Global circuit breaker -------------------------------------------------


def test_record_call_aborts_at_global_ceiling():
    guard = CallGuard()
    for _ in range(MAX_TOTAL_TOOL_CALLS - 1):
        assert guard.record_call() is None
    signal = guard.record_call()
    assert signal == {"stop": True, "reason": "tool_call_limit"}


# --- Empty-availability streak ---------------------------------------------


def test_empty_availability_streak_stops_at_threshold():
    guard = CallGuard()
    for _ in range(MAX_EMPTY_AVAILABILITY_ROUNDS - 1):
        assert guard.update("list_availability_slots", []) is None
    signal = guard.update("list_availability_slots", [])
    assert signal == {"stop": True, "reason": "no_availability"}


def test_non_empty_availability_resets_the_streak():
    guard = CallGuard()
    for _ in range(MAX_EMPTY_AVAILABILITY_ROUNDS - 1):
        guard.update("list_availability_slots", [])
    # A round with real slots clears the streak, so we can go empty again safely.
    assert guard.update("list_availability_slots", [SLOT]) is None
    assert guard.empty_availability_rounds == 0
    assert guard.update("list_availability_slots", []) is None


def test_availability_error_does_not_count_as_empty():
    guard = CallGuard()
    for _ in range(MAX_EMPTY_AVAILABILITY_ROUNDS + 2):
        assert guard.update("list_availability_slots", {"error": "down"}) is None
    assert guard.empty_availability_rounds == 0


# --- Rejected-offer streak --------------------------------------------------


def _reject_once(guard: CallGuard):
    """Model holds a slot then releases it: one rejected offer."""
    guard.update("create_appointment", HELD)
    return guard.update("cancel_appointment", CANCELLED)


def test_rejected_offer_streak_stops_at_threshold():
    guard = CallGuard()
    for _ in range(MAX_REJECTED_OFFERS - 1):
        assert _reject_once(guard) is None
    assert _reject_once(guard) == {"stop": True, "reason": "too_many_rejections"}


def test_confirm_resets_rejection_streak():
    guard = CallGuard()
    for _ in range(MAX_REJECTED_OFFERS - 1):
        _reject_once(guard)
    guard.update("create_appointment", HELD)
    guard.update("confirm_appointment", SCHEDULED)
    assert guard.rejected_offers == 0


def test_cancelling_a_real_booking_is_not_a_rejection():
    guard = CallGuard()
    # An appointment that was never held by us (e.g. from the cancel flow).
    signal = guard.update("cancel_appointment", {"id": "real-booking", "status": "cancelled"})
    assert signal is None
    assert guard.rejected_offers == 0


# --- Handler wiring ---------------------------------------------------------


def _fake_params(name: str, arguments: dict, calls: dict):
    async def result_callback(result):
        calls["result"] = result

    async def push_frame(frame, direction):
        calls.setdefault("frames", []).append((frame, direction))

    return SimpleNamespace(
        function_name=name,
        arguments=arguments,
        result_callback=result_callback,
        llm=SimpleNamespace(push_frame=push_frame),
    )


async def test_handler_attaches_stop_signal_to_streak_result(monkeypatch):
    """An empty-availability streak handler call returns the slots plus stop, no abort."""

    async def fake_slots(**kwargs):
        return []

    guard = CallGuard()
    guard.empty_availability_rounds = MAX_EMPTY_AVAILABILITY_ROUNDS - 1
    handler = agent._make_handler("list_availability_slots", fake_slots, guard)

    calls: dict = {}
    await handler(_fake_params("list_availability_slots", {"date": "2026-07-06"}, calls))

    assert calls["result"] == {"result": [], "stop": True, "reason": "no_availability"}
    assert "frames" not in calls  # streaks rely on the prompt, not a programmatic end


async def test_handler_ends_call_at_global_ceiling():
    """The global ceiling returns the stop signal AND pushes an EndTaskFrame."""
    from pipecat.frames.frames import EndTaskFrame

    async def fake_tool(**kwargs):
        raise AssertionError("tool must not run once the ceiling is hit")

    guard = CallGuard()
    guard.total_calls = MAX_TOTAL_TOOL_CALLS - 1
    handler = agent._make_handler("find_patient", fake_tool, guard)

    calls: dict = {}
    await handler(_fake_params("find_patient", {}, calls))

    assert calls["result"] == {"stop": True, "reason": "tool_call_limit"}
    assert len(calls["frames"]) == 1
    frame, _direction = calls["frames"][0]
    assert isinstance(frame, EndTaskFrame)


async def test_handler_normalizes_missing_patient_to_found_false():
    async def fake_find(**kwargs):
        return None

    guard = CallGuard()
    handler = agent._make_handler("find_patient", fake_find, guard)

    calls: dict = {}
    await handler(_fake_params("find_patient", {}, calls))
    assert calls["result"] == {"found": False}
