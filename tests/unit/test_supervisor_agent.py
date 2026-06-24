"""Unit tests for the supervisor-workers agent.

These cover the parts unique to ``supervisor_agent.py``: the supervisor's
delegation tool surface, each worker's EHR tool subset, the FunctionSchema ->
OpenAI tool conversion, the shared-state observation, and the worker tool-calling
loop driven by a fake OpenAI client (no network). Loop-safety itself is reused
from ``agent.py`` and covered by ``test_call_guard.py``.
"""

import json
from types import SimpleNamespace

from pipecat.frames.frames import EndTaskFrame

import supervisor_agent as sup
from agent import MAX_TOTAL_TOOL_CALLS, CallGuard

IDENTIFIER_TOOLS = {"confirm_patient_data", "find_patient", "create_patient"}
BOOKER_TOOLS = {
    "list_availability_slots",
    "create_appointment",
    "confirm_appointment",
    "cancel_appointment",
}
CANCELLER_TOOLS = {"list_patient_appointments", "cancel_appointment"}


def _schema_names(schemas):
    return {s.name for s in schemas}


# --- Surfaces ---------------------------------------------------------------


def test_supervisor_only_has_the_three_delegation_tools():
    names = {t.name for t in sup.SUPERVISOR_TOOLS}
    assert names == {"identify_caller", "book_appointment", "cancel_appointment"}
    # The supervisor must NOT be handed any raw EHR tools. ("cancel_appointment" is
    # excluded: the supervisor's cancel *delegation* tool deliberately shares that
    # name, but it is a different schema from the EHR tool the workers call.)
    ehr_only = (IDENTIFIER_TOOLS | BOOKER_TOOLS | CANCELLER_TOOLS) - {"cancel_appointment"}
    assert not (names & ehr_only)


def test_each_worker_owns_its_ehr_subset():
    assert _schema_names(sup.IDENTIFIER_WORKER.schemas) == IDENTIFIER_TOOLS
    assert _schema_names(sup.BOOKER_WORKER.schemas) == BOOKER_TOOLS
    assert _schema_names(sup.CANCELLER_WORKER.schemas) == CANCELLER_TOOLS


def test_to_openai_tool_shape():
    tool = sup._to_openai_tool(sup.identify_caller)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "identify_caller"
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"full_name", "date_of_birth", "phone"}
    assert "phone" in params["properties"]


# --- Shared-state observation ----------------------------------------------


def test_state_captures_patient_from_identification():
    state = sup.SessionState()
    sup._update_state(state, "find_patient", {"id": "pat-1", "full_name": "Jane Doe"})
    assert state.patient_id == "pat-1"
    assert state.patient_name == "Jane Doe"


def test_state_tracks_and_clears_held_appointment():
    state = sup.SessionState()
    sup._update_state(state, "create_appointment", {"id": "appt-1", "status": "held",
                                                    "starts_at": "2026-07-06T15:00:00"})
    assert state.held_id == "appt-1"
    assert state.held_slot == "2026-07-06T15:00:00"

    sup._update_state(state, "confirm_appointment", {"id": "appt-1", "status": "scheduled"})
    assert state.held_id is None and state.held_slot is None


def test_cancelling_the_held_slot_clears_it_but_a_real_booking_does_not():
    state = sup.SessionState(held_id="appt-1", held_slot="2026-07-06T15:00:00")
    sup._update_state(state, "cancel_appointment", {"id": "other", "status": "cancelled"})
    assert state.held_id == "appt-1"  # unrelated cancellation leaves the hold alone
    sup._update_state(state, "cancel_appointment", {"id": "appt-1", "status": "cancelled"})
    assert state.held_id is None


# --- Worker tool-calling loop (fake OpenAI client) --------------------------


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


class _FakeOpenAI:
    """Returns a scripted sequence of chat-completion responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

        async def create(**kwargs):
            self.calls.append(kwargs)
            message = self._responses.pop(0)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def test_worker_loop_runs_a_tool_then_reports():
    """One tool call, then a final text report; the tool result feeds the model."""
    seen = {}

    async def fake_find(**kwargs):
        seen["args"] = kwargs
        return {"id": "pat-7", "full_name": "Jane Doe"}

    client = _FakeOpenAI(
        [
            _msg(tool_calls=[_tool_call("c1", "find_patient", {"full_name": "Jane Doe"})]),
            _msg(content="The caller is identified: Jane Doe."),
        ]
    )
    guard = CallGuard()
    state = sup.SessionState()

    report = await sup._run_agent_loop(
        system_prompt="ignored",
        tools=[],
        task="identify",
        client=client,
        guard=guard,
        state=state,
        handlers={"find_patient": fake_find},
    )

    assert report == "The caller is identified: Jane Doe."
    assert seen["args"] == {"full_name": "Jane Doe"}
    # State was observed from the worker's tool result.
    assert state.patient_id == "pat-7"
    # Second model call saw the tool result appended as a tool message.
    second_call_messages = client.calls[1]["messages"]
    assert second_call_messages[-1]["role"] == "tool"
    assert json.loads(second_call_messages[-1]["content"])["id"] == "pat-7"


async def test_worker_loop_stops_at_max_steps_without_a_final_message():
    """A model that only ever calls tools is bounded by WORKER_MAX_STEPS."""

    async def fake_slots(**kwargs):
        return []

    responses = [
        _msg(tool_calls=[_tool_call(f"c{i}", "list_availability_slots", {})])
        for i in range(sup.WORKER_MAX_STEPS)
    ]
    client = _FakeOpenAI(responses)

    report = await sup._run_agent_loop(
        system_prompt="ignored",
        tools=[],
        task="book",
        client=client,
        guard=CallGuard(),
        state=sup.SessionState(),
        handlers={"list_availability_slots": fake_slots},
    )

    assert "couldn't finish" in report
    assert len(client.calls) == sup.WORKER_MAX_STEPS


# --- Delegation handler wiring ----------------------------------------------


def _params(arguments, calls):
    async def result_callback(result):
        calls["result"] = result

    async def push_frame(frame, direction):
        calls.setdefault("frames", []).append(frame)

    return SimpleNamespace(
        arguments=arguments,
        result_callback=result_callback,
        llm=SimpleNamespace(push_frame=push_frame),
    )


async def test_delegation_handler_returns_worker_report():
    supervisor = sup.Supervisor()
    supervisor._client_obj = _FakeOpenAI([_msg(content="All done.")])

    handler = supervisor._make_delegation_handler(sup.BOOKER_WORKER)
    calls: dict = {}
    await handler(_params({"request": "Tuesday morning"}, calls))

    assert calls["result"] == {"report": "All done."}
    assert "frames" not in calls


async def test_delegation_handler_ends_call_at_global_ceiling():
    supervisor = sup.Supervisor()
    supervisor.guard.total_calls = MAX_TOTAL_TOOL_CALLS  # already at the ceiling
    supervisor._client_obj = _FakeOpenAI([_msg(content="partial")])

    handler = supervisor._make_delegation_handler(sup.BOOKER_WORKER)
    calls: dict = {}
    await handler(_params({"request": "x"}, calls))

    assert calls["result"]["stop"] is True
    assert calls["result"]["reason"] == "tool_call_limit"
    assert len(calls["frames"]) == 1 and isinstance(calls["frames"][0], EndTaskFrame)
