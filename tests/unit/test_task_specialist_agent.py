"""Unit tests for the phased-specialist (sequential handoff) agent.

These cover the parts unique to ``architectures/specialist.py``: the per-phase tool
subsets, the transfer-tool routing, registration of every handler, and the core
mechanic — a ``transfer_to_*`` call swapping the live context's system prompt and
offered tool subset. Loop-safety itself is reused from ``guard.py`` and already
covered by ``test_call_guard.py``.
"""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from pipecat.processors.aggregators.llm_context import LLMContext

import voice_agent.architectures.specialist as tsa

IDENTIFY_TOOLS = {
    "confirm_patient_data",
    "find_patient",
    "create_patient",
    "transfer_to_booking",
    "transfer_to_cancellation",
}
BOOK_TOOLS = {
    "list_availability_slots",
    "create_appointment",
    "confirm_appointment",
    "cancel_appointment",
}
CANCEL_TOOLS = {"list_patient_appointments", "cancel_appointment"}


def _names(schema):
    return {tool.name for tool in schema.standard_tools}


def _phase_names(phase):
    return {tool.name for tool in phase.tools}


# --- Phase definitions ------------------------------------------------------


def test_each_phase_offers_exactly_its_tool_subset():
    assert _phase_names(tsa.IDENTIFY) == IDENTIFY_TOOLS
    assert _phase_names(tsa.BOOK) == BOOK_TOOLS
    assert _phase_names(tsa.CANCEL) == CANCEL_TOOLS


def test_only_the_identifier_can_transfer():
    """Transfer tools must not be offered once a specialist takes over."""
    assert {"transfer_to_booking", "transfer_to_cancellation"} <= _phase_names(tsa.IDENTIFY)
    assert not ({"transfer_to_booking", "transfer_to_cancellation"} & _phase_names(tsa.BOOK))
    assert not ({"transfer_to_booking", "transfer_to_cancellation"} & _phase_names(tsa.CANCEL))


def test_transfer_tools_route_to_the_right_phase():
    assert tsa.TRANSFERS["transfer_to_booking"] is tsa.BOOK
    assert tsa.TRANSFERS["transfer_to_cancellation"] is tsa.CANCEL


def test_initial_phase_is_the_identifier():
    assert _names(tsa.get_initial_tools_schema()) == IDENTIFY_TOOLS
    prompt = tsa.get_initial_system_prompt().lower()
    assert "confirm_patient_data" in prompt
    assert "identify the caller" in prompt


# --- Registration -----------------------------------------------------------


class _FakeLLM:
    def __init__(self):
        self.registered: dict = {}

    def register_function(self, name, handler):
        self.registered[name] = handler


def test_register_tools_registers_every_ehr_and_transfer_handler():
    llm = _FakeLLM()
    guard = tsa.register_tools(llm)

    assert set(llm.registered) == set(tsa.TOOL_HANDLERS) | set(tsa.TRANSFERS)
    assert isinstance(guard, tsa.CallGuard)


# --- The handoff mechanic ---------------------------------------------------


def _params(arguments, context, calls):
    async def result_callback(result):
        calls["result"] = result

    return SimpleNamespace(arguments=arguments, context=context, result_callback=result_callback)


def _fresh_identify_context():
    return LLMContext(
        [{"role": "system", "content": tsa.get_initial_system_prompt()}],
        tools=tsa.get_initial_tools_schema(),
    )


async def test_transfer_to_booking_swaps_tools_and_prompt():
    context = _fresh_identify_context()
    handler = tsa._make_transfer_handler(tsa.BOOK)

    calls: dict = {}
    await handler(_params({"patient_id": "pat-123"}, context, calls))

    # Offered tools are now exactly the booker subset.
    assert _names(context.tools) == BOOK_TOOLS
    # System prompt is the booker's and carries the handed-off patient_id.
    system_msg = context.messages[0]
    assert system_msg["role"] == "system"
    assert "pat-123" in system_msg["content"]
    assert "BOOK FLOW" in system_msg["content"]
    assert calls["result"] == {"status": "transferred", "phase": "book"}


async def test_transfer_to_cancellation_swaps_tools_and_prompt():
    context = _fresh_identify_context()
    handler = tsa._make_transfer_handler(tsa.CANCEL)

    calls: dict = {}
    await handler(_params({"patient_id": "pat-9"}, context, calls))

    assert _names(context.tools) == CANCEL_TOOLS
    system_msg = context.messages[0]
    assert "pat-9" in system_msg["content"]
    assert "CANCEL FLOW" in system_msg["content"]
    assert calls["result"] == {"status": "transferred", "phase": "cancel"}


def test_set_system_prompt_replaces_only_the_first_system_message():
    """A later kick-off system message (as bot.py appends) must survive a swap."""
    context = LLMContext(
        [
            {"role": "system", "content": "ORIGINAL IDENTIFY PROMPT"},
            {"role": "system", "content": "Greet the caller and ask for their details."},
        ],
        tools=tsa.get_initial_tools_schema(),
    )

    tsa._set_system_prompt(context, "NEW BOOKER PROMPT")

    assert context.messages[0]["content"] == "NEW BOOKER PROMPT"
    assert context.messages[1]["content"] == "Greet the caller and ask for their details."


# --- Deterministic patient_id capture ---------------------------------------


async def test_capturing_coro_records_patient_id_from_a_successful_result():
    state = tsa.CallState()

    async def fake_find(**kwargs):
        return {"id": "real-42", "full_name": "Jane"}

    wrapped = tsa._make_capturing_coro("find_patient", fake_find, state)
    result = await wrapped(full_name="Jane")

    assert result == {"id": "real-42", "full_name": "Jane"}  # passes the result through unchanged
    assert state.patient_id == "real-42"


async def test_capturing_coro_ignores_a_result_without_an_id():
    state = tsa.CallState()

    async def fake_find(**kwargs):
        return {"found": False}

    await tsa._make_capturing_coro("find_patient", fake_find, state)()

    assert state.patient_id is None


async def test_transfer_prefers_the_captured_id_over_the_model_argument():
    """A hallucinated patient_id in the tool call must never override the real one."""
    state = tsa.CallState(patient_id="real-42")
    context = _fresh_identify_context()
    handler = tsa._make_transfer_handler(tsa.BOOK, state)

    await handler(_params({"patient_id": "hallucinated-99"}, context, {}))

    content = context.messages[0]["content"]
    assert "real-42" in content
    assert "hallucinated-99" not in content


async def test_transfer_falls_back_to_the_model_argument_when_nothing_captured():
    state = tsa.CallState()  # no id captured yet
    context = _fresh_identify_context()
    handler = tsa._make_transfer_handler(tsa.BOOK, state)

    await handler(_params({"patient_id": "pat-7"}, context, {}))

    assert "pat-7" in context.messages[0]["content"]


async def test_register_tools_wires_capture_so_a_transfer_reuses_the_real_id(monkeypatch):
    """End-to-end: the registered find_patient handler feeds the transfer handler."""

    async def fake_find(**kwargs):
        return {"id": "db-77", "full_name": "Jane Doe"}

    monkeypatch.setitem(tsa.TOOL_HANDLERS, "find_patient", fake_find)

    llm = _FakeLLM()
    tsa.register_tools(llm)

    # Drive the genuine registered find_patient handler so the id is captured.
    await llm.registered["find_patient"](_params({"full_name": "Jane Doe"}, _fresh_identify_context(), {}))

    # A transfer carrying a wrong model-supplied id must still use the captured one.
    context = _fresh_identify_context()
    await llm.registered["transfer_to_booking"](_params({"patient_id": "wrong"}, context, {}))

    content = context.messages[0]["content"]
    assert "db-77" in content
    assert "wrong" not in content


# --- The call's "today" anchor survives a phase swap ------------------------


async def test_phase_swap_preserves_the_call_now_anchor():
    """The booker prompt must keep the identifier's 'today', not reset to wall-clock."""
    fixed = datetime(2026, 7, 1, 10, 0, tzinfo=ZoneInfo("Europe/Madrid"))  # a Wednesday
    state = tsa.CallState(now=fixed, patient_id="pat-1")
    context = _fresh_identify_context()
    handler = tsa._make_transfer_handler(tsa.BOOK, state)

    await handler(_params({}, context, {}))

    assert "2026-07-01" in context.messages[0]["content"]
