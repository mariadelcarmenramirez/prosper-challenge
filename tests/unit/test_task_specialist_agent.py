"""Unit tests for the phased-specialist (sequential handoff) agent.

These cover the parts unique to ``architectures/specialist.py``: the per-phase tool
subsets, the transfer-tool routing, registration of every handler, and the core
mechanic — a ``transfer_to_*`` call swapping the live context's system prompt and
offered tool subset. Loop-safety itself is reused from ``guard.py`` and already
covered by ``test_call_guard.py``.
"""

from types import SimpleNamespace

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
