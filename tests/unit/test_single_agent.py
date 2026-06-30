from types import SimpleNamespace

from pipecat.frames.frames import EndTaskFrame, TTSSpeakFrame

import voice_agent.architectures.single as single
from voice_agent.core.guard import MAX_TOTAL_TOOL_CALLS, CallGuard
from voice_agent.tools.schemas import TOOL_SCHEMAS

# The full EHR tool set the single agent exposes (handlers == schemas).
ALL_TOOLS = set(single.TOOL_HANDLERS)


class _FakeLLM:
    """Stand-in for the Pipecat LLM service: just records registered handlers."""

    def __init__(self) -> None:
        self.functions: dict = {}

    def register_function(self, name, handler) -> None:
        self.functions[name] = handler


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


# --- Tool surface -----------------------------------------------------------


def test_single_registers_the_full_ehr_tool_set():
    llm = _FakeLLM()
    single.register_tools(llm)
    # Every EHR tool is offered at once — the defining trait of the single-context
    # architecture, versus the supervisor's three delegation tools.
    assert set(llm.functions) == ALL_TOOLS
    assert len(llm.functions) == 8


def test_tools_schema_exposes_every_ehr_tool():
    schema = single.get_tools_schema()
    names = {t.name for t in schema.standard_tools}
    assert names == {t.name for t in TOOL_SCHEMAS}
    assert names == ALL_TOOLS  # schemas and handlers stay in lockstep


# --- Per-call guard lifecycle -----------------------------------------------


def test_register_tools_returns_a_fresh_guard_per_call():
    g1 = single.register_tools(_FakeLLM())
    g2 = single.register_tools(_FakeLLM())
    assert isinstance(g1, CallGuard) and isinstance(g2, CallGuard)
    assert g1 is not g2  # counters are scoped to a single conversation
    assert g1.total_calls == 0


def test_register_tools_accepts_a_preseeded_guard():
    guard = CallGuard()
    guard.total_calls = 5
    returned = single.register_tools(_FakeLLM(), guard)
    assert returned is guard  # the caller's guard is reused, not replaced
    assert returned.total_calls == 5


# --- Wiring: the returned guard governs the registered handlers --------------


async def test_returned_guard_governs_the_registered_handlers():
    """The guard register_tools hands back is the one wired into every handler: at
    the global ceiling a registered handler returns the stop signal and ends the
    call, without ever running the EHR tool."""
    llm = _FakeLLM()
    guard = single.register_tools(llm)
    guard.total_calls = MAX_TOTAL_TOOL_CALLS - 1  # the next call crosses the ceiling

    calls: dict = {}
    await llm.functions["find_patient"](_params({"full_name": "x"}, calls))

    assert calls["result"] == {"stop": True, "reason": "tool_call_limit"}
    assert len(calls["frames"]) == 1 and isinstance(calls["frames"][0], EndTaskFrame)


async def test_spoken_ack_is_debounced_across_a_tool_burst(monkeypatch):
    """All handlers share one ack clock, so a burst of tools in a single turn (e.g.
    find_patient then create_patient) speaks one acknowledgement, not one per tool."""

    async def fake_tool(**kwargs):
        return {"ok": True}

    for name in ("find_patient", "create_patient"):
        monkeypatch.setitem(single.TOOL_HANDLERS, name, fake_tool)

    llm = _FakeLLM()
    single.register_tools(llm)

    calls: dict = {}
    await llm.functions["find_patient"](_params({}, calls))
    await llm.functions["create_patient"](_params({}, calls))

    spoken = [f for f in calls.get("frames", []) if isinstance(f, TTSSpeakFrame)]
    assert len(spoken) == 1
