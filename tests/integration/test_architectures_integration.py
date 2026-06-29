import json
import uuid
from types import SimpleNamespace

import voice_agent.architectures.single as single
import voice_agent.architectures.supervisor as sup
from voice_agent.core.guard import CallGuard
from voice_agent.core.runtime import TOOL_HANDLERS
from voice_agent.tools import implementations as tool_implementations


# --- Single architecture: dispatch through its registered Pipecat handlers ---


class _FakeLLM:
    """Stand-in for the Pipecat LLM service: records the registered handlers."""

    def __init__(self) -> None:
        self.functions: dict = {}

    def register_function(self, name, handler) -> None:
        self.functions[name] = handler


def _params(arguments):
    captured: dict = {}

    async def result_callback(result):
        captured["result"] = result

    async def push_frame(frame, direction=None):
        pass

    params = SimpleNamespace(
        arguments=arguments,
        result_callback=result_callback,
        llm=SimpleNamespace(push_frame=push_frame),
    )
    return params, captured


async def _dispatch_single(llm: _FakeLLM, name: str, arguments: dict):
    params, captured = _params(arguments)
    await llm.functions[name](params)
    return captured["result"]


async def _register_via_single(full_name: str, dob: str, phone: str) -> dict:
    """Identify-and-register a caller through the single agent's real dispatch path.

    ``register_tools`` wires every real EHR coroutine behind a loop-safe handler;
    dispatching the validate -> find -> create steps as the model would makes each
    one a real HTTP call to the test backend.
    """
    llm = _FakeLLM()
    single.register_tools(llm)

    valid = await _dispatch_single(
        llm, "confirm_patient_data",
        {"full_name": full_name, "date_of_birth": dob, "phone": phone},
    )
    assert valid["valid"] is True
    identity = {
        "full_name": valid["full_name"],
        "date_of_birth": valid["date_of_birth"],
        "phone": valid["phone"],
    }
    # Not registered yet: find_patient 404s, which the handler normalizes.
    assert await _dispatch_single(llm, "find_patient", identity) == {"found": False}
    return await _dispatch_single(llm, "create_patient", identity)


# --- Supervisor architecture: drive the worker loop with a scripted client ----


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments))
    )


class _FakeOpenAI:
    """Returns a scripted sequence of chat-completion responses (no network)."""

    def __init__(self, responses):
        self._responses = list(responses)

        async def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=self._responses.pop(0))])

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


async def _register_via_supervisor(full_name: str, dob: str, phone: str) -> tuple[dict, str]:
    """Identify-and-register a caller through the supervisor's identification worker.

    The model is scripted, but the worker runs the real ``TOOL_HANDLERS`` through
    ``_run_agent_loop`` — the exact code path production takes — so confirm/find/
    create are genuine HTTP calls to the test backend. Returns the create result
    (captured off the worker's nested-tool recorder) and the worker's report.
    """
    identity = {"full_name": full_name, "date_of_birth": dob, "phone": phone}
    client = _FakeOpenAI(
        [
            _msg(tool_calls=[_tool_call("c1", "confirm_patient_data", identity)]),
            _msg(tool_calls=[_tool_call("c2", "find_patient", identity)]),
            _msg(tool_calls=[_tool_call("c3", "create_patient", identity)]),
            _msg(content=f"Registered {full_name}."),
        ]
    )
    recorded: list[dict] = []

    report = await sup._run_agent_loop(
        system_prompt=sup.IDENTIFIER_WORKER.build_prompt(),
        tools=sup.IDENTIFIER_WORKER.openai_tools(),
        task="Validate, then find or register the caller.",
        client=client,
        guard=CallGuard(),
        state=sup.SessionState(),
        handlers=TOOL_HANDLERS,
        recorder=lambda event, **data: recorded.append(data),
    )
    created = next(e["result"] for e in recorded if e["name"] == "create_patient")
    return created, report


# --- The cross-architecture integration check -------------------------------


async def test_both_architectures_register_a_patient_against_the_live_backend():
    """Exercise BOTH architectures end-to-end against the live EHR.

    Each drives its own real tool-dispatch path to register a distinct caller, and
    every tool call is a genuine HTTP round-trip to the isolated test backend. We
    then confirm both records persisted via a fresh find_patient straight against
    the API — not through either architecture.
    """
    dob = "1990-01-01"
    name_single = f"Single Caller {uuid.uuid4().hex[:8]}"
    name_super = f"Supervisor Caller {uuid.uuid4().hex[:8]}"
    phone_single, phone_super = "+34600000001", "+34600000002"

    # 1) Single-context architecture -> backend.
    created_single = await _register_via_single(name_single, dob, phone_single)
    assert "id" in created_single

    # 2) Supervisor-workers architecture -> backend.
    created_super, report = await _register_via_supervisor(name_super, dob, phone_super)
    assert "id" in created_super
    assert report  # the worker produced a report for the supervisor

    # 3) Independently verify both patients landed in the live EHR.
    found_single = await tool_implementations.find_patient(name_single, dob, phone_single)
    found_super = await tool_implementations.find_patient(name_super, dob, phone_super)
    assert found_single is not None and found_single["id"] == created_single["id"]
    assert found_super is not None and found_super["id"] == created_super["id"]
    assert found_single["id"] != found_super["id"]  # two distinct real records
