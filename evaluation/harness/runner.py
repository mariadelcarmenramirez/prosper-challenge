"""The conversation runner: drives one simulated call end-to-end, text-only.

It alternates between the agent under test and the LLM-simulated caller, exactly
like a phone call but with no audio. Per agent turn it runs the architecture's real
tool-calling loop (dispatching each model tool call through the genuine registered
handler, which carries the real ``CallGuard`` and, for the specialist, the real
phase swap), measures the wall-clock latency of the whole turn (nested worker calls
included), and records everything on the trace. The call ends when the agent
signals a graceful stop (``stop: true`` / ``EndTaskFrame``), the caller hangs up,
or a safety turn cap is hit.
"""

from __future__ import annotations

import json
import time
from typing import Any

# Reuse the supervisor's schema/tool-call (de)serialization so the eval speaks the
# exact same OpenAI wire format the production worker loop does.
from voice_agent.architectures.supervisor import _serialize_tool_call, _to_openai_tool

from .adapters import AgentSetup
from .caller import CallerTurn, SimulatedCaller
from .client import InstrumentedClient
from .shim import FakeFunctionCallParams
from .trace import ConversationTrace

# Kick-off mirrors bot.py's on_client_connected: a system nudge to open the call.
KICKOFF = (
    "Greet the caller, introduce yourself as the Prosper Health scheduling "
    "assistant, and ask for their full name, date of birth and phone number "
    "so you can find them in the system."
)

MAX_CONVERSATION_TURNS = 24  # safety cap on caller<->agent exchanges
MAX_TOOL_ITERS_PER_TURN = 16  # safety cap on the agent's tool loop within one turn


def _parse_args(raw: str | None) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _stop_seen(result: Any) -> bool:
    return isinstance(result, dict) and result.get("stop") is True


class ConversationRunner:
    def __init__(
        self,
        setup: AgentSetup,
        agent_client: InstrumentedClient,
        caller: SimulatedCaller,
        trace: ConversationTrace,
        max_turns: int = MAX_CONVERSATION_TURNS,
    ) -> None:
        self.setup = setup
        self.agent_client = agent_client
        self.caller = caller
        self.trace = trace
        self.max_turns = max_turns
        self._stop = False

    async def run(self) -> ConversationTrace:
        self.trace.start()
        # Open the call the same way the production bot does.
        self.setup.context.messages.append({"role": "system", "content": KICKOFF})
        try:
            await self._loop()
        except Exception as exc:  # never let one bad call sink the whole matrix
            self.trace.error = f"{type(exc).__name__}: {exc}"
            self.trace.end_reason = self.trace.end_reason or "error"
        return self.trace

    async def _loop(self) -> None:
        for _ in range(self.max_turns):
            agent_text = await self._agent_turn()
            if self._stop:
                self.trace.end_reason = self.trace.end_reason or "agent_ended"
                return
            turn: CallerTurn = await self._caller_turn(agent_text)
            if turn.hang_up:
                self.trace.end_reason = "caller_hangup"
                return
            self.setup.context.messages.append({"role": "user", "content": turn.text})
        self.trace.end_reason = "max_turns"

    async def _agent_turn(self) -> str:
        """Run the agent's tool loop until it produces a spoken turn; time the whole thing."""
        t0 = time.perf_counter()
        spoken = ""
        tools = [_to_openai_tool(s) for s in self.setup.context.tools.standard_tools]
        for _ in range(MAX_TOOL_ITERS_PER_TURN):
            resp = await self.agent_client.chat.completions.create(
                model=self.setup.model,
                messages=self.setup.context.messages,
                tools=tools or None,
            )
            message = resp.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                spoken = message.content or ""
                self.setup.context.messages.append({"role": "assistant", "content": spoken})
                break
            self.setup.context.messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [_serialize_tool_call(tc) for tc in tool_calls],
                }
            )
            for tc in tool_calls:
                result = await self._dispatch(tc.function.name, _parse_args(tc.function.arguments))
                self.setup.context.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
                )
                if _stop_seen(result) or self.setup.llm.end_requested:
                    self._stop = True
                    self.trace.end_reason = self._reason(result)
            # A phase swap (specialist) may have changed the offered tools mid-turn.
            tools = [_to_openai_tool(s) for s in self.setup.context.tools.standard_tools]
        latency = time.perf_counter() - t0
        self.trace.agent_turn_latencies.append(round(latency, 4))
        self.trace.add("agent_turn", text=spoken, latency=round(latency, 4))
        return spoken

    def _reason(self, result: Any) -> str:
        if isinstance(result, dict) and result.get("reason"):
            return f"stop:{result['reason']}"
        return "stop:end_call"

    async def _dispatch(self, name: str, args: dict) -> Any:
        handler = self.setup.llm.functions.get(name)
        if handler is None:
            result = {"error": f"unknown tool {name}"}
            self.trace.add("tool_call", name=name, args=args, result=result, unknown=True)
            return result
        params = FakeFunctionCallParams(arguments=args, llm=self.setup.llm, context=self.setup.context)
        t0 = time.perf_counter()
        await handler(params)
        latency = time.perf_counter() - t0
        result = params.result
        self.trace.add(
            "tool_call", name=name, args=args, result=result, latency=round(latency, 4)
        )
        return result

    async def _caller_turn(self, agent_text: str) -> CallerTurn:
        t0 = time.perf_counter()
        turn = await self.caller.respond(agent_text)
        latency = time.perf_counter() - t0
        self.trace.caller_turn_latencies.append(round(latency, 4))
        self.trace.add("caller_turn", text=turn.text, hang_up=turn.hang_up, latency=round(latency, 4))
        return turn
