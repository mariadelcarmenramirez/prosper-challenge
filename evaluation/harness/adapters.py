"""Uniform adapters over the three agent architectures.

The runner shouldn't care whether it is driving the single-context agent, the
phased specialist, or the supervisor — it only needs, per call: an initial system
prompt, an initial tool subset, and a registry to dispatch tool calls into. This
module builds each architecture through its *real* public surface (so the genuine
prompts, ``CallGuard``, phase swaps and worker loops are exercised) and returns
that uniform :class:`AgentSetup`.

The one architecture that needs special handling is the supervisor: its nested
worker loop runs its own OpenAI client, so we (a) inject the instrumented client
so worker spend/latency is metered too, and (b) force the worker model to match
the model under test, so "supervisor on gpt-5-nano" really means the whole system
runs on gpt-5-nano rather than a gpt-5-nano brain delegating to gpt-4.1 workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from voice_agent.architectures import single, specialist, supervisor
from voice_agent.architectures.supervisor import _run_agent_loop as _ORIG_WORKER_LOOP
from voice_agent.core.guard import CallGuard
from voice_agent.core.prompts import build_system_prompt

from .shim import FakeLLM, FakeLLMContext

ARCHITECTURES = ("single", "specialist", "supervisor")


@dataclass
class AgentSetup:
    """Everything the runner needs to drive one architecture for one call."""

    arch: str
    model: str
    llm: FakeLLM
    context: FakeLLMContext
    guard: CallGuard


def _build_single(model: str, now: datetime | None) -> AgentSetup:
    llm = FakeLLM()
    guard = single.register_tools(llm)
    context = FakeLLMContext(
        messages=[{"role": "system", "content": build_system_prompt(now=now)}],
        tools=single.get_tools_schema(),
    )
    return AgentSetup("single", model, llm, context, guard)


def _build_specialist(model: str, now: datetime | None) -> AgentSetup:
    llm = FakeLLM()
    guard = specialist.register_tools(llm)
    context = FakeLLMContext(
        messages=[{"role": "system", "content": specialist.get_initial_system_prompt(now=now)}],
        tools=specialist.get_initial_tools_schema(),
    )
    return AgentSetup("specialist", model, llm, context, guard)


def _patch_worker_model(model: str) -> None:
    """Force the supervisor's nested worker loop onto ``model``.

    The delegation handlers call the module-global ``_run_agent_loop`` by name at
    call time, so reassigning it on the module swaps the worker model for the whole
    eval process (runs are sequential, so this is safe).
    """

    async def loop(*args: Any, **kwargs: Any) -> str:
        kwargs["model"] = model
        return await _ORIG_WORKER_LOOP(*args, **kwargs)

    supervisor._run_agent_loop = loop


def _build_supervisor(model: str, now: datetime | None, agent_client: Any) -> AgentSetup:
    _patch_worker_model(model)
    orchestrator = supervisor.Supervisor(now=now)
    orchestrator._client_obj = agent_client  # meter + shape the worker calls too
    llm = FakeLLM()
    guard = orchestrator.register_tools(llm)
    context = FakeLLMContext(
        messages=[{"role": "system", "content": orchestrator.get_initial_system_prompt()}],
        tools=orchestrator.get_initial_tools_schema(),
    )
    return AgentSetup("supervisor", model, llm, context, guard)


def build_agent(arch: str, model: str, now: datetime | None, agent_client: Any) -> AgentSetup:
    """Construct one architecture, wired to the shim, ready for the runner."""
    if arch == "single":
        return _build_single(model, now)
    if arch == "specialist":
        return _build_specialist(model, now)
    if arch == "supervisor":
        return _build_supervisor(model, now, agent_client)
    raise ValueError(f"Unknown architecture: {arch!r} (expected one of {ARCHITECTURES})")
