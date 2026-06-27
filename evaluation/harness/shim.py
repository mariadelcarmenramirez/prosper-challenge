"""A minimal, audio-free Pipecat shim so the *real* agent handlers run unchanged.

The whole evaluation rests on the fact that the agent's "brain" is decoupled from
audio (see the architecture modules). The only things the production tool handlers
touch are a tiny slice of Pipecat's surface:

* ``llm.register_function(name, handler)`` — how every architecture wires its tools.
* ``params.arguments`` / ``params.result_callback`` — how a handler receives args
  and returns a result.
* ``params.llm.push_frame(EndTaskFrame(), ...)`` — how ``end_call`` ends a call.
* ``params.context.{messages,set_messages,set_tools,tools}`` — how an architecture
  reads and updates its system prompt and tool subset mid-call.

This module reimplements *only* that slice with in-memory objects, so the eval can
drive ``single`` and ``supervisor`` through their genuine code paths (real
prompts, ``make_handler``, ``CallGuard``, nested workers) without a transport,
audio, or a websocket. Nothing here mocks agent behaviour; it only stands in for
the pipeline plumbing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import EndTaskFrame


class FakeLLMContext:
    """Stand-in for Pipecat's ``LLMContext``: holds the conversation + tool subset.

    The runner uses ``messages`` as the single source of truth for the conversation
    and ``tools`` for the offered functions. An architecture can mutate both via
    ``set_messages`` / ``set_tools`` from inside a handler, exactly as it does in
    production, so changes take effect on the very next agent turn.
    """

    def __init__(self, messages: list[dict], tools: ToolsSchema) -> None:
        self.messages = messages
        self.tools = tools

    def set_messages(self, messages: list[dict]) -> None:
        self.messages = messages

    def set_tools(self, tools: ToolsSchema) -> None:
        self.tools = tools


class FakeLLM:
    """Stand-in for the Pipecat LLM service: a function registry + a frame sink.

    Architectures call ``register_function`` to wire each tool handler; the runner
    later looks the handler up by name to dispatch a model tool call. ``push_frame``
    is how ``end_call`` signals a graceful hang-up — we just record that an
    ``EndTaskFrame`` was pushed so the runner can end the conversation.
    """

    def __init__(self) -> None:
        self.functions: dict[str, Callable[[Any], Awaitable[None]]] = {}
        self.end_requested = False

    def register_function(self, name: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        self.functions[name] = handler

    async def push_frame(self, frame: Any, direction: Any = None) -> None:
        if isinstance(frame, EndTaskFrame):
            self.end_requested = True


@dataclass
class FakeFunctionCallParams:
    """Stand-in for Pipecat's ``FunctionCallParams`` handed to each tool handler."""

    arguments: dict
    llm: FakeLLM
    context: FakeLLMContext
    _result: dict = field(default_factory=dict)

    async def result_callback(self, result: Any) -> None:
        self._result["value"] = result

    @property
    def result(self) -> Any:
        return self._result.get("value")
