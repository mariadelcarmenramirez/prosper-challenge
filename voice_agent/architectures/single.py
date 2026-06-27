"""Single-context architecture: one LLM, one prompt, the full tool set.

The simplest of the three scheduling "brains": every EHR tool is offered at once
and a single system prompt (``prompts.build_system_prompt``) drives the whole
identify -> book | cancel flow. ``bot.py`` selects it with ``AGENT_ARCH=single``.

It is now a thin architecture on top of the shared kernel — loop safety lives in
``guard`` and the Pipecat tool wiring in ``runtime`` — so it no longer carries any
machinery the other two architectures need; none of the three depends on another.
``build_llm`` is re-exported so ``bot.py`` can treat every architecture the same.
"""

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai.llm import OpenAILLMService

from ..core.guard import CallGuard
from ..core.runtime import (  
    TOOL_HANDLERS,
    build_llm,
    make_handler,
)
from ..tools.schemas import TOOL_SCHEMAS


def register_tools(llm: OpenAILLMService, guard: CallGuard | None = None) -> CallGuard:
    """Register every tool handler on the LLM service.

    A fresh ``CallGuard`` is created per call (one per ``register_tools``) so its
    counters are scoped to a single conversation. The guard is returned so the
    caller (or a test/eval harness) can inspect or pre-seed the per-call state.
    """
    guard = guard or CallGuard()
    for name, coro in TOOL_HANDLERS.items():
        llm.register_function(name, make_handler(name, coro, guard))
    return guard


def get_tools_schema() -> ToolsSchema:
    """Tool schemas to attach to the LLM context."""
    return ToolsSchema(standard_tools=TOOL_SCHEMAS)
