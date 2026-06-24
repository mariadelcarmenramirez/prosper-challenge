"""The audio-free "brain": binds LLM tools to their EHR implementations.

Imported by both ``bot.py`` (audio pipeline) and the test/eval harness so they
build the LLM and register the same handlers identically. Nothing here depends
on the transport or audio — only on the LLM service abstraction.
"""

import os
from collections.abc import Awaitable, Callable
from typing import Any

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService

import tool_implementations
from tool_schemas import TOOL_SCHEMAS

# Tool name -> the coroutine that implements it. Names must match tool_schemas.py.
TOOL_HANDLERS: dict[str, Callable[..., Awaitable[Any]]] = {
    "find_patient": tool_implementations.find_patient,
    "create_patient": tool_implementations.create_patient,
    "list_availability_slots": tool_implementations.list_availability_slots,
    "list_patient_appointments": tool_implementations.list_patient_appointments,
    "create_appointment": tool_implementations.create_appointment,
    "confirm_appointment": tool_implementations.confirm_appointment,
    "cancel_appointment": tool_implementations.cancel_appointment,
}


def _make_handler(coro: Callable[..., Awaitable[Any]]):
    async def handler(params: FunctionCallParams) -> None:
        result = await coro(**params.arguments)
        # find_patient returns None when the patient is unknown; give the model
        # an explicit, JSON-serializable signal it can branch on.
        if result is None:
            result = {"found": False}
        await params.result_callback(result)

    return handler


def register_tools(llm: OpenAILLMService) -> None:
    """Register every tool handler on the LLM service."""
    for name, coro in TOOL_HANDLERS.items():
        llm.register_function(name, _make_handler(coro))


def get_tools_schema() -> ToolsSchema:
    """Tool schemas to attach to the LLM context."""
    return ToolsSchema(standard_tools=TOOL_SCHEMAS)


def build_llm(model: str = "gpt-4.1") -> OpenAILLMService:
    """Construct the LLM service identically for the bot and the eval harness."""
    return OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model=model)
