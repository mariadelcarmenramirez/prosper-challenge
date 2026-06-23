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

import ehr_tools
from tools import TOOL_SCHEMAS

# Tool name -> the coroutine that implements it. Names must match tools.py.
TOOL_HANDLERS: dict[str, Callable[..., Awaitable[Any]]] = {
    "find_patient": ehr_tools.find_patient,
    "create_patient": ehr_tools.create_patient,
    "list_availability_slots": ehr_tools.list_availability_slots,
    "list_patient_appointments": ehr_tools.list_patient_appointments,
    "create_appointment": ehr_tools.create_appointment,
    "confirm_appointment": ehr_tools.confirm_appointment,
    "cancel_appointment": ehr_tools.cancel_appointment,
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
