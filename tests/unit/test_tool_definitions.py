"""Unit tests for the tool definitions and the handler registry."""

from voice_agent.core.runtime import TOOL_HANDLERS
from voice_agent.tools.schemas import TOOL_SCHEMAS

REQUIRED_BY_README = {
    "create_patient",
    "find_patient",
    "list_availability_slots",
    "create_appointment",
    "cancel_appointment",
}


def test_all_required_tools_defined():
    names = {t.name for t in TOOL_SCHEMAS}
    assert REQUIRED_BY_README <= names


def test_every_tool_has_description_and_parameters():
    for tool in TOOL_SCHEMAS:
        assert tool.description, f"{tool.name} missing description"
        assert isinstance(tool.properties, dict)
        assert isinstance(tool.required, list)
        # Every required param must be declared in properties.
        assert set(tool.required) <= set(tool.properties)


def test_handlers_and_schemas_are_in_sync():
    """No orphan tools: every schema has a handler and vice-versa."""
    schema_names = {t.name for t in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_HANDLERS)
