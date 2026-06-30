import json

import httpx
import respx

from voice_agent.tools import implementations as tool_implementations


def _base() -> str:
    """Read the base URL the client will actually use *now*, not a snapshot.

    The implementation reads the ``EHR_BASE_URL`` module-global at call time, and
    the integration suite's session fixture repoints it at its own test API. Binding
    respx to a value captured at import would then mock the wrong host whenever those
    suites run first in the same session, so we read it live to mock the real target.
    """
    return tool_implementations.EHR_BASE_URL


async def test_confirm_patient_data_returns_validation_result():
    """The client forwards the caller's raw inputs and returns the server's verdict
    verbatim. Normalization is the server's job (covered in the EHR suite), so here
    we assert faithful pass-through — not the phone value we mocked ourselves."""
    payload = {
        "valid": True,
        "full_name": "Jane Doe",
        "date_of_birth": "1990-01-01",
        "phone": "+34600000000",
        "issues": [],
    }
    with respx.mock(base_url=_base()) as mock:
        route = mock.post("/patients/validate").mock(
            return_value=httpx.Response(200, json=payload)
        )
        result = await tool_implementations.confirm_patient_data(
            "  Jane Doe ", "1990-01-01", "+34 600 000 000"
        )
    # The raw, un-normalized inputs are sent for the server to validate.
    assert json.loads(route.calls.last.request.content) == {
        "full_name": "  Jane Doe ",
        "date_of_birth": "1990-01-01",
        "phone": "+34 600 000 000",
    }
    # The server's verdict is returned to the agent unchanged.
    assert result == payload


async def test_find_patient_returns_none_on_404():
    with respx.mock(base_url=_base()) as mock:
        mock.get("/patients/find").mock(return_value=httpx.Response(404, json={"detail": "x"}))
        result = await tool_implementations.find_patient("Ghost", "1990-01-01", "+34600000000")
    assert result is None


async def test_find_patient_returns_dict_on_200():
    patient = {"id": "abc", "full_name": "Jane Doe", "date_of_birth": "1990-01-01"}
    with respx.mock(base_url=_base()) as mock:
        mock.get("/patients/find").mock(return_value=httpx.Response(200, json=patient))
        result = await tool_implementations.find_patient("Jane Doe", "1990-01-01", "+34600000000")
    assert result == patient


async def test_create_appointment_returns_error_on_409():
    with respx.mock(base_url=_base()) as mock:
        mock.post("/appointments").mock(
            return_value=httpx.Response(409, json={"detail": "That slot is no longer available."})
        )
        result = await tool_implementations.create_appointment("pid", "2026-07-06T10:00:00")
    assert "error" in result
    assert result["status_code"] == 409


async def test_network_failure_returns_friendly_error():
    with respx.mock(base_url=_base()) as mock:
        mock.get("/slots").mock(side_effect=httpx.ConnectError("down"))
        result = await tool_implementations.list_availability_slots(date="2026-07-06")
    assert "error" in result
