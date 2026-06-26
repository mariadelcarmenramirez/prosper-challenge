"""Unit tests for tool_implementations — httpx mocked with respx (no EHR API needed)."""

import httpx
import respx

from voice_agent.tools import implementations as tool_implementations

BASE = tool_implementations.EHR_BASE_URL


async def test_confirm_patient_data_returns_validation_result():
    payload = {
        "valid": True,
        "full_name": "Jane Doe",
        "date_of_birth": "1990-01-01",
        "phone": "+34600000000",
        "issues": [],
    }
    with respx.mock(base_url=BASE) as mock:
        mock.post("/patients/validate").mock(return_value=httpx.Response(200, json=payload))
        result = await tool_implementations.confirm_patient_data(
            "  Jane Doe ", "1990-01-01", "+34 600 000 000"
        )
    assert result["valid"] is True
    assert result["phone"] == "+34600000000"
    assert result["issues"] == []


async def test_find_patient_returns_none_on_404():
    with respx.mock(base_url=BASE) as mock:
        mock.get("/patients/find").mock(return_value=httpx.Response(404, json={"detail": "x"}))
        result = await tool_implementations.find_patient("Ghost", "1990-01-01", "+34600000000")
    assert result is None


async def test_find_patient_returns_dict_on_200():
    patient = {"id": "abc", "full_name": "Jane Doe", "date_of_birth": "1990-01-01"}
    with respx.mock(base_url=BASE) as mock:
        mock.get("/patients/find").mock(return_value=httpx.Response(200, json=patient))
        result = await tool_implementations.find_patient("Jane Doe", "1990-01-01", "+34600000000")
    assert result == patient


async def test_create_appointment_returns_error_on_409():
    with respx.mock(base_url=BASE) as mock:
        mock.post("/appointments").mock(
            return_value=httpx.Response(409, json={"detail": "That slot is no longer available."})
        )
        result = await tool_implementations.create_appointment("pid", "2026-07-06T10:00:00")
    assert "error" in result
    assert result["status_code"] == 409


async def test_network_failure_returns_friendly_error():
    with respx.mock(base_url=BASE) as mock:
        mock.get("/slots").mock(side_effect=httpx.ConnectError("down"))
        result = await tool_implementations.list_availability_slots(date="2026-07-06")
    assert "error" in result
