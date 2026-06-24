"""Unit tests for the system prompt builder."""

from datetime import datetime
from zoneinfo import ZoneInfo

from prompts import build_system_prompt

MADRID = ZoneInfo("Europe/Madrid")


def test_prompt_contains_injected_today():
    now = datetime(2026, 6, 23, 10, 0, tzinfo=MADRID)  # a Tuesday
    prompt = build_system_prompt(now=now)
    assert "2026-06-23" in prompt
    assert "Tuesday" in prompt


def test_prompt_contains_clinic_name():
    assert "Prosper Health" in build_system_prompt(clinic_name="Prosper Health")


def test_prompt_describes_all_flows():
    prompt = build_system_prompt().lower()
    for keyword in ["register", "book", "cancel", "confirm_patient_data", "find_patient", "confirm"]:
        assert keyword in prompt


def test_prompt_states_clinic_hours():
    prompt = build_system_prompt()
    assert "Monday to Friday" in prompt
    assert "9:00" in prompt and "18:00" in prompt
