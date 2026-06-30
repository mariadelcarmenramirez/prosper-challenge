"""HTTP client for the EHR API — one async function per endpoint.

These are the functions the LLM calls during a conversation (wired up in
``agent.py``). They are deliberately free of any Pipecat/audio dependency so the
behaviour can be unit-tested and run through a text-only eval.

On any non-2xx response (other than a 404 from ``find_patient``) the helpers
return ``{"error": "..."}`` so the agent can apologise gracefully instead of
crashing the call.
"""

import asyncio
import os

import httpx

EHR_BASE_URL = os.environ.get("EHR_BASE_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(10.0)

# Reads (GET) are safe to retry: re-running them can't change server state. Writes
# (POST/DELETE) are NOT retried here — a request that reached the server before the
# connection dropped may have already created the patient/appointment, so a blind
# retry could double-book. Those need an idempotency key to be retried safely.
GET_MAX_ATTEMPTS = 3  # total attempts for a GET (1 try + up to 2 retries)
RETRY_BACKOFF = 0.3  # seconds; grows per attempt (0.3s, 0.6s, ...)


def _is_retryable(resp: httpx.Response) -> bool:
    """Only server-side errors (5xx) are worth retrying; a 4xx won't fix itself."""
    return resp.status_code >= 500


async def _request(method: str, path: str, **kwargs):
    """Issue a request and return parsed JSON, or an error dict.

    GET requests are retried a few times on transient failures (network error or
    5xx); every other method is attempted exactly once to avoid duplicate writes.
    """
    attempts = GET_MAX_ATTEMPTS if method.upper() == "GET" else 1
    fallback = {"error": "I'm having trouble reaching the scheduling system right now."}

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(base_url=EHR_BASE_URL, timeout=TIMEOUT) as client:
                resp = await client.request(method, path, **kwargs)
        except httpx.HTTPError:
            fallback = {"error": "I'm having trouble reaching the scheduling system right now."}
        else:
            if resp.status_code >= 400:
                if _is_retryable(resp) and attempt < attempts - 1:
                    fallback = {"error": _detail(resp), "status_code": resp.status_code}
                else:
                    return {"error": _detail(resp), "status_code": resp.status_code}
            else:
                return resp.json()

        if attempt < attempts - 1:
            await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))

    return fallback


def _detail(resp: httpx.Response) -> str:
    try:
        return resp.json().get("detail", "Request failed.")
    except Exception:
        return "Request failed."


# --- Patients ---------------------------------------------------------------


async def confirm_patient_data(full_name: str, date_of_birth: str, phone: str):
    """Validate and normalize the caller's identity before any lookup.

    Returns ``{"valid": bool, "full_name", "date_of_birth", "phone", "issues":
    [...]}``. Always call this before ``find_patient`` and, when ``valid`` is
    true, pass the normalized fields it returns on to find/create_patient."""
    return await _request(
        "POST",
        "/patients/validate",
        json={"full_name": full_name, "date_of_birth": date_of_birth, "phone": phone},
    )


async def find_patient(full_name: str, date_of_birth: str, phone: str):
    """Look up a patient by full identity. Returns the patient dict, or ``None``
    if they are not registered yet."""
    async with httpx.AsyncClient(base_url=EHR_BASE_URL, timeout=TIMEOUT) as client:
        try:
            resp = await client.get(
                "/patients/find",
                params={"full_name": full_name, "date_of_birth": date_of_birth, "phone": phone},
            )
        except httpx.HTTPError:
            return {"error": "I'm having trouble reaching the scheduling system right now."}
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        return {"error": _detail(resp), "status_code": resp.status_code}
    return resp.json()


async def create_patient(full_name: str, date_of_birth: str, phone: str):
    """Register a new patient; the EHR generates the patient_id."""
    return await _request(
        "POST",
        "/patients",
        json={
            "full_name": full_name,
            "date_of_birth": date_of_birth,
            "phone": phone,
        },
    )


# --- Availability & appointments -------------------------------------------


async def list_availability_slots(
    date: str | None = None, start: str | None = None, end: str | None = None
):
    """Free, bookable slots for a day or range (YYYY-MM-DD)."""
    params = {k: v for k, v in {"date": date, "start": start, "end": end}.items() if v}
    return await _request("GET", "/slots", params=params)


async def list_patient_appointments(patient_id: str):
    """A patient's upcoming, still-scheduled (cancellable) appointments."""
    return await _request("GET", "/appointments", params={"patient_id": patient_id})


async def create_appointment(patient_id: str, starts_at: str):
    """Reserve a slot: creates a held appointment that expires shortly. Call
    ``confirm_appointment`` once the caller says yes."""
    return await _request(
        "POST", "/appointments", json={"patient_id": patient_id, "starts_at": starts_at}
    )


async def confirm_appointment(appointment_id: str):
    """Finalize a held appointment (held -> scheduled)."""
    return await _request("POST", f"/appointments/{appointment_id}/confirm")


async def cancel_appointment(appointment_id: str):
    """Cancel an appointment (soft delete). Also releases a held reservation."""
    return await _request("DELETE", f"/appointments/{appointment_id}")
