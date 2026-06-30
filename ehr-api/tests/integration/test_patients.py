"""Integration tests for the patient endpoints."""

PATIENT = {
    "full_name": "Jane Doe",
    "date_of_birth": "1990-01-01",
    "phone": "+34600000000",
}


async def test_validate_patient_data_normalizes(client):
    """Whitespace and phone formatting collapse to canonical values."""
    r = await client.post(
        "/patients/validate",
        json={
            "full_name": "  Jane   Doe ",
            "date_of_birth": "1990-01-01",
            "phone": "+34 600 000 000",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["full_name"] == "Jane Doe"
    assert body["phone"] == "+34600000000"
    assert body["date_of_birth"] == "1990-01-01"
    assert body["issues"] == []


async def test_validate_patient_data_rejects_future_dob(client):
    r = await client.post(
        "/patients/validate",
        json={"full_name": "Jane Doe", "date_of_birth": "2999-01-01", "phone": "+34600000000"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["issues"]
    assert body["date_of_birth"] is None


async def test_validate_patient_data_rejects_bad_date(client):
    r = await client.post(
        "/patients/validate",
        json={"full_name": "Jane Doe", "date_of_birth": "not-a-date", "phone": "+34600000000"},
    )
    assert r.json()["valid"] is False


async def test_validate_patient_data_rejects_short_phone(client):
    r = await client.post(
        "/patients/validate",
        json={"full_name": "Jane Doe", "date_of_birth": "1990-01-01", "phone": "123"},
    )
    assert r.json()["valid"] is False


async def test_create_patient_returns_201(client):
    r = await client.post("/patients", json=PATIENT)
    assert r.status_code == 201
    body = r.json()
    assert body["full_name"] == "Jane Doe"
    assert body["id"]  # DB-generated UUID


async def test_create_patient_duplicate_returns_409(client):
    await client.post("/patients", json=PATIENT)
    r = await client.post("/patients", json=PATIENT)
    assert r.status_code == 409


async def test_find_patient_returns_patient(client):
    created = (await client.post("/patients", json=PATIENT)).json()
    r = await client.get(
        "/patients/find",
        params={
            "full_name": PATIENT["full_name"],
            "date_of_birth": PATIENT["date_of_birth"],
            "phone": PATIENT["phone"],
        },
    )
    assert r.status_code == 200
    assert r.json()["id"] == created["id"]


async def test_find_patient_not_found_returns_404(client):
    r = await client.get(
        "/patients/find",
        params={
            "full_name": "Nobody Here",
            "date_of_birth": "2000-12-31",
            "phone": "+34000000000",
        },
    )
    assert r.status_code == 404


async def test_find_patient_wrong_phone_not_found(client):
    """Phone is part of the identity: a different phone must not match."""
    await client.post("/patients", json=PATIENT)
    r = await client.get(
        "/patients/find",
        params={
            "full_name": PATIENT["full_name"],
            "date_of_birth": PATIENT["date_of_birth"],
            "phone": "+34999999999",
        },
    )
    assert r.status_code == 404
