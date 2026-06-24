-- EHR schema. Two objects only: patients and appointments.
-- Availability is NOT a table: it is derived from the clinic's working-hours
-- rules (Mon-Fri 09:00-18:00, 1h slots) minus existing appointments. 

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS patients (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name     TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    phone         TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (full_name, date_of_birth, phone)
);


CREATE TABLE IF NOT EXISTS appointments (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id  UUID NOT NULL REFERENCES patients(id),
    starts_at   TIMESTAMPTZ NOT NULL,
    ends_at     TIMESTAMPTZ NOT NULL,
    status      TEXT NOT NULL DEFAULT 'held'
                CHECK (status IN ('held', 'scheduled', 'cancelled')),
    held_until  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS one_scheduled_per_slot
    ON appointments (starts_at) WHERE status = 'scheduled';

CREATE INDEX IF NOT EXISTS idx_appt_patient   ON appointments (patient_id);
CREATE INDEX IF NOT EXISTS idx_appt_starts_at ON appointments (starts_at);
