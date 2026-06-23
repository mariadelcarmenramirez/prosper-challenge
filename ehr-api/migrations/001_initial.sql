-- EHR schema. Two objects only: patients and appointments.
-- Availability is NOT a table: it is derived from the clinic's working-hours
-- rules (Mon-Fri 09:00-18:00, 1h slots) minus existing appointments. That way
-- the bookable catalog and the bookings can never drift out of sync, and the
-- clinic has an unbounded scheduling horizon with no seed job.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS patients (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    full_name     TEXT NOT NULL,
    date_of_birth DATE NOT NULL,
    phone         TEXT NOT NULL,
    email         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Business identity: a patient is fully identified by name + DOB + phone.
    -- All three are asked on the call and used by find_patient.
    UNIQUE (full_name, date_of_birth, phone)
);

-- An appointment is a (patient, hour) booking. ends_at = starts_at + 1h.
-- status lifecycle: held -> scheduled -> cancelled.
--   held       : tentatively reserved while the agent reads it back for
--                confirmation; auto-expires at held_until (no background job,
--                expiry is evaluated at query time).
--   scheduled  : confirmed booking.
--   cancelled  : soft delete; row stays for the audit trail.
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

-- Hard guarantee against double-booking (single doctor): at most one SCHEDULED
-- appointment per start time. Held and cancelled rows are excluded, so a slot
-- can be re-booked after a cancellation, and a confirm race resolves here.
CREATE UNIQUE INDEX IF NOT EXISTS one_scheduled_per_slot
    ON appointments (starts_at) WHERE status = 'scheduled';

CREATE INDEX IF NOT EXISTS idx_appt_patient   ON appointments (patient_id);
CREATE INDEX IF NOT EXISTS idx_appt_starts_at ON appointments (starts_at);
