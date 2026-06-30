## Overview

This is a voice AI agent that answers phone calls for the Prosper Health clinic and handles appointment scheduling end-to-end — greeting the caller, verifying and looking up (or registering) their identity, offering open slots, and booking or cancelling appointments, entirely by voice. The caller connects over WebRTC into a [Pipecat](bot.py) pipeline that chains speech-to-text (ElevenLabs) → an OpenAI LLM (`gpt-4.1-mini`) → text-to-speech (ElevenLabs); the LLM is given the EHR operations as function-calling tools. Each tool call is a plain HTTP request to a separate FastAPI EHR API (`ehr-api/`), which is the only component that touches the database and persists every patient and appointment in Postgres. 

```
    Caller  (WebRTC audio)
       │  speech ▲ / voice ▼
       ▼
┌──────────────────────────────────────────────────┐
│  Voice agent   ·  bot.py / voice_agent  (Pipecat)│
│  STT ─▶ LLM (gpt-4.1-mini + EHR tools) ─▶ TTS   │
└──────────────────────────────────────────────────┘
       │  HTTP  (find/create patient, list slots,
       │         create/confirm/cancel appointment)
       ▼
┌──────────────────────────────────────────────────┐
│  EHR API       ·  ehr-api/  (FastAPI · asyncpg)  │
└──────────────────────────────────────────────────┘
       │  SQL
       ▼
┌──────────────────────────────────────────────────┐
│  Postgres      ·  patients, appointments         │
│                   (named docker volume)          │
└──────────────────────────────────────────────────┘
```


## Key design decisions 

### EHR API

A simple EHR service built with FastAPI. The required endpoints:

   - `create_patient` — register a new patient (name, date of birth, phone). Rejects duplicates.
   - `find_patient` — look up a patient by full identity (name + date of birth + phone).
   - `list_availability_slots` — return the clinic's free slots for a given day or date range.
   - `create_appointment` — inserts a held reservation, that expires in 5 min. Booking become real after calling `confirm_appointment`.
   - `cancel_appointment` — cancel an appointment (soft delete; releases the slot).

I added three extra endpoints to make the flow safer and to support cancellations:

   - `confirm_patient_data` — validate and normalize the caller's identity before any lookup, so we never search or insert on a typo'd date or a differently-formatted phone (which would miss the real patient or create a duplicate).
   - `list_patient_appointments` — list a patient's upcoming appointments, so the agent knows which one to cancel.
   - `confirm_appointment` — confirm a proposed slot before it is finalized.

The data model is just two tables: `patients` and `appointments`. Each row carries its own identifier — a `patient_id` for patients and an `appointment_id` for appointments — so every object can be referenced unambiguously in later operations. There is no availability/slots table — free slots are derived on the fly from the clinic's working hours (Mon–Fri, 09:00–18:00, 1-hour slots) minus the appointments already taken.

Booking is two-phase, which the `appointments` table supports with a `status` column (`held` → `scheduled` → `cancelled`) and a `held_until` timestamp. `create_appointment` inserts a row as `held` with `held_until = now + 5 min`, reserving the slot; `confirm_appointment` promotes it to `scheduled` once the caller agrees, provided the hold hasn't expired. Held and scheduled slots are both excluded from availability, so a proposed time can't be taken by someone else mid-conversation, while expired holds free themselves. A partial unique index on `starts_at` (where `status = 'scheduled'`) guarantees two confirmed bookings can never share a slot. 

**Persistence.** Data survives restarts for two reasons: the API is stateless (nothing is kept in memory, every read/write hits the database), and Postgres stores its data on a named Docker volume (`postgres_data`) that outlives the container. 



### Conversation flow & tool design

The call follows one simple path:

1. **Incoming call** — greet the caller and collect their full name, date of birth and phone.
2. **Confirm the data** — `confirm_patient_data` validates and cleans up those values; the agent reads them back and waits for a clear "yes" before any lookup, so we never search or register on a typo.
3. **Find or register** — `find_patient` looks the caller up in the database; if they are not there, `create_patient` registers them.
4. **Ask the intent** — book a new appointment, or cancel an existing one.
   - **Book:** ask for availability, repeat every time back as an exact date to remove ambiguity, check the free slots, hold the earliest match, and confirm only after the caller agrees.
   - **Cancel:** list the caller's upcoming appointments and cancel only after an explicit confirmation.

The agent knows today's date (Europe/Madrid), so it can turn words like "tomorrow" or "next Friday" into exact dates, and it respects the clinic hours (Monday to Friday, 9:00–18:00).

**Tool design.** Each EHR operation is one `FunctionSchema` written directly for the Pipecat function-calling format, instead of pulling in a heavier agent framework (LangChain, LangGraph, etc.), to ensure compatibility with Pipecat. 

**Loop verifier.** `CallGuard` counts per-call events and returns a `stop` signal when the caller rejects too many offers in a row (set in `.env` with `MAX_REJECTED_OFFERS`), when there are too many rounds with no availability (set in `.env` with `MAX_EMPTY_AVAILABILITY_ROUNDS`), or when a global tool-call ceiling is reached (set in `.env` with `MAX_TOTAL_TOOL_CALLS`). On `stop`, the agent gives one short apology and ends the call, so it can never loop forever.


### Single-agent vs supervisor

Two standard agent architectures were tested to find the best fit for this use case.

* **Single agent** — one agent holds the full context and has every tool at once; it decides which tool to call at each step.
* **Supervisor + workers** — the supervisor is the only one that talks to the caller (wired to `bot.py`). It has no EHR tools of its own; it delegates to three specialist workers, each with its own small tool subset.

![schema](img/architectures.png)

## Latency & Evaluation

The goal here is to test the agent automatically — trying to catch hallucinations and mistakes without having to dial in by hand every time — and to balance speed against accuracy.

To do this a **patient simulator** was built: a second LLM service using a stronger model (gpt-4.1) plays the caller across 11 personas, each one a situation that can happen on a real call (new patient, existing patient, cancel, no availability, misheard date of birth, etc.). For every scenario a deterministic oracle reads the conversation traces and the final database rows and decides pass or fail (several checks per scenario: did it reach the goal, and did it do it safely). The same run also measures the agent's mean turn latency and cost [1], so accuracy, speed and cost are compared together.

Two models were tested on both architectures:

| Architecture | Model | Success | Mean turn latency | Avg LLM calls | Cost / call | Total cost |
|---|---|---|---|---|---|---|
| **single** | **gpt-4.1-mini** | **81.8%** | **2.12 s** | **11.2** | **$0.0115** | **$0.127** |
| single | gpt-5-nano | 81.8% | 3.57 s | 11.7 | $0.0023 | $0.026 |
| supervisor | gpt-4.1-mini | 81.8% | 4.08 s | 17.1 | $0.0104 | $0.114 |
| supervisor | gpt-5-nano | 72.7% | 6.97 s | 16.5 | $0.0029 | $0.031 |

**Decision: single agent with gpt-4.1-mini.** It ties for the best success rate and has the lowest latency (2.12 s per turn), which matters most on a live phone call. The supervisor adds latency and extra LLM calls without buying better accuracy, and gpt-5-nano is cheaper but clearly slower.

## Test and evaluation isolation

Every automated run — the EHR-API suite, the
voice-agent integration tests, and the evaluation harness — talks to a separate
`ehr_test` database, never the dev/production `ehr` one. It lives on the same
Postgres container but is a distinct database, created on demand and reset to a
clean, known state before each test or scenario, so a run seeds its own ground
truth without ever reading or truncating real data. A guard enforces this: the
code refuses to start against any database whose name doesn't end in `_test`
(override with `TEST_DATABASE_URL` / `EVAL_DATABASE_URL`). The test and eval APIs
also run on their own ports (`:8011`, `:8012`), so a live dev API on `:8000` is
never touched.


## Continuous integration

`.github/workflows/tests.yml` runs the full suite on every pull request targeting `main`. 


## How to use

### Prerequisites (run once, in order)

These three steps are sequential and mandatory before anything else.

**1. Clone this repository**

```
git clone <repository-url>
cd prosper-challenge
```

**2. Create a `.env` file and set your api keys.**

```
OPENAI_API_KEY=your_key
ELEVENLABS_API_KEY=your_key
DATABASE_URL=postgresql://ehr:ehr@localhost:5432/ehr

EHR_BASE_URL=http://localhost:8000

AGENT_ARCH = supervisor # single or supervisor

MAX_EMPTY_AVAILABILITY_ROUNDS = "4"  # consecutive list_availability_slots with no free slots
MAX_REJECTED_OFFERS = "4"            # consecutive held offers the caller turned down
MAX_TOTAL_TOOL_CALLS = "40"          # global circuit breaker: any runaway loop, not just slots
```

**3. Set up a virtual environment and install dependencies**

```
uv sync
```

### Running the project (each in its own terminal, in parallel)

Once setup is done, the steps below are independent — run whichever you need, each
in a separate terminal. They share one backend: Postgres (`docker compose up -d`),
which several of them depend on.

```
docker compose up -d            # start the shared Postgres backend first
```

**4. EHR API**

```
cd ehr-api
uv run uvicorn app.main:app --port 8000
```

**5. Bot** (needs the EHR API from step 4 running)

```
uv run bot.py
```

Open http://localhost:7860 in your browser and click Connect to start talking to the bot.

**6. Voice-agent tests**

```
uv run pytest tests/unit/
uv run pytest tests/integration/
```

**7. EHR-API tests**

```
uv run pytest ehr-api/tests/integration/
```

**8. Evaluating the agent**

```
uv run python -m evaluation.run_eval
``` 


## Future improvements

**Reliability — multiple providers with fallback.** In order to keep the agent
answering even when an external dependency fails — most importantly when the AI
provider is down or rate-limiting, several providers must be considered. Now there is a single provider per service
(OpenAI for the LLM, ElevenLabs for STT/TTS), so any one of them going down takes
the whole call down with it. The natural next step is to wire in a second provider
for each — for the LLM, Azure OpenAI or Anthropic (Claude); for speech, an
alternative STT/TTS vendor.

**Production observability with OpenTelemetry.** Right now traces are extracted
manually in the evaluation harness (`evaluation/harness/trace.py` writes per-call
JSONL), which is useful for offline runs but gives no visibility into live calls.
Instrumenting the pipeline with OpenTelemetry would emit spans for each turn, tool call, and LLM/STT/TTS request to a standard
backend, so latency, errors, and cost can be monitored in production instead of
only reconstructed after the fact.

**An ORM for the database layer.** Useing an ORM
such as SQLAlchemy would make the data
layer easier to evolve.

**Cloud deployment.** Setting up managed Postgres, secrets, and a public transport so the agent runs on real phone calls rather than
locally.


## Reference

[1] OpenAI API pricing (https://developers.openai.com/api/docs/pricing), accessed 26/06/2026