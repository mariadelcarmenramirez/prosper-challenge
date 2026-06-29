## Expectations & Deliverables

To make the agent functional we expect you to implement at least the following:

1. **EHR HTTP API**: Build a simple EHR service (any framework you like) that exposes at least these endpoints:

   - `create_patient` — register a new patient (e.g. name, date of birth, contact info)
   - `find_patient` — look up an existing patient by name and date of birth
   - `list_availability_slots` — return the clinic's available appointment slots for a given date or range
   - `create_appointment` — book a slot for a given patient
   - `cancel_appointment` — cancel an existing appointment

   The EHR should persist its data in a database so patients and appointments survive across restarts — please don't keep state in memory only. The shape of the request/response is up to you — design it the way you'd want a real integration to look.

2. **Conversation Flow**: Modify the agent's behavior so that it can:

   - Identify whether the caller is a new or existing patient
   - Register them in the EHR if they're new
   - Schedule a new appointment or cancel an existing one

3. **Integration**: Wire the voice agent to your EHR's HTTP API so it can actually create patients, look them up, and create or cancel appointments during conversations.

We encourage you to use AI tools (Claude Code, Cursor, etc.) to help you with this challenge. We don't mind if you "vibe code" everything, that probably means you have good prompting skills. What we do care about is whether you understand the decisions and trade-offs behind your solution. That's why, apart from the code itself, we'd like you to write a high-level overview of your solution and the decisions you've made to get to it—do this in a `SOLUTION.md` file at the root of your fork. During the interview we'll dive deeper into it and discuss opportunities to improve it in the future.

If you'd like to go further, you can already document some of those potential improvements in your `SOLUTION.md`. Some areas that we'd love to hear your thoughts on are:
- Latency: balancing speed with user experience and accuracy
- Reliability: ensuring that the agent is always available to answer, regardless of external factors (e.g. AI provider unavailable)
- Evaluation: brainstorming or even prototyping a method to check that the agent is behaving how it is supposed to. We're particularly interested in ways to automatically test or simulate calls so that hallucinations and agent mistakes can be caught without having to dial in by hand every time.


Once you are done, please share the link to your fork so that we can get familiar with it before our chat.





1. Overview          – 3-4 sentences: what the system does + an architecture
                       sketch (voice agent ⇄ EHR API ⇄ Postgres). A tiny ASCII
                       diagram here earns a lot of goodwill.
2. Key design decisions & trade-offs   ← the heart of it
     - EHR API shape / data model
     - Conversation flow & tool design
     - Single-agent vs supervisor (you have both in tests — talk about it)


3. Latency / Reliability / Evaluation  ← the 3 areas they explicitly call out

[1]



## How to use

1. Clone this repository

```bash
git clone <repository-url>
cd prosper-challenge
```

2. Set your api keys:

Create a `.env` file:

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

3. Set up a virtual environment and install dependencies

```bash
uv sync
```

### Set EHR API

```
docker compose up -d            # Postgres must be running first
cd ehr-api
uv run uvicorn app.main:app --port 8000
```

### Running the Bot

```bash
uv run bot.py
```

### Running the voice-agent test (test unit and integration)

```
uv run pytest tests/unit/

uv run pytest tests/integration/
```
### running the ehr-api (ehr-api/tests)

```
cd ehr-api

uv run pytest tests/integration/
```

### evaluating agent 

```
uv run python -m evaluation.run_eval
``` 

5. Future improvements


## Reference

[1] OpenAI API pricing (https://developers.openai.com/api/docs/pricing), accessed 26/06/2026