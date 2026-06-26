# Prosper Challenge

This is a template repository for an AI voice agent that schedules appointments for a health clinic. In a real deployment the agent would talk to the clinic's CRM, which in healthcare is known as an Electronic Health Record (EHR). For this challenge we'd like you to build a small EHR yourself and wire the voice agent to it.

The foundations are already set: Pipecat is configured with sensible defaults and the bot already introduces itself when initialized.

You'll be responsible for:

- Building a simple EHR exposed as an HTTP API
- Expanding the agent's configuration so that it can identify patients, schedule appointments, and cancel them
- Connecting the agent to your EHR so it can actually act during a conversation

## Setup

To get started, fork this repository so that you can start commiting and pushing changes to your own copy.

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager installed

### Installation

1. Clone this repository

   ```bash
   git clone <repository-url>
   cd prosper-challenge
   ```

2. Copy the API keys we've shared with you:

   Create a `.env` file:

   ```bash
   cp env.example .env
   ```

   Then, add your API keys:

   ```ini
   ELEVENLABS_API_KEY=your_elevenlabs_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Set up a virtual environment and install dependencies

   ```bash
   uv sync
   ```

### Running the Bot

```bash
uv run bot.py
```

**Open http://localhost:7860 in your browser** and click `Connect` to start talking to your bot.

> 💡 First run note: The initial startup may take ~20 seconds as Pipecat downloads required models and imports.



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
