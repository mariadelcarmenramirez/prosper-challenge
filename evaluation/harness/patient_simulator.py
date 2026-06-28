from __future__ import annotations

from dataclasses import dataclass

from .client import InstrumentedClient

HANGUP = "[[HANGUP]]"

PATIENT_PROTOCOL = f"""
You are role-playing a PATIENT phoning a medical clinic's scheduling line. You are
NOT the assistant — never schedule, look things up, or speak as the clinic.

How to behave:
- Speak naturally and briefly, one short phone turn at a time (under ~30 words).
- Only share details when the assistant asks for them; then give what your card says.
- Stay in character and follow your goal and behaviour notes exactly, even if the
  assistant is slow or repeats itself.
- If the assistant misunderstands, react like a real caller would (correct them once).
- When the call is finished — your goal is done, the assistant has ended the call,
  or it is clearly impossible — say a brief natural goodbye and then append {HANGUP}
  on the same message. Only then. Do not hang up early.
"""


@dataclass
class PatientTurn:
    text: str
    hang_up: bool


class PatientCaller:
    def __init__(self, client: InstrumentedClient, model: str, persona: str) -> None:
        self._client = client
        self._model = model
        self._messages: list[dict] = [
            {"role": "system", "content": persona.strip() + "\n" + PATIENT_PROTOCOL}
        ]

    async def respond(self, agent_text: str) -> PatientTurn:
        # From the patient's point of view, the agent's speech is the incoming "user" turn.
        self._messages.append({"role": "user", "content": agent_text})
        resp = await self._client.chat.completions.create(
            model=self._model, messages=self._messages
        )
        raw = resp.choices[0].message.content or ""
        self._messages.append({"role": "assistant", "content": raw})
        hang_up = HANGUP in raw
        text = raw.replace(HANGUP, "").strip()
        return PatientTurn(text=text, hang_up=hang_up)
