"""LLM-as-judge — deferred placeholder (the hard oracle is the current accuracy signal).

The plan is for a fixed, strong model to grade each transcript on a qualitative
rubric the deterministic oracle can't easily capture: did the agent confirm before
mutating, read details back, avoid reading ids/JSON aloud, speak dates in words,
and end politely on a stop signal? That score would be a *secondary* metric
alongside the oracle's pass/fail.

It is intentionally not wired in yet (we chose to ship the hard oracle first). The
trace already carries everything a judge needs — ``agent_turn`` / ``caller_turn``
events form the transcript — so adding this later is self-contained: implement
``score_transcript`` and merge its result into the trace before metrics aggregation.
"""

from __future__ import annotations

from evaluation.harness.trace import ConversationTrace

RUBRIC = [
    "confirmed exact date/time before any create/confirm/cancel",
    "read identity details back before lookup",
    "never read out ids, UUIDs, or JSON",
    "spoke dates and times in words, not ISO strings",
    "on a stop signal, apologised once and ended without more tool calls",
]


async def score_transcript(trace: ConversationTrace, client, model: str) -> dict:
    """Not implemented yet — see module docstring. Returns an empty score."""
    raise NotImplementedError("LLM judge is deferred; the oracle is the accuracy metric for now.")
