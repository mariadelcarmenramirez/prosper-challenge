"""Structured per-conversation traces — the single artifact every metric reads.

One ``ConversationTrace`` is the ordered story of a single simulated call: each
caller turn, each agent spoken turn, every tool call (name, args, result, latency)
including the supervisor's nested worker calls, phase transitions, and the reason
the call ended. From this one record we derive:

* **accuracy** — the oracle replays the tool sequence and inspects the outcome;
* **latency** — per-agent-turn wall times are recorded here;
* **cost** — the attached :class:`~evaluation.harness.cost.Ledger` snapshot;
* **debugging** — a human can read exactly what happened on a failed run.

Traces are written as JSON Lines (one conversation per line) under
``evaluation/results/`` so a whole matrix is greppable and re-analysable offline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """One thing that happened during the call, in order."""

    type: str  # caller_turn | agent_turn | tool_call | phase | stop | end
    t: float  # seconds since the conversation started
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTrace:
    """The full record of one simulated call for one (arch, model, scenario)."""

    scenario_id: str
    arch: str
    model: str
    events: list[Event] = field(default_factory=list)
    agent_turn_latencies: list[float] = field(default_factory=list)
    caller_turn_latencies: list[float] = field(default_factory=list)
    end_reason: str = ""
    # Filled in after the call by the oracle / cost ledger.
    oracle: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    _start: float = field(default=0.0, repr=False)

    def start(self) -> None:
        self._start = time.perf_counter()

    def _now(self) -> float:
        return time.perf_counter() - self._start

    def add(self, type: str, **data: Any) -> None:
        self.events.append(Event(type=type, t=round(self._now(), 4), data=data))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_start", None)
        return d


def write_traces(traces: list[ConversationTrace], path: Path) -> None:
    """Append all traces to a JSONL file (one conversation per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tr in traces:
            fh.write(json.dumps(tr.to_dict(), default=str) + "\n")
