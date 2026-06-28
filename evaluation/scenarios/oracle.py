from __future__ import annotations

from dataclasses import dataclass, field

from evaluation.harness.trace import ConversationTrace


@dataclass
class Check:
    name: str
    ok: bool
    hard: bool = True
    detail: str = ""


@dataclass
class OracleResult:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, hard: bool = True, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=ok, hard=hard, detail=detail))

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks if c.hard)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "ok": c.ok, "hard": c.hard, "detail": c.detail}
                for c in self.checks
            ],
        }


# --- trace inspection helpers ----------------------------------------------


def tool_events(trace: ConversationTrace) -> list[dict]:
    return [e.data for e in trace.events if e.type == "tool_call"]


def tool_names(trace: ConversationTrace) -> list[str]:
    return [d.get("name", "") for d in tool_events(trace)]


def count_tool(trace: ConversationTrace, name: str) -> int:
    return tool_names(trace).count(name)


def results_of(trace: ConversationTrace, name: str) -> list:
    return [d.get("result") for d in tool_events(trace) if d.get("name") == name]


def called_before(trace: ConversationTrace, first: str, second: str) -> bool:
    """True if ``first`` appears before ``second`` at least once (and both appear)."""
    names = tool_names(trace)
    if first not in names or second not in names:
        return False
    return names.index(first) < names.index(second)


def agent_texts(trace: ConversationTrace) -> list[str]:
    return [e.data.get("text", "") for e in trace.events if e.type == "agent_turn"]
