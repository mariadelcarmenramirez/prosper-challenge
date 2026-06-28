from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from .trace import ConversationTrace

# USD per 1,000,000 tokens: (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-nano": (0.05, 0.40),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one call's usage, from the per-1M pricing table."""
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def count_tokens(text: str, model: str = "gpt-4.1-mini") -> int:
    """Offline tiktoken estimate for plain text (cross-check only, not for billing).

    Reasoning-token usage is invisible here; use the API ``usage`` for real cost.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


@dataclass
class Ledger:
    """Accumulates token usage, cost and call counts for one conversation.

    Calls are tagged by ``role`` so the agent's own spend (the thing we compare
    across architectures and models) is kept apart from the simulated caller's.
    """

    # Agent-side totals (the supervisor's nested worker calls land here too).
    agent_calls: int = 0
    agent_prompt_tokens: int = 0
    agent_completion_tokens: int = 0
    agent_cost_usd: float = 0.0
    # Caller-side totals, tracked but excluded from the agent comparison.
    caller_calls: int = 0
    caller_cost_usd: float = 0.0
    # Per-model agent spend, so a run can attribute cost when a worker model differs.
    per_model_cost: dict[str, float] = field(default_factory=dict)

    def record(self, role: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        cost = cost_usd(model, prompt_tokens, completion_tokens)
        if role == "caller":
            self.caller_calls += 1
            self.caller_cost_usd += cost
            return
        # Everything that is not the caller counts as agent spend (agent, worker).
        self.agent_calls += 1
        self.agent_prompt_tokens += prompt_tokens
        self.agent_completion_tokens += completion_tokens
        self.agent_cost_usd += cost
        self.per_model_cost[model] = self.per_model_cost.get(model, 0.0) + cost


# GPT-5 reasoning models run at "low" effort by default in the eval: a voice agent
# wants the lowest honest latency, and it keeps the comparison against the classic
# models from being dominated by reasoning time. Recorded on every trace.
GPT5_REASONING_EFFORT = "low"


def is_reasoning_model(model: str) -> bool:
    return model.startswith("gpt-5")


def shape_request(model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize request kwargs for the model's family (in place, returned too)."""
    if is_reasoning_model(model):
        kwargs.pop("temperature", None)  # reasoning models reject an explicit temperature
        kwargs.setdefault("reasoning_effort", GPT5_REASONING_EFFORT)
    else:
        kwargs.setdefault("temperature", 0)  # classic models: pin for determinism
    return kwargs


class _Completions:
    def __init__(self, outer: "InstrumentedClient") -> None:
        self._outer = outer

    async def create(self, **kwargs: Any) -> Any:
        return await self._outer._create(**kwargs)


class _Chat:
    def __init__(self, outer: "InstrumentedClient") -> None:
        self.completions = _Completions(outer)


class InstrumentedClient:
    """Wraps a raw ``AsyncOpenAI``; meters every chat-completions call."""

    def __init__(
        self,
        raw: Any,
        ledger: Ledger,
        role: str,
        trace: ConversationTrace | None = None,
    ) -> None:
        self._raw = raw
        self.ledger = ledger
        self.role = role  # "agent" (incl. workers) or "caller"
        self.trace = trace
        self.chat = _Chat(self)

    async def _create(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        shape_request(model, kwargs)
        t0 = time.perf_counter()
        resp = await self._raw.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.ledger.record(self.role, model, prompt_tokens, completion_tokens)
        if self.trace is not None:
            self.trace.add(
                "llm_call",
                role=self.role,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=round(dt, 4),
            )
        return resp
