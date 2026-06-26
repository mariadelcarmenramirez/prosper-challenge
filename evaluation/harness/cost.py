"""Token counting and cost accounting for an evaluation run.

Cost is computed from the **billed token usage** the OpenAI API returns on every
response (``usage.prompt_tokens`` / ``usage.completion_tokens``). That is the
authoritative count: for the GPT-5 reasoning models it already folds in the hidden
reasoning tokens, which a local tokenizer cannot see. We still expose a tiktoken
helper (``count_tokens``) for an independent, offline estimate of plain text, but
the dollar figures in the report are always usage-based so they match the invoice.

Prices are USD per 1M tokens (input / output), provided by the project. Unknown
models fall back to zero so a typo never crashes a run — it just shows $0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

# USD per 1,000,000 tokens: (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
}


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one call's usage, from the per-1M pricing table."""
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000


def count_tokens(text: str, model: str = "gpt-4o") -> int:
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
