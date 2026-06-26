"""An instrumented OpenAI client: the one place every LLM call is shaped + metered.

It is a drop-in for ``AsyncOpenAI`` (exposes ``.chat.completions.create``) so it
works both for the runner's top-level agent/caller calls *and* when injected into
the supervisor as ``_client_obj`` for its nested worker loop. Routing every call
through here buys three things in one spot:

1. **Per-family request shaping.** The matrix mixes classic chat models
   (gpt-4.1*) with reasoning models (gpt-5*). Reasoning models reject
   ``temperature`` and take a ``reasoning_effort`` knob instead. We normalize that
   here so the architectures never have to care which family they're talking to.
2. **Billed usage capture.** Every response's ``usage`` is folded into the
   :class:`~evaluation.harness.cost.Ledger` (reasoning tokens included), tagged by
   role so agent spend stays separate from the simulated caller's.
3. **Per-call latency** is recorded on the trace, which is what lets us see *where*
   the supervisor spends its time (top-level vs. each worker call).
"""

from __future__ import annotations

import time
from typing import Any

from .cost import Ledger
from .trace import ConversationTrace

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
