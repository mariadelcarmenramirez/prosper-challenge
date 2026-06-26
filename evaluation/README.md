# Evaluation pipeline

Automatically simulates phone calls against the scheduling agent — no dialling in
by hand — to measure **accuracy**, **latency** and **cost** across the three agent
architectures (`single`, `specialist`, `supervisor`) and several OpenAI models, in
a static, controlled environment.

It works because the agent's brain is decoupled from audio: the eval drives the
*real* prompts, tools, `CallGuard`, specialist phase-swaps and supervisor worker
loops through a tiny in-memory Pipecat shim, with a second LLM playing the patient.

## What it measures

| Axis | How |
|---|---|
| **Accuracy** | A deterministic **oracle** per scenario checks the final database rows *and* the recorded tool trace (e.g. *was the exact slot booked? was nothing cancelled when there was nothing to cancel? was the right namesake found by phone?*). Pass/fail → success rate. |
| **Latency** | Wall-clock per **agent turn** (caller-turn-in → next spoken-turn-ready), folding in the whole tool loop — including the supervisor's nested worker calls. Reported as mean / p50 / p90 / p95. |
| **Cost** | Billed token usage (`usage` from every API response, reasoning tokens included) priced with the per-model table in [`harness/cost.py`](harness/cost.py). Reported as **average $/call** and total spend per model. |

The simulated caller is pinned to one fixed model so it never confounds the
comparison; its tokens/latency are tracked separately and excluded.

## How it's wired

```
run_eval.py ── launches EHR API on ehr_test ──▶ FastAPI ──▶ Postgres (ehr_test)
     │                                                          ▲
     │  per (arch × model × scenario):                          │ seed + inspect
     ▼                                                          │ (asyncpg, direct)
ConversationRunner ──drives──▶ agent (shim: real handlers/guard/phase/workers)
     │                          ▲                                │
     └──── SimulatedCaller ─────┘   every LLM call ── InstrumentedClient ──▶ OpenAI
                                     (shapes gpt-5 vs gpt-4.1 requests, meters cost)
```

- `harness/shim.py` — in-memory `FakeLLM` / `FakeLLMContext` / `FakeFunctionCallParams`.
- `harness/adapters.py` — wraps each architecture uniformly (and points the
  supervisor's workers at the model under test).
- `harness/client.py` — one place that shapes per-family requests (gpt-5 reasoning
  vs. classic) and records usage + latency.
- `harness/runner.py` — the agent-turn / caller-turn loop + end detection.
- `harness/caller.py` — the LLM patient.
- `scenarios/` — the 11 edge cases, their DB setup, and their oracles.
- `harness/metrics.py` — writes the CSVs.
- `harness/judge.py` — deferred LLM-as-judge stub (oracle is the current metric).

## Running it

Requires Postgres up. The runner launches its own EHR API against the **dedicated
`ehr_test` database** (never dev data) and tears it down afterwards.

```bash
docker compose up -d                                   # Postgres

uv run python -m evaluation.run_eval                   # full matrix (3 arch × 3 models × 11)
uv run python -m evaluation.run_eval --arch single --model gpt-4.1-mini   # one slice
uv run python -m evaluation.run_eval --scenario cancel_nonexistent        # one scenario
uv run python -m evaluation.run_eval --no-launch       # reuse an API already at EHR_BASE_URL
```

Useful env / flags:
- `--model` / `--arch` / `--scenario` — restrict any axis (cheap slices).
- `--caller-model` — the fixed model playing the patient (default `gpt-4.1-mini`).
- `EVAL_DATABASE_URL` — the `*_test` DB (default `…/ehr_test`).
- `EVAL_API_PORT` — port for the eval's EHR API (default `8011`, avoids the dev `8000`).
- Models in the matrix are set by `DEFAULT_MODELS` in `run_eval.py`
  (`gpt-4.1`, `gpt-4.1-mini`, `gpt-5-nano`; add `gpt-5-mini` there to widen it).

> Each conversation makes real OpenAI calls; the supervisor multiplies them via its
> nested workers. Run slices while iterating, the full matrix for the final numbers.

## Outputs

- `evaluation/results/traces_<stamp>.jsonl` — one full trace per conversation
  (every caller/agent turn, tool call + args + result + latency, end reason). The
  single source for the oracle, the latency numbers, and debugging.
- `evaluation/metrics/runs_<stamp>.csv` — one row per conversation.
- `evaluation/metrics/by_scenario_<stamp>.csv` — pass/fail per cell × scenario.
- `evaluation/metrics/summary_<stamp>.csv` — the headline table:
  per (arch, model) success rate, latency distribution, LLM-call count, tokens,
  **average $/call** and total cost.

## Scenarios

`book_new_patient`, `book_existing_patient`, `cancel_existing`,
`cancel_nonexistent`, `same_name_diff_phone`, `empty_availability_loop`,
`reject_offers_loop`, `invalid_then_valid_time`, `slot_already_taken`,
`confirm_before_cancel`, `misheard_dob` — see
[`scenarios/scenarios.py`](scenarios/scenarios.py) for each one's setup, caller
persona, and oracle checks.
