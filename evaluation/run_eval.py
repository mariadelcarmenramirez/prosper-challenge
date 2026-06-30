"""Evaluation matrix runner.

Drives every (architecture x model x scenario) cell as a text-only simulated call
against a live EHR on a dedicated ``*_test`` database, then writes the trace JSONL
and the CSV comparison tables. The default matrix is 2 architectures x N models x
11 scenarios (no repetitions — the scenario spread is the variability), and every
axis is filterable so you can run a slice cheaply:

    uv run python -m evaluation.run_eval                       # full matrix
    uv run python -m evaluation.run_eval --arch single --model gpt-4.1-mini
    uv run python -m evaluation.run_eval --scenario cancel_nonexistent
    uv run python -m evaluation.run_eval --no-launch           # reuse a running API

Requires Postgres up (``docker compose up -d``). By default it launches its own EHR
API process pointed at the test database; ``--no-launch`` reuses whatever is at
``EHR_BASE_URL``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# --- environment must be set BEFORE importing the agent code ----------------
# tool_implementations reads EHR_BASE_URL at import time, and the call guard reads
# MAX_* at import time, so load .env and point the agent at the test API first.
load_dotenv(override=True)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ["gpt-4.1-mini", "gpt-5-nano"]
EVAL_API_PORT = int(os.environ.get("EVAL_API_PORT", "8011"))
EVAL_DATABASE_URL = os.environ.get(
    "EVAL_DATABASE_URL", "postgresql://ehr:ehr@localhost:5432/ehr_test"
)
os.environ["EHR_BASE_URL"] = os.environ.get("EVAL_EHR_BASE_URL", f"http://localhost:{EVAL_API_PORT}")

from dataclasses import asdict  

import asyncpg  
from openai import AsyncOpenAI  

from evaluation.harness.adapters import ARCHITECTURES, build_agent  
from evaluation.harness.patient_simulator import PatientCaller  
from evaluation.harness.client import InstrumentedClient, Ledger
from evaluation.harness.metrics import write_metrics  
from evaluation.harness.runner import ConversationRunner  
from evaluation.harness.trace import ConversationTrace, write_traces  
from evaluation.scenarios import db  
from evaluation.scenarios.scenarios import SCENARIOS  

# --- EHR API lifecycle ------------------------------------------------------


def _api_healthy() -> bool:
    try:
        return httpx.get(f"{os.environ['EHR_BASE_URL']}/health", timeout=2.0).status_code == 200
    except httpx.HTTPError:
        return False


def launch_api() -> subprocess.Popen:
    """Start the EHR API against the test DB and wait until it is healthy."""
    env = {**os.environ, "DATABASE_URL": EVAL_DATABASE_URL}
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--port", str(EVAL_API_PORT)],
        cwd=ROOT / "ehr-api",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if _api_healthy():
            return proc
        if proc.poll() is not None:
            raise RuntimeError("EHR API process exited before becoming healthy.")
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("EHR API did not become healthy in time.")


# --- one conversation -------------------------------------------------------


async def run_one(scenario, arch: str, model: str, caller_model: str, raw: AsyncOpenAI,
                  conn: asyncpg.Connection) -> ConversationTrace:
    trace = ConversationTrace(scenario_id=scenario.id, arch=arch, model=model)
    ledger = Ledger()
    agent_client = InstrumentedClient(raw, ledger, role="agent", trace=trace)
    caller_client = InstrumentedClient(raw, ledger, role="caller", trace=trace)

    # Fresh, known starting state for this scenario.
    await db.reset(conn)
    ctx = await scenario.setup(conn)

    setup = build_agent(arch, model, now=None, agent_client=agent_client)
    caller = PatientCaller(caller_client, caller_model, scenario.persona(ctx))
    runner = ConversationRunner(setup, agent_client, caller, trace)
    await runner.run()

    # Oracle reads the final DB rows + the recorded trace.
    try:
        result = await scenario.oracle(conn, trace, ctx)
        trace.oracle = result.to_dict()
    except Exception as exc:  # a broken oracle shouldn't lose the conversation data
        trace.oracle = {"passed": False, "checks": [], "error": f"{type(exc).__name__}: {exc}"}
    trace.ledger = asdict(ledger)
    return trace


# --- matrix -----------------------------------------------------------------


async def main_async(args) -> None:
    arches = [args.arch] if args.arch else list(ARCHITECTURES)
    models = [args.model] if args.model else DEFAULT_MODELS
    scenarios = [s for s in SCENARIOS if not args.scenario or s.id == args.scenario]
    if not scenarios:
        raise SystemExit(f"No scenario matches {args.scenario!r}.")

    raw = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    conn = await asyncpg.connect(EVAL_DATABASE_URL)
    traces: list[ConversationTrace] = []
    total = len(arches) * len(models) * len(scenarios)
    i = 0
    try:
        for arch in arches:
            for model in models:
                for scenario in scenarios:
                    i += 1
                    print(f"[{i}/{total}] {arch:11} {model:13} {scenario.id}", flush=True)
                    trace = await run_one(scenario, arch, model, args.caller_model, raw, conn)
                    status = "PASS" if trace.oracle.get("passed") else "FAIL"
                    print(f"        -> {status}  end={trace.end_reason}  "
                          f"turns={len(trace.agent_turn_latencies)}  "
                          f"cost=${trace.ledger.get('agent_cost_usd', 0):.4f}"
                          f"{'  ERROR: ' + trace.error if trace.error else ''}", flush=True)
                    traces.append(trace)
    finally:
        await conn.close()
        await raw.close()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "evaluation"
    write_traces(traces, out_dir / "results" / f"traces_{stamp}.jsonl")
    report = write_metrics(traces, out_dir / "metrics", stamp=stamp)
    _print_summary(report["summary"])
    print(f"\nTraces:  evaluation/results/traces_{stamp}.jsonl")
    print(f"Metrics: evaluation/metrics/summary_{stamp}.csv")


def _print_summary(summary: list[dict]) -> None:
    print("\n=== SUMMARY (success rate / mean latency / avg cost) ===")
    print(f"{'arch':11} {'model':13} {'succ':>6} {'mean_s':>8} {'calls':>6} {'$/call':>9}")
    for row in summary:
        print(f"{row['arch']:11} {row['model']:13} {row['success_rate']*100:5.0f}% "
              f"{row['mean_turn_latency_s']:8.2f} "
              f"{row['avg_agent_llm_calls']:6.1f} {row['avg_cost_per_call_usd']:9.5f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the voice-agent evaluation matrix.")
    parser.add_argument("--arch", choices=ARCHITECTURES, help="Limit to one architecture.")
    parser.add_argument("--model", help="Limit to one agent model.")
    parser.add_argument("--scenario", help="Limit to one scenario id.")
    parser.add_argument("--caller-model", default="gpt-4.1",
                        help="Fixed model that plays the simulated caller (default: gpt-4.1).")
    parser.add_argument("--no-launch", action="store_true",
                        help="Reuse an EHR API already running at EHR_BASE_URL instead of launching one.")
    args = parser.parse_args()

    import asyncio

    proc = None
    try:
        asyncio.run(_ensure_db())
        if args.no_launch:
            if not _api_healthy():
                raise SystemExit(f"No healthy EHR API at {os.environ['EHR_BASE_URL']} (--no-launch).")
        else:
            print(f"Launching EHR API on :{EVAL_API_PORT} against the test database...", flush=True)
            proc = launch_api()
        asyncio.run(main_async(args))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


async def _ensure_db() -> None:
    await db.ensure_database(EVAL_DATABASE_URL)


if __name__ == "__main__":
    sys.exit(main())
