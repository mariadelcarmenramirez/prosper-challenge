from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .trace import ConversationTrace


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _failed_checks(oracle: dict) -> str:
    """';'-joined names of the checks that caused this run to fail (passed==0).

    ``passed`` is ``all(checks ok)``, so every failing check is a cause. Passing
    runs have no failing checks and get an empty string.
    """
    names = [
        c.get("name", "")
        for c in (oracle or {}).get("checks", [])
        if not c.get("ok")
    ]
    return ";".join(names)


def _run_rows(traces: list[ConversationTrace]) -> list[dict]:
    rows = []
    for tr in traces:
        led = tr.ledger or {}
        lat = tr.agent_turn_latencies
        rows.append(
            {
                "arch": tr.arch,
                "model": tr.model,
                "scenario": tr.scenario_id,
                "passed": int(bool(tr.oracle.get("passed"))),
                "end_reason": tr.end_reason,
                "error": tr.error or "",
                "failed_checks": _failed_checks(tr.oracle),
                "agent_turns": len(lat),
                "mean_turn_latency_s": _mean(lat),
                "agent_llm_calls": led.get("agent_calls", 0),
                "prompt_tokens": led.get("agent_prompt_tokens", 0),
                "completion_tokens": led.get("agent_completion_tokens", 0),
                "agent_cost_usd": round(led.get("agent_cost_usd", 0.0), 6),
                "caller_cost_usd": round(led.get("caller_cost_usd", 0.0), 6),
            }
        )
    return rows


def _summary_rows(run_rows: list[dict], traces: list[ConversationTrace]) -> list[dict]:
    # Gather per-(arch, model) for averages, and all turn latencies for the mean.
    by_cell_runs: dict[tuple, list[dict]] = defaultdict(list)
    by_cell_lat: dict[tuple, list[float]] = defaultdict(list)
    for row in run_rows:
        by_cell_runs[(row["arch"], row["model"])].append(row)
    for tr in traces:
        by_cell_lat[(tr.arch, tr.model)].extend(tr.agent_turn_latencies)

    summary = []
    for cell, rows in sorted(by_cell_runs.items()):
        arch, model = cell
        n = len(rows)
        lats = by_cell_lat[cell]
        agent_costs = [r["agent_cost_usd"] for r in rows]
        summary.append(
            {
                "arch": arch,
                "model": model,
                "conversations": n,
                "success_rate": round(sum(r["passed"] for r in rows) / n, 4) if n else 0.0,
                "mean_turn_latency_s": _mean(lats),
                "avg_agent_llm_calls": _mean([float(r["agent_llm_calls"]) for r in rows]),
                "avg_prompt_tokens": round(_mean([float(r["prompt_tokens"]) for r in rows])),
                "avg_completion_tokens": round(_mean([float(r["completion_tokens"]) for r in rows])),
                "avg_cost_per_call_usd": round(sum(agent_costs) / n, 6) if n else 0.0,
                "total_cost_usd": round(sum(agent_costs), 6),
                "errors": sum(1 for r in rows if r["error"]),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(traces: list[ConversationTrace], out_dir: Path, stamp: str | None = None) -> dict:
    """Write the two CSVs and return the summary rows (for console printing)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows = _run_rows(traces)
    summary = _summary_rows(run_rows, traces)

    _write_csv(out_dir / f"runs_{stamp}.csv", run_rows)
    _write_csv(out_dir / f"summary_{stamp}.csv", summary)
    return {"stamp": stamp, "summary": summary}
