"""Aggregate finished traces into the comparison tables (CSV under evaluation/metrics/).

Three CSVs come out of a run:

* ``runs_*.csv`` — one row per conversation (the raw detail behind everything).
* ``by_scenario_*.csv`` — pass/fail per (arch, model, scenario), to see *where* an
  architecture or model breaks.
* ``summary_*.csv`` — the headline table: per (arch, model), the accuracy
  (oracle success rate), the mean agent-turn latency, LLM-call count, token usage,
  and the **average cost per conversation in USD** plus the run's total spend — the
  accuracy-vs-latency-vs-cost picture in one place.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .trace import ConversationTrace


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


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


def _by_scenario_rows(run_rows: list[dict]) -> list[dict]:
    return sorted(run_rows, key=lambda r: (r["arch"], r["model"], r["scenario"]))


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
    """Write the three CSVs and return the summary rows (for console printing)."""
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_rows = _run_rows(traces)
    summary = _summary_rows(run_rows, traces)
    by_scenario = _by_scenario_rows(run_rows)

    _write_csv(out_dir / f"runs_{stamp}.csv", run_rows)
    _write_csv(out_dir / f"by_scenario_{stamp}.csv",
               [{k: r[k] for k in ("arch", "model", "scenario", "passed", "end_reason", "error")}
                for r in by_scenario])
    _write_csv(out_dir / f"summary_{stamp}.csv", summary)
    return {"stamp": stamp, "summary": summary}
