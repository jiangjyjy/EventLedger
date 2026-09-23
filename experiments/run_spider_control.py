from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from carve.schemas import Trace


def _hidden_verifier_event(trace: Trace):
    return next(
        (event for event in reversed(trace.events) if event.agent_role == "hidden_sql_verifier"),
        None,
    )


def _telemetry(trace: Trace) -> dict[str, float | int]:
    return {
        "tokens": trace.total_tokens,
        "cost_usd": trace.total_cost_usd,
        "latency_ms": sum(event.latency_ms for event in trace.events),
        "tool_calls": sum(event.type == "tool" for event in trace.events),
        "api_calls": int(trace.manifest.get("telemetry", {}).get("api_calls", 0)),
    }


def evaluate_spider_traces(traces: list[Trace]) -> tuple[list[Trace], dict[str, Any]]:
    controlled: list[Trace] = []
    records: list[dict[str, Any]] = []
    for trace in traces:
        if trace.dataset != "spider":
            raise ValueError(f"Spider control received non-Spider dataset: {trace.dataset}")
        hidden = _hidden_verifier_event(trace)
        score = float(hidden.metadata.get("verifier_score", 0.0)) if hidden else 0.0
        success = bool(hidden and hidden.metadata.get("verifier_success", False))
        controlled_trace = trace.clone_with_events(
            list(trace.events),
            final_answer=trace.final_answer,
            verifier_score=score,
            oracle_score=None,
            success=success,
            manifest={
                **trace.manifest,
                "control": {
                    "mode": "offline_dependency_safe_no_pruning",
                    "removed_events": 0,
                    "verifier_source": "hidden_sql_verifier_event_metadata",
                },
            },
        )
        controlled.append(controlled_trace)
        records.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "success": success,
                "verifier_score": score,
                "kept_events": len(trace.events),
                "removed_events": 0,
                **_telemetry(trace),
            }
        )
    count = len(records)
    summary = {
        "control_mode": "offline_dependency_safe_no_pruning",
        "controlled_traces": count,
        "controlled_success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "controlled_mean_tokens": mean(row["tokens"] for row in records) if records else 0.0,
        "controlled_mean_api_calls": mean(row["api_calls"] for row in records) if records else 0.0,
        "mean_removed_events": 0.0,
        "removed_events_total": 0,
        "records": records,
    }
    return controlled, summary


def _read_traces(path: Path) -> list[Trace]:
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Spider-only dependency-safe control evaluator")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    traces = _read_traces(args.input_traces)
    if args.limit is not None:
        traces = traces[: args.limit]
    controlled, summary = evaluate_spider_traces(traces)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "controlled_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in controlled:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
