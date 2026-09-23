"""Zero-provider-API MBPP RQ3 baseline controls.

These are explicitly local reimplementations for comparison, not claims of
official RULER or MASPRM implementations. They operate on saved factual traces
and never read CARVE credit/reward labels.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean
from typing import Any, Optional


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _events(trace: Any) -> list[Any]:
    return list(_get(trace, "events", []) or [])


def _event_metadata(event: Any) -> dict[str, Any]:
    return dict(_get(event, "metadata", {}) or {})


def _telemetry(trace: Any) -> dict[str, float]:
    api = input_tokens = output_tokens = latency = 0.0
    tools = 0
    for event in _events(trace):
        metadata = _event_metadata(event)
        usage = metadata.get("telemetry", {}) or {}
        api += float(usage.get("api_calls", 0) or 0)
        input_tokens += float(usage.get("input_tokens", 0) or 0)
        output_tokens += float(usage.get("output_tokens", 0) or 0)
        latency += float(usage.get("wall_clock_latency_ms", _get(event, "latency_ms", 0)) or 0)
        tools += int(_get(event, "type") == "tool")
    return {
        "api_calls": api,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens": input_tokens + output_tokens,
        "tool_calls": float(tools),
        "latency_ms": latency,
    }


def outcome_reward(trace: Any) -> float:
    """Pure outcome reward: no event-level CARVE signal is consulted."""
    score = _get(trace, "verifier_score")
    if score is not None:
        return max(0.0, min(1.0, float(score)))
    return 1.0 if bool(_get(trace, "success", False)) else 0.0


def _event_evidence(event: Any) -> float:
    metadata = _event_metadata(event)
    if metadata.get("verifier_success") is True:
        return 1.0
    if metadata.get("verifier_score") is not None:
        return max(0.0, min(1.0, float(metadata["verifier_score"])))
    event_type = str(_get(event, "type", ""))
    content = str(_get(event, "content", "")).lower()
    if event_type == "tool":
        return 0.6
    if any(token in content for token in ("candidate", "selected_candidate", "assert", "passed_verifier")):
        return 0.35
    return 0.0


def ruler_trajectory_score(trace: Any) -> float:
    """Deterministic local trajectory-judge score.

    The terminal verifier outcome dominates; evidence density and normalized
    cost provide deterministic tie-breaking for trajectory ranking.
    """
    events = _events(trace)
    evidence_values = [_event_evidence(event) for event in events]
    evidence = mean(evidence_values) if evidence_values else 0.0
    telemetry = _telemetry(trace)
    cost_factor = 1.0 / (1.0 + telemetry["tokens"] / 10000.0)
    return max(0.0, min(1.0, 0.70 * outcome_reward(trace) + 0.20 * evidence + 0.10 * cost_factor))


def masprm_event_scores(trace: Any) -> list[float]:
    """Independent local process-reward scores, one for each factual event."""
    events = _events(trace)
    scores: list[float] = []
    running = 0.0
    for event in events:
        evidence = _event_evidence(event)
        running = max(running, evidence)
        scores.append(max(0.0, min(1.0, 0.65 * evidence + 0.35 * running)))
    if scores:
        scores[-1] = outcome_reward(trace)
    return scores


def _keep_indices(method: str, trace: Any) -> set[int]:
    events = _events(trace)
    if not events:
        return set()
    if method == "outcome_reward_rl":
        # Outcome-only policy keeps the terminal decision path and verifier tool.
        return {i for i, event in enumerate(events) if _get(event, "type") in {"tool", "aggregate", "stop"}}
    if method == "ruler_style_local":
        # The local trajectory judge keeps evidence-bearing events and terminal stop.
        return {i for i, event in enumerate(events) if _event_evidence(event) >= 0.35 or _get(event, "type") == "stop"}
    if method == "masprm_style_local":
        scores = masprm_event_scores(trace)
        threshold = 0.35
        return {i for i, score in enumerate(scores) if score >= threshold or _get(events[i], "type") == "stop"}
    raise ValueError(f"unknown method: {method}")


def _controlled_trace(trace: Any, keep: set[int]) -> Any:
    controlled = copy.deepcopy(trace)
    if isinstance(controlled, dict):
        controlled["events"] = [event for i, event in enumerate(controlled.get("events", [])) if i in keep]
    else:
        controlled.events = [event for i, event in enumerate(controlled.events) if i in keep]
    return controlled


def _trace_row(trace: Any, method: str) -> dict[str, Any]:
    keep = _keep_indices(method, trace)
    controlled = _controlled_trace(trace, keep)
    telemetry = _telemetry(controlled)
    return {
        "trace_id": str(_get(trace, "trace_id")),
        "task_id": str(_get(trace, "task_id")),
        "method": method,
        "success": bool(_get(trace, "success", False)),
        "trajectory_score": ruler_trajectory_score(trace),
        "outcome_reward": outcome_reward(trace),
        "events_scored": len(_events(trace)) if method == "masprm_style_local" else len(keep),
        "events_kept": len(keep),
        "events_removed": len(_events(trace)) - len(keep),
        **telemetry,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    average = lambda key: mean(float(row[key]) for row in rows) if rows else 0.0
    events_scored = sum(int(row.get("events_scored", row.get("events", 0))) for row in rows)
    events_kept = sum(int(row.get("events_kept", row.get("events", 0))) for row in rows)
    events_removed = sum(int(row.get("events_removed", 0)) for row in rows)
    return {
        "method": rows[0].get("method") if rows else None,
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in rows) / count if count else 0.0,
        "mean_api_calls": average("api_calls"),
        "mean_tokens": average("tokens"),
        "mean_tool_calls": average("tool_calls"),
        "mean_latency_ms": average("latency_ms"),
        "events_scored": events_scored,
        "events_kept": events_kept,
        "events_removed": events_removed,
        "events_total": events_scored,
        "api_calls_for_run": 0,
        "implementation": "local_zero_provider_api_reimplementation",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_baseline(source_traces: Path, output_dir: Path, method: str, limit: Optional[int] = None, offset: int = 0) -> dict[str, Any]:
    traces = _load_jsonl(source_traces)[offset:]
    if limit is not None:
        traces = traces[:limit]
    rows = [_trace_row(trace, method) for trace in traces]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    summary = summarize_rows(rows)
    summary.update({"source_traces": str(source_traces), "subset": len(rows), "offset": offset})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps({"method": method, "api": False, "source_traces": str(source_traces), "limit": limit, "offset": offset}, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-traces", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=("outcome_reward_rl", "ruler_style_local", "masprm_style_local"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_baseline(args.source_traces, args.output_dir, args.method, args.limit, args.offset), indent=2))


if __name__ == "__main__":
    main()
