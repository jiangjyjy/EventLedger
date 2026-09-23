from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from carve.control import export_rl_samples, prune_negative_events, summarize_rl_export
from carve.rewards.stop import compute_stop_signal
from carve.schemas import Event, Trace
from carve.scoring.credit_value import effective_credit
from carve.verifiers import CodeVerifier, MathVerifier, RubricVerifier, SWEBenchVerifier


def scoped_event_key(trace_id: str | None, event_id: str) -> str:
    return f"{trace_id}::{event_id}" if trace_id else event_id


def load_control_event_scores(run_dir: Path) -> tuple[dict[str, float], dict]:
    reward_path = run_dir / "reward_labels.jsonl"
    if reward_path.exists():
        scores: dict[str, float] = {}
        for line in reward_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scores[scoped_event_key(row.get("trace_id"), row["event_id"])] = float(row["total_reward"])
        if scores:
            return scores, {"score_source": "reward_labels", "score_path": str(reward_path), "scored_events": len(scores)}

    credit_path = run_dir / "credit_labels.jsonl"
    if credit_path.exists():
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for line in credit_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            value = effective_credit(row)
            if value is None:
                continue
            key = scoped_event_key(row.get("trace_id"), row["event_id"])
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
        if totals:
            return (
                {key: total / counts[key] for key, total in totals.items()},
                {"score_source": "teacher_credit_labels", "score_path": str(credit_path), "scored_events": len(totals)},
            )
    return {}, {"score_source": "heuristic_event_type", "score_path": None, "scored_events": 0}


def load_teacher_credit_scores(run_dir: Path) -> tuple[dict[str, float], dict]:
    """Load teacher scores from credit labels without falling back to rewards."""
    credit_path = run_dir / "credit_labels.jsonl"
    if not credit_path.exists():
        return {}, {
            "score_source": "teacher_credit_labels",
            "score_path": str(credit_path),
            "scored_events": 0,
        }

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in credit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = effective_credit(row)
        if value is None:
            continue
        key = scoped_event_key(row.get("trace_id"), row["event_id"])
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1

    return (
        {key: total / counts[key] for key, total in totals.items()},
        {
            "score_source": "teacher_credit_labels",
            "score_path": str(credit_path),
            "scored_events": len(totals),
        },
    )


def fallback_event_scores(trace: Trace) -> dict[str, float]:
    return {scoped_event_key(trace.trace_id, event.event_id): (0.1 if event.type != "critique" else -0.1) for event in trace.events}


def score_trace_for_control(trace: Trace, event_scores: dict[str, float]) -> float:
    return float(sum(event_scores.get(scoped_event_key(trace.trace_id, event.event_id), event_scores.get(event.event_id, 0.0)) for event in trace.events))


def _terminal_answer(events: list[Event]) -> str:
    for event in reversed(events):
        if event.type == "aggregate":
            return event.content
    for event in reversed(events):
        if event.type in {"revise", "msg"}:
            return event.content
    return events[-1].content if events else ""


def _verify_answer(trace: Trace, answer: str) -> tuple[float | None, float | None, bool]:
    task = trace.manifest.get("task", {})
    if trace.dataset in {"humaneval", "mbpp"}:
        result = CodeVerifier().verify(answer, task.get("tests"))
        return float(result.score), None, bool(result.success)
    if trace.dataset == "gsm8k":
        result = MathVerifier().verify(answer, task.get("reference"))
        return float(result.score), None, bool(result.success)
    if trace.dataset == "swebench_lite":
        result = SWEBenchVerifier().verify(answer, task.get("tests"), repo_path=task.get("repo_path"), base_commit=task.get("base_commit"))
        return float(result.score), None, bool(result.success)
    result = RubricVerifier().verify(answer, task.get("reference", trace.final_answer))
    return None, float(result.score), bool(result.success)


def _trace_telemetry(trace: Trace) -> dict[str, float | int]:
    telemetry = [event.metadata.get("telemetry", {}) for event in trace.events if isinstance(event.metadata.get("telemetry"), dict)]
    return {
        "tokens": trace.total_tokens,
        "cost_usd": trace.total_cost_usd,
        "latency_ms": sum(event.latency_ms for event in trace.events),
        "tool_calls": sum(1 for event in trace.events if event.type == "tool"),
        "api_calls": sum(int(item.get("api_calls", 0)) for item in telemetry),
        "api_request_attempts": sum(int(item.get("api_request_attempts", item.get("api_calls", 0))) for item in telemetry),
        "input_tokens": sum(int(item.get("input_tokens", 0)) for item in telemetry),
        "output_tokens": sum(int(item.get("output_tokens", 0)) for item in telemetry),
    }


def evaluate_controlled_traces(traces: list[Trace], event_scores: dict[str, float]) -> tuple[list[Trace], dict]:
    controlled: list[Trace] = []
    records: list[dict] = []
    for trace in traces:
        scores = event_scores or fallback_event_scores(trace)
        pruned = prune_negative_events(trace, scores)
        answer = _terminal_answer(pruned.events)
        verifier_score, oracle_score, success = _verify_answer(trace, answer)
        removed = len(trace.events) - len(pruned.events)
        controlled_trace = pruned.clone_with_events(
            pruned.events,
            final_answer=answer,
            verifier_score=verifier_score,
            oracle_score=oracle_score,
            success=success,
            manifest={
                **trace.manifest,
                "control": {
                    "mode": "offline_structural_pruning",
                    "source_trace_id": trace.trace_id,
                    "verified_after_pruning": True,
                    "kept_events": len(pruned.events),
                    "removed_events": removed,
                },
            },
        )
        controlled.append(controlled_trace)
        records.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "success": success,
                "verifier_score": verifier_score,
                "oracle_score": oracle_score,
                "kept_events": len(pruned.events),
                "removed_events": removed,
                **_trace_telemetry(controlled_trace),
            }
        )

    count = len(controlled)
    total_kept = sum(record["kept_events"] for record in records)
    total_removed = sum(record["removed_events"] for record in records)
    summary = {
        "control_mode": "offline_structural_pruning_reverified",
        "controlled_traces": count,
        "controlled_success_rate": sum(int(record["success"]) for record in records) / count if count else 0.0,
        "controlled_mean_tokens": mean(record["tokens"] for record in records) if records else 0.0,
        "controlled_mean_cost_usd": mean(record["cost_usd"] for record in records) if records else 0.0,
        "controlled_mean_latency_ms": mean(record["latency_ms"] for record in records) if records else 0.0,
        "controlled_mean_tool_calls": mean(record["tool_calls"] for record in records) if records else 0.0,
        "controlled_mean_api_calls": mean(record["api_calls"] for record in records) if records else 0.0,
        "controlled_mean_api_request_attempts": mean(record["api_request_attempts"] for record in records) if records else 0.0,
        "controlled_mean_input_tokens": mean(record["input_tokens"] for record in records) if records else 0.0,
        "controlled_mean_output_tokens": mean(record["output_tokens"] for record in records) if records else 0.0,
        "kept_events_total": total_kept,
        "removed_events_total": total_removed,
        "mean_kept_events": total_kept / count if count else 0.0,
        "mean_removed_events": total_removed / count if count else 0.0,
        "controlled_records": records,
    }
    return controlled, summary


def summarize_controlled_trace(trace: Trace) -> dict:
    values = _trace_telemetry(trace)
    return {
        "controlled_success": bool(trace.success),
        "controlled_verifier_score": trace.verifier_score,
        "controlled_oracle_score": trace.oracle_score,
        **{f"controlled_{key}": value for key, value in values.items()},
    }


def control_reward_source(run_dir: Path) -> str:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return "teacher_rewards" if manifest.get("skip_student") else "CARVE-S"


def load_stop_signals(run_dir: Path) -> dict[str, dict]:
    credit_path = run_dir / "credit_labels.jsonl"
    if not credit_path.exists():
        return {}
    signals: dict[str, dict] = {}
    for line in credit_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("abstained", False):
            continue
        if row.get("operator_family") != "stop" and row.get("operator_name") not in {"force_continue", "force_stop"}:
            continue
        metadata = row.get("metadata", {})
        signal = compute_stop_signal(
            event_type="stop" if row.get("operator_name") == "force_continue" else "non_stop",
            factual_score=float(row.get("factual_score", 0.0)),
            counterfactual_scores=[float(value) for value in row.get("counterfactual_scores", [])],
            factual_cost=float(metadata.get("factual_cost", metadata.get("total_cost_usd", 0.0))),
            counterfactual_costs=[float(value) for value in metadata.get("counterfactual_costs", [])],
            beta_cost=float(metadata.get("beta_cost", 1.0)),
        )
        signals[scoped_event_key(row.get("trace_id"), row["event_id"])] = {
            "operator_name": row.get("operator_name"),
            "saved_cost": signal.saved_cost,
            "forgone_gain": signal.forgone_gain,
            "stop_reward": signal.stop_reward,
            "continued_score": signal.continued_score,
            "factual_score": signal.factual_score,
        }
    return signals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--reward-source", choices=["CARVE-S", "teacher_rewards"], default=None)
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    traces = [Trace.from_dict(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    loaded_event_scores, score_metadata = load_control_event_scores(run_dir)
    stop_signals = load_stop_signals(run_dir)
    controlled, aggregate = evaluate_controlled_traces(traces, loaded_event_scores)
    trace_scores = {trace.trace_id: score_trace_for_control(trace, loaded_event_scores) for trace in traces}
    reward_source = args.reward_source or control_reward_source(run_dir)
    samples = []
    for trace in controlled:
        samples.extend(export_rl_samples(trace, loaded_event_scores or fallback_event_scores(trace), reward_source=reward_source, policy_objective="ppo"))

    with (run_dir / "controlled_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in controlled:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str) + "\n")
    with (run_dir / "rl_samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.__dict__, ensure_ascii=False, default=str) + "\n")

    summary = {
        **aggregate,
        "trace_scores": trace_scores,
        "stop_signals": stop_signals,
        "stop_signal_count": len(stop_signals),
        "rl_samples": len(samples),
        **score_metadata,
        **summarize_rl_export(samples),
    }
    (run_dir / "rl_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
