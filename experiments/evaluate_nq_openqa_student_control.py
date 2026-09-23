from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.schemas import Trace
from carve.student_lora.data import build_examples, load_traces
from experiments.evaluate_student_lora_control import _load_student, _read_split, _score_student_events
from experiments.nq_openqa_training import select_split_traces
from experiments.openqa_dag_rl import ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL, evaluate_openqa_action


def evaluate_openqa_branch_selection(trace: Trace, case: Any, scores: dict[str, float]) -> dict[str, Any]:
    score_a = float(scores.get(f"{trace.trace_id}::e3", 0.0))
    score_b = float(scores.get(f"{trace.trace_id}::e4", 0.0))
    action = ACTION_USE_FACTUAL if score_a == score_b else ACTION_USE_A if score_a > score_b else ACTION_USE_B
    outcome = evaluate_openqa_action(trace, case, action)
    factual = evaluate_openqa_action(trace, case, ACTION_USE_FACTUAL)
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "action": outcome.action_name,
        "factual_action": factual.action_name,
        "score_a": score_a,
        "score_b": score_b,
        "success": outcome.success,
        "verifier_score": outcome.verifier_score,
        "api_calls": outcome.api_calls,
        "saved_api_calls": outcome.saved_api_calls,
        "factual_success": factual.success,
        "factual_api_calls": factual.api_calls,
    }


def evaluate_openqa_selective_branch_selection(
    trace: Trace,
    case: Any,
    scores: dict[str, float],
    *,
    threshold: float,
) -> dict[str, Any]:
    score_a = float(scores.get(f"{trace.trace_id}::e3", 0.0))
    score_b = float(scores.get(f"{trace.trace_id}::e4", 0.0))
    margin = abs(score_a - score_b)
    shortcut = margin >= threshold and score_a != score_b
    action = ACTION_USE_A if shortcut and score_a > score_b else ACTION_USE_B if shortcut else ACTION_USE_FACTUAL
    outcome = evaluate_openqa_action(trace, case, action)
    factual = evaluate_openqa_action(trace, case, ACTION_USE_FACTUAL)
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "threshold": threshold,
        "margin": margin,
        "shortcut": shortcut,
        "action": outcome.action_name,
        "success": outcome.success,
        "verifier_score": outcome.verifier_score,
        "api_calls": outcome.api_calls,
        "saved_api_calls": outcome.saved_api_calls,
        "factual_success": factual.success,
        "factual_api_calls": factual.api_calls,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    return {
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "factual_success_rate": sum(int(row["factual_success"]) for row in records) / count if count else 0.0,
        "mean_api_calls": sum(row["api_calls"] for row in records) / count if count else 0.0,
        "mean_saved_api_calls": sum(row["saved_api_calls"] for row in records) / count if count else 0.0,
        "branch_actions": sum(int(row["action"] != row["factual_action"]) for row in records),
    }


def _selective_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(records)
    return {
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "factual_success_rate": sum(int(row["factual_success"]) for row in records) / count if count else 0.0,
        "mean_api_calls": sum(row["api_calls"] for row in records) / count if count else 0.0,
        "mean_saved_api_calls": sum(row["saved_api_calls"] for row in records) / count if count else 0.0,
        "shortcut_coverage": sum(int(row["shortcut"]) for row in records) / count if count else 0.0,
    }


def _choose_threshold(validation_scans: dict[float, list[dict[str, Any]]]) -> float:
    summaries = {threshold: _selective_summary(records) for threshold, records in validation_scans.items()}
    feasible = [
        threshold
        for threshold, summary in summaries.items()
        if summary["success_rate"] >= summary["factual_success_rate"]
    ]
    candidates = feasible or list(summaries)
    return max(candidates, key=lambda threshold: (summaries[threshold]["mean_saved_api_calls"], summaries[threshold]["success_rate"], threshold))


def run(args: argparse.Namespace) -> dict[str, Any]:
    student_run = Path(args.student_run).resolve()
    source_run = Path(args.source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite OpenQA control output: {output_dir}")
    split = _read_split(student_run / "split.json")
    traces = load_traces(source_run / "traces.jsonl")
    if args.cross_domain:
        if args.target_split_file is None:
            raise ValueError("--target-split-file is required with --cross-domain")
        # Load target-domain examples with the target split. The checkpoint's
        # source split has unrelated task IDs and would produce zero scores.
        target_split = _read_split(args.target_split_file)
        validation_traces = []
        test_traces = select_split_traces(traces, target_split, "test")
        bundle = build_examples(source_run, target_split)
    else:
        validation_traces = select_split_traces(traces, split, "validation")
        test_traces = select_split_traces(traces, split, "test")
        bundle = build_examples(source_run, split)
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.cases)}
    if any(trace.task_id not in cases for trace in validation_traces + test_traces):
        raise ValueError("held-out OpenQA cases are missing")
    device = torch.device(args.device)
    model, tokenizer, config = _load_student(student_run, device)
    scores = _score_student_events(
        model,
        tokenizer,
        validation_traces + test_traces,
        bundle,
        max_event_tokens=int(config["max_event_tokens"]),
        event_micro_batch_size=int(config["event_micro_batch_size"]),
        device=device,
    )
    records = [evaluate_openqa_branch_selection(trace, cases[trace.task_id], scores) for trace in test_traces]
    validation_margins = {abs(float(scores.get(f"{trace.trace_id}::e3", 0.0)) - float(scores.get(f"{trace.trace_id}::e4", 0.0))) for trace in validation_traces}
    thresholds = sorted({0.0, *validation_margins, max(validation_margins, default=0.0) + 1e-6})
    validation_scans = {
        threshold: [evaluate_openqa_selective_branch_selection(trace, cases[trace.task_id], scores, threshold=threshold) for trace in validation_traces]
        for threshold in thresholds
    }
    selected_threshold = _choose_threshold(validation_scans) if validation_scans else 0.0
    test_selective_records = [evaluate_openqa_selective_branch_selection(trace, cases[trace.task_id], scores, threshold=selected_threshold) for trace in test_traces]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    with (output_dir / "selective_test_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in test_selective_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "evaluation": "held_out_nq_openqa_student_branch_control",
        "student_run": str(student_run),
        "source_run": str(source_run),
        "api_calls": 0,
        "test_tasks": [trace.task_id for trace in test_traces],
        "summary": _summary(records),
        "selective_control": {
            "threshold_tuned_on": "validation",
            "selected_threshold": selected_threshold,
            "validation_scan": {str(threshold): _selective_summary(scan) for threshold, scan in validation_scans.items()},
            "held_out_test": _selective_summary(test_selective_records),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-API held-out NQ-open student branch control evaluation")
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cross-domain", action="store_true")
    parser.add_argument("--target-split-file", type=Path)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
