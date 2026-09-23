from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from carve.datasets.spider import SpiderCase, load_spider_dev
from carve.schemas import Trace
from carve.student_lora.data import build_examples, load_traces
from carve.verifiers.spider import SpiderVerifier
from experiments.evaluate_student_lora_control import _load_student, _read_split, _score_student_events
from experiments.run_control import load_teacher_credit_scores


def _event_content(trace: Trace, event_id: str) -> str:
    return trace.get_event(event_id).content


def _event_score(trace: Trace, event_id: str, scores: dict[str, float]) -> float:
    return float(scores.get(f"{trace.trace_id}::{event_id}", 0.0))


def _factual_choice(trace: Trace) -> str:
    try:
        choice = _event_content(trace, "e6").strip().lower()
    except KeyError:
        return "candidate_a"
    return choice if choice in {"candidate_a", "candidate_b"} else "candidate_a"


def evaluate_branch_selection(trace: Trace, case: SpiderCase, scores: dict[str, float]) -> dict[str, Any]:
    """Select an independent Writer branch using event credits and reverify SQL."""
    score_a = _event_score(trace, "e2", scores)
    score_b = _event_score(trace, "e3", scores)
    if score_a == score_b:
        choice = _factual_choice(trace)
    else:
        choice = "candidate_a" if score_a > score_b else "candidate_b"
    sql = _event_content(trace, "e2" if choice == "candidate_a" else "e3")
    result = SpiderVerifier().verify(sql, case)
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "choice": choice,
        "factual_choice": _factual_choice(trace),
        "score_a": score_a,
        "score_b": score_b,
        "sql": sql,
        "verifier_score": float(result.score),
        "success": bool(result.success),
    }


def _case_by_task(spider_root: Path, task_id: str) -> SpiderCase:
    try:
        _, index_text = task_id.rsplit("-dev-", 1)
        index = int(index_text)
    except ValueError as error:
        raise ValueError(f"invalid Spider task ID: {task_id}") from error
    cases = load_spider_dev(spider_root, limit=1, offset=index)
    if not cases or cases[0].case_id != task_id:
        raise ValueError(f"Spider case mismatch for {task_id}")
    return cases[0]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "traces": count,
        "success_rate": sum(int(row["success"]) for row in records) / count if count else 0.0,
        "mean_verifier_score": sum(row["verifier_score"] for row in records) / count if count else 0.0,
        "factual_choice_matches": sum(int(row["choice"] == row["factual_choice"]) for row in records),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    student_run = Path(args.student_run).resolve()
    source_run = Path(args.source_run).resolve()
    split = _read_split(student_run / "split.json")
    traces = load_traces(source_run / "traces.jsonl")
    test_tasks = set(split.test)
    test_traces = [trace for trace in traces if trace.task_id in test_tasks]
    if len(test_traces) != len(test_tasks):
        raise ValueError("held-out Spider traces do not match the student test split")
    device = torch.device(args.device)
    bundle = build_examples(source_run, split)
    model, tokenizer, config = _load_student(student_run, device)
    student_scores = _score_student_events(
        model,
        tokenizer,
        test_traces,
        bundle,
        max_event_tokens=int(config["max_event_tokens"]),
        event_micro_batch_size=int(config["event_micro_batch_size"]),
        device=device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    teacher_scores, teacher_metadata = load_teacher_credit_scores(source_run)
    student_records = [evaluate_branch_selection(trace, _case_by_task(args.spider_root, trace.task_id), student_scores) for trace in test_traces]
    teacher_records = [evaluate_branch_selection(trace, _case_by_task(args.spider_root, trace.task_id), teacher_scores) for trace in test_traces]
    factual = {"traces": len(test_traces), "success_rate": sum(int(trace.success) for trace in test_traces) / len(test_traces)}
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "student_branch_records.jsonl", student_records)
    _write_jsonl(output_dir / "teacher_branch_records.jsonl", teacher_records)
    result = {
        "evaluation": "held_out_spider_dag_branch_selection_reverified",
        "student_run": str(student_run),
        "source_run": str(source_run),
        "test_tasks": sorted(test_tasks),
        "factual_selector": factual,
        "teacher_credit": _summary(teacher_records) | teacher_metadata,
        "student_credit": _summary(student_records),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out Spider DAG Writer-branch control using reverified SQLite execution")
    parser.add_argument("--student-run", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--spider-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
