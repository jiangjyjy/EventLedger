from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from carve.schemas import Trace
from formal_ablation import VARIANT_NAMES, aggregate_label_scores, validate_formal_contract, variant_spec


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_traces(path: Path) -> list[Trace]:
    return [Trace.from_dict(row) for row in _read_jsonl(path)]


def _subset_traces(path: Path, task_ids: set[str]) -> list[Trace]:
    traces = [trace for trace in _read_traces(path) if trace.task_id in task_ids]
    if len(traces) != len(task_ids):
        raise ValueError(f"{path}: expected {len(task_ids)} traces, found {len(traces)}")
    return traces


def _subset_labels(path: Path, trace_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(path) if row.get("trace_id") in trace_ids]


def _terminal_answer(trace: Trace, events: list) -> str:
    for event in reversed(events):
        if event.type == "aggregate":
            return event.content
    for event in reversed(events):
        if event.type in {"revise", "msg"}:
            return event.content
    return events[-1].content if events else ""


def _generic_rows(traces: list[Trace], labels: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    from experiments.run_control import _verify_answer
    from carve.control.pruning import prune_negative_events

    task_ids = {trace.trace_id for trace in traces}
    scores = aggregate_label_scores(labels, variant=variant, trace_ids=task_ids)
    rows = []
    for trace in traces:
        if variant_spec(variant).decision_policy == "no_stop":
            controlled = trace
            answer = trace.final_answer
        else:
            controlled = prune_negative_events(trace, scores)
            answer = _terminal_answer(trace, controlled.events)
        verifier_score, oracle_score, success = _verify_answer(trace, answer)
        rows.append({
            "trace_id": trace.trace_id,
            "task_id": trace.task_id,
            "verifier_score": verifier_score if verifier_score is not None else oracle_score,
            "success": success,
            "kept_events": len(controlled.events),
            "removed_events": len(trace.events) - len(controlled.events),
        })
    return rows


def _spider_rows(traces: list[Trace], labels: list[dict[str, Any]], variant: str, spider_root: Path) -> list[dict[str, Any]]:
    from experiments.evaluate_spider_dag_student_control import _case_by_task, evaluate_branch_selection

    scores = aggregate_label_scores(labels, variant=variant, trace_ids={trace.trace_id for trace in traces})
    return [
        evaluate_branch_selection(trace, _case_by_task(spider_root, trace.task_id), scores)
        for trace in traces
    ]


def _openqa_rows(traces: list[Trace], labels: list[dict[str, Any]], variant: str, cases_path: Path) -> list[dict[str, Any]]:
    from carve.datasets.nq_openqa import load_nq_openqa_jsonl
    from experiments.evaluate_nq_openqa_student_control import evaluate_openqa_branch_selection

    cases = {case.task_id: case for case in load_nq_openqa_jsonl(cases_path)}
    scores = aggregate_label_scores(labels, variant=variant, trace_ids={trace.trace_id for trace in traces})
    return [evaluate_openqa_branch_selection(trace, cases[trace.task_id], scores) for trace in traces]


def _row(regime: str, variant: str, records: list[dict[str, Any]], *, status: str = "measured") -> dict[str, Any]:
    scores = [float(item["verifier_score"]) for item in records if item.get("verifier_score") is not None]
    return {
        "regime": regime,
        "variant": variant,
        "status": status,
        "tasks": len(records),
        "verifier_score": sum(scores) / len(scores) if scores else None,
        "success_rate": sum(int(bool(item.get("success"))) for item in records) / len(records) if records else None,
        "records": records,
        "provenance": {
            "api_calls": 0,
            "source": "saved_factual_traces_and_credit_labels",
            "verifier": "domain_local_verifier",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_formal_contract(manifest)
    configs = manifest["regimes"]
    sources = {
        "code_math": (args.gsm_traces, args.gsm_labels),
        "sql": (args.spider_traces, args.spider_labels),
        "openqa": (args.openqa_traces, args.openqa_labels),
    }
    rows = []
    for regime, (trace_path, label_path) in sources.items():
        task_ids = set(configs[regime]["task_ids"])
        traces = _subset_traces(trace_path, task_ids)
        labels = _subset_labels(label_path, {trace.trace_id for trace in traces})
        for variant in VARIANT_NAMES:
            spec = variant_spec(variant)
            if spec.requires_recollection:
                rows.append(_row(regime, variant, [], status="not_collected_requires_no_crn_replay"))
                continue
            if spec.requires_student_training and not args.ranking_student_summary:
                rows.append(_row(regime, variant, [], status="not_collected_requires_ranking_beta_zero_student"))
                continue
            if regime == "code_math":
                records = _generic_rows(traces, labels, variant)
            elif regime == "sql":
                records = _spider_rows(traces, labels, variant, args.spider_root)
            else:
                records = _openqa_rows(traces, labels, variant, args.openqa_cases)
            rows.append(_row(regime, variant, records))

    by_variant: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_NAMES:
        values = [item for item in rows if item["variant"] == variant and item["status"] == "measured"]
        by_variant[variant] = {
            regime: next((item["verifier_score"] for item in values if item["regime"] == regime), None)
            for regime in ("code_math", "sql", "openqa")
        }
        scores = [value for value in by_variant[variant].values() if value is not None]
        by_variant[variant]["avg"] = sum(scores) / len(scores) if len(scores) == 3 else None
    result = {"manifest": str(args.manifest), "api_calls": 0, "rows": rows, "table2": by_variant}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal 3x100 zero-API RQ2 ablation evaluator")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gsm-traces", type=Path, required=True)
    parser.add_argument("--gsm-labels", type=Path, required=True)
    parser.add_argument("--spider-traces", type=Path, required=True)
    parser.add_argument("--spider-labels", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path, required=True)
    parser.add_argument("--openqa-traces", type=Path, required=True)
    parser.add_argument("--openqa-labels", type=Path, required=True)
    parser.add_argument("--openqa-cases", type=Path, required=True)
    parser.add_argument("--ranking-student-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
