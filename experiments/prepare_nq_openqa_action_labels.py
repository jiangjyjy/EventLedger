from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.schemas import Trace
from carve.student_lora.data import load_traces
from experiments.openqa_dag_rl import ACTION_NAMES, ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL, evaluate_openqa_action


def _state(trace: Trace, case: Any) -> dict[str, str]:
    return {
        "question": case.question,
        "retrieved_evidence": trace.get_event("e1").content,
        "router_assignment": trace.get_event("e2").content,
    }


def build_action_labels(traces: list[Trace], cases: dict[str, Any], *, efficiency_weight: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        case = cases[trace.task_id]
        outcomes = {
            name: evaluate_openqa_action(trace, case, action)
            for name, action in zip(ACTION_NAMES, (ACTION_USE_A, ACTION_USE_B, ACTION_USE_FACTUAL), strict=True)
        }
        factual = outcomes["use_factual_selector"]
        values = {
            name: {
                "verifier_score": outcome.verifier_score,
                "success": outcome.success,
                "api_calls": outcome.api_calls,
                "saved_api_calls": outcome.saved_api_calls,
                "safe_vs_factual": outcome.success >= factual.success,
                "utility": outcome.verifier_score + efficiency_weight * outcome.saved_api_calls / max(1, outcome.raw_api_calls),
            }
            for name, outcome in outcomes.items()
        }
        safe_actions = [name for name in ACTION_NAMES if values[name]["safe_vs_factual"]]
        teacher_action = max(safe_actions, key=lambda name: (values[name]["utility"], name == "use_factual_selector"))
        rows.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "state": _state(trace, case),
                "action_values": values,
                "teacher_action": teacher_action,
                "efficiency_weight": efficiency_weight,
                "label_source": "stored_trace_candidates_hidden_alias_em",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build zero-API pre-generation OpenQA action labels")
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--efficiency-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite action labels: {args.output}")
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.cases)}
    rows = build_action_labels(load_traces(args.traces), cases, efficiency_weight=args.efficiency_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"traces": len(rows), "actions": len(rows) * len(ACTION_NAMES)}, sort_keys=True))


if __name__ == "__main__":
    main()
