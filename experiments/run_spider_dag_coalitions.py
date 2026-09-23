from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.counterfactuals.spider_dag_replay import replay_branch_dropout
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Trace
from carve.verifiers.spider import SpiderVerifier


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _score_sql(trace: Trace, case: SpiderCase, choice: str) -> float:
    role = "sql_writer_a" if choice == "candidate_a" else "sql_writer_b"
    return SpiderVerifier().verify(_event(trace, role).content, case).score


def _labels_for_trace(trace: Trace, case: SpiderCase, client: Any, seed: int) -> list[dict[str, Any]]:
    factual = float(trace.verifier_score or 0.0)
    a_only = replay_branch_dropout(trace, "b", case, client, seed)
    b_only = replay_branch_dropout(trace, "a", case, client, seed)
    values = {"empty": 0.0, "a_only": float(a_only.verifier_score or 0.0), "b_only": float(b_only.verifier_score or 0.0), "both": factual}
    credit_a = 0.5 * ((values["a_only"] - values["empty"]) + (values["both"] - values["b_only"]))
    credit_b = 0.5 * ((values["b_only"] - values["empty"]) + (values["both"] - values["a_only"]))
    factual_choice = _event(trace, "selector").content
    alternative = "candidate_b" if factual_choice == "candidate_a" else "candidate_a"
    alternative_score = _score_sql(trace, case, alternative) if factual_choice in {"candidate_a", "candidate_b"} else 0.0
    common = {"coalitions": values, "a_only_choice": _event(a_only, "selector").content, "b_only_choice": _event(b_only, "selector").content, "api_calls": int(a_only.manifest.get("reexecuted_api_calls", 1)) + int(b_only.manifest.get("reexecuted_api_calls", 1))}
    labels = [
        CreditLabel(trace.trace_id, "e2", "sql_writer_a", "shapley_branch", factual, [values["a_only"], values["b_only"], values["both"]], credit_a, 0.0, 1, False, "verifier", {**common, "branch": "a"}),
        CreditLabel(trace.trace_id, "e3", "sql_writer_b", "shapley_branch", factual, [values["a_only"], values["b_only"], values["both"]], credit_b, 0.0, 1, False, "verifier", {**common, "branch": "b"}),
        CreditLabel(trace.trace_id, "e6", "selector", "force_alternative", factual, [alternative_score], alternative_score - factual, 0.0, 1, False, "verifier", {"factual_choice": factual_choice, "alternative_choice": alternative, "alternative_score": alternative_score}),
    ]
    return [label.__dict__ for label in labels]


def _abstained_labels(trace: Trace, error: Exception) -> list[dict[str, Any]]:
    rows = []
    for event_id, family, operator in (("e2", "sql_writer_a", "shapley_branch"), ("e3", "sql_writer_b", "shapley_branch"), ("e6", "selector", "force_alternative")):
        rows.append(CreditLabel(trace.trace_id, event_id, family, operator, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"error": f"{type(error).__name__}: {error}"}).__dict__)
    return rows


def run_coalitions(traces: list[Trace], cases_by_id: dict[str, SpiderCase], client: Any, output_dir: Path, *, seed: int, resume: bool = False, retry_abstained: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "coalition_labels.jsonl"
    if output.exists() and not resume:
        raise ValueError("coalition output must be create-once")
    existing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()] if output.exists() else []
    grouped = {}
    for row in existing:
        grouped.setdefault(row["trace_id"], []).append(row)
    produced: list[dict[str, Any]] = []
    for trace in traces:
        previous = grouped.get(trace.trace_id, [])
        if previous and not (retry_abstained and all(row["abstained"] for row in previous)):
            continue
        if previous:
            existing = [row for row in existing if row["trace_id"] != trace.trace_id]
        try:
            rows = _labels_for_trace(trace, cases_by_id[trace.task_id], client, seed)
        except (RuntimeError, TimeoutError) as error:
            rows = _abstained_labels(trace, error)
        existing.extend(rows)
        grouped[trace.trace_id] = rows
        _write_atomic(output, existing)
        produced.extend(rows)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Spider DAG coalition counterfactual labels")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        traces = traces[:args.limit]
    cases = {row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in (json.loads(line) for line in args.input_cases.read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = run_coalitions(traces, cases, OpenAICompatibleClient(APIClientConfig.from_env()), args.output_dir, seed=args.seed, resume=args.resume, retry_abstained=args.retry_abstained)
    print(json.dumps({"traces": len(traces), "labels": len(rows), "api_calls": 2 * len(rows) // 3}, sort_keys=True))


if __name__ == "__main__":
    main()
