from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.counterfactuals.spider_operators import mutate_spider_event
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Trace
from carve.verifiers.spider import SpiderVerifier


def build_jobs(preflight_path: Path) -> dict[str, list[str]]:
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    return {
        row["task_id"]: [item["operator"] for item in row["operators"] if item.get("applicable") and item.get("changed") and float(item.get("semantic_delta", 0.0)) != 0.0]
        for row in report["rows"]
    }


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


def _replay(trace: Trace, case: SpiderCase, operator: str, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    writer_a, writer_b = _event(trace, "sql_writer_a"), _event(trace, "sql_writer_b")
    public_a, public_b = _event(trace, "public_sql_verifier_a"), _event(trace, "public_sql_verifier_b")
    mutated = mutate_spider_event(writer_a.clone(agent_role="sql_writer"), operator, random.Random(seed)).clone(agent_role="sql_writer_a")
    mutation_score = SpiderVerifier().verify(mutated.content, case)
    candidate_a, candidate_b = _sql(mutated.content), writer_b.content
    context = {"plan": _event(trace, "planner").content, "candidate_a": candidate_a, "candidate_b": candidate_b, "public_a": mutation_score.success, "public_b": bool(public_b.metadata.get("verifier_success"))}
    choice = str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)).strip().splitlines()[0].lower()
    if choice not in {"candidate_a", "candidate_b", "abstain"}:
        choice = "abstain"
    final = candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""
    final_score = SpiderVerifier().verify(final, case).score if final else 0.0
    telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {}
    return final_score, {"operator": operator, "selector_choice": choice, "mutated_sql_before": writer_a.content, "mutated_sql_after": candidate_a, "public_a": mutation_score.success, "public_b": bool(public_b.metadata.get("verifier_success")), "mutation_verifier_score": mutation_score.score, "final_sql": final, "api_telemetry": telemetry}


def run_structural(traces: list[Trace], cases: dict[str, SpiderCase], jobs: dict[str, list[str]], client: Any, output_dir: Path, *, seed: int, resume: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume:
        raise ValueError("structural v2 output must be create-once")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    done = {(row["trace_id"], row["operator_name"]) for row in rows}
    produced = []
    for trace in traces:
        for operator in jobs.get(trace.task_id, []):
            key = (trace.trace_id, operator)
            if key in done:
                continue
            try:
                score, evidence = _replay(trace, cases[trace.task_id], operator, client, seed)
                label = CreditLabel(trace.trace_id, "e2", "sql_writer_a", operator, float(trace.verifier_score or 0.0), [score], score - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"operator_set": "spider_dag_structural_cf_v2", "replay_evidence": evidence}).__dict__
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(trace.trace_id, "e2", "sql_writer_a", operator, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "spider_dag_structural_cf_v2", "error": f"{type(error).__name__}: {error}"}).__dict__
            rows.append(label)
            done.add(key)
            _write_atomic(path, rows)
            produced.append(label)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preflight-filtered Spider DAG structural counterfactuals")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    jobs = build_jobs(args.preflight)
    traces = [trace for trace in traces if trace.task_id in jobs]
    cases = {row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in (json.loads(line) for line in args.input_cases.read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = run_structural(traces, cases, jobs, OpenAICompatibleClient(APIClientConfig.from_env()), args.output_dir, seed=args.seed, resume=args.resume)
    print(json.dumps({"traces": len(traces), "labels": len(rows), "api_calls": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
