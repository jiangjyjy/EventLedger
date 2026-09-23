from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.counterfactuals.spider_operators import applicable_spider_operators, mutate_spider_event
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Trace
from carve.verifiers.spider import SpiderVerifier


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def paired_shapley_credits(*, factual: float, mutated_a: float, mutated_b: float, mutated_both: float) -> dict[str, float]:
    return {
        "a": 0.5 * ((factual - mutated_a) + (mutated_b - mutated_both)),
        "b": 0.5 * ((factual - mutated_b) + (mutated_a - mutated_both)),
    }


def build_paired_jobs(traces: list[Trace], cases: dict[str, SpiderCase], seed: int) -> dict[str, list[str]]:
    verifier = SpiderVerifier()
    jobs: dict[str, list[str]] = {}
    for trace in traces:
        case = cases[trace.task_id]
        a, b = _event(trace, "sql_writer_a"), _event(trace, "sql_writer_b")
        operators = set(applicable_spider_operators(a.clone(agent_role="sql_writer"))) & set(applicable_spider_operators(b.clone(agent_role="sql_writer")))
        valid: list[str] = []
        for operator in sorted(operators):
            mutated_a = mutate_spider_event(a.clone(agent_role="sql_writer"), operator, random.Random(seed)).content
            mutated_b = mutate_spider_event(b.clone(agent_role="sql_writer"), operator, random.Random(seed)).content
            if mutated_a == a.content or mutated_b == b.content:
                continue
            if verifier.verify(mutated_a, case).score == verifier.verify(a.content, case).score:
                continue
            if verifier.verify(mutated_b, case).score == verifier.verify(b.content, case).score:
                continue
            valid.append(operator)
        jobs[trace.task_id] = valid
    return jobs


def limit_jobs_per_trace(jobs: dict[str, list[str]], maximum: int | None) -> dict[str, list[str]]:
    if maximum is None:
        return jobs
    if maximum < 1:
        raise ValueError("maximum must be positive")
    return {task_id: operators[:maximum] for task_id, operators in jobs.items()}


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _replay(trace: Trace, case: SpiderCase, a_sql: str, b_sql: str, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    verifier = SpiderVerifier()
    context = {
        "plan": _event(trace, "planner").content,
        "candidate_a": _sql(a_sql),
        "candidate_b": _sql(b_sql),
        "public_a": verifier.verify(a_sql, case).success,
        "public_b": verifier.verify(b_sql, case).success,
    }
    choice = str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)).strip().splitlines()[0].lower()
    if choice not in {"candidate_a", "candidate_b", "abstain"}:
        choice = "abstain"
    final = context[choice] if choice in {"candidate_a", "candidate_b"} else ""
    score = verifier.verify(final, case).score if final else 0.0
    telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {}
    return score, {"selector_choice": choice, "public_a": context["public_a"], "public_b": context["public_b"], "final_sql": final, "api_telemetry": telemetry}


def _label_for_job(trace: Trace, case: SpiderCase, operator: str, client: Any, seed: int) -> list[dict[str, Any]]:
    a, b = _event(trace, "sql_writer_a"), _event(trace, "sql_writer_b")
    mutated_a = mutate_spider_event(a.clone(agent_role="sql_writer"), operator, random.Random(seed)).content
    mutated_b = mutate_spider_event(b.clone(agent_role="sql_writer"), operator, random.Random(seed)).content
    factual = float(trace.verifier_score or 0.0)
    score_ma_b, evidence_ma_b = _replay(trace, case, mutated_a, b.content, client, seed)
    score_a_mb, evidence_a_mb = _replay(trace, case, a.content, mutated_b, client, seed)
    score_ma_mb, evidence_ma_mb = _replay(trace, case, mutated_a, mutated_b, client, seed)
    credits = paired_shapley_credits(factual=factual, mutated_a=score_ma_b, mutated_b=score_a_mb, mutated_both=score_ma_mb)
    values = {"factual": factual, "mutated_a": score_ma_b, "mutated_b": score_a_mb, "mutated_both": score_ma_mb}
    metadata = {"operator_set": "spider_dag_paired_structural_cf_v1", "credit_scheme": "paired_structural_shapley", "operator": operator, "coalition_values": values, "mutated_sql_a": mutated_a, "mutated_sql_b": mutated_b, "replays": {"mutated_a": evidence_ma_b, "mutated_b": evidence_a_mb, "mutated_both": evidence_ma_mb}}
    return [
        CreditLabel(trace.trace_id, "e2", "sql_writer_a", operator, factual, [score_ma_b, score_a_mb, score_ma_mb], credits["a"], 0.0, 1, False, "verifier", {**metadata, "branch": "a"}).__dict__,
        CreditLabel(trace.trace_id, "e3", "sql_writer_b", operator, factual, [score_ma_b, score_a_mb, score_ma_mb], credits["b"], 0.0, 1, False, "verifier", {**metadata, "branch": "b"}).__dict__,
    ]


def run_paired_structural(
    traces: list[Trace],
    cases: dict[str, SpiderCase],
    jobs: dict[str, list[str]],
    client: Any,
    output_dir: Path,
    *,
    seed: int,
    resume: bool = False,
    retry_abstained: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume:
        raise ValueError("paired structural output must be create-once")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in existing:
        grouped.setdefault((row["trace_id"], row["operator_name"]), []).append(row)
    produced = []
    for trace in traces:
        for operator in jobs.get(trace.task_id, []):
            key = (trace.trace_id, operator)
            previous = grouped.get(key, [])
            if previous and not (retry_abstained and all(row["abstained"] for row in previous)):
                continue
            if previous:
                existing = [row for row in existing if (row["trace_id"], row["operator_name"]) != key]
            try:
                rows = _label_for_job(trace, cases[trace.task_id], operator, client, seed)
            except (RuntimeError, TimeoutError) as error:
                rows = [CreditLabel(trace.trace_id, event_id, family, operator, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "spider_dag_paired_structural_cf_v1", "error": f"{type(error).__name__}: {error}"}).__dict__ for event_id, family in (("e2", "sql_writer_a"), ("e3", "sql_writer_b"))]
            existing.extend(rows)
            grouped[key] = rows
            _write_atomic(path, existing)
            produced.extend(rows)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect paired structural Shapley labels for Spider DAG traces")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-operators-per-trace", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        traces = traces[:args.limit]
    cases = {row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in (json.loads(line) for line in args.input_cases.read_text(encoding="utf-8").splitlines() if line.strip())}
    jobs = limit_jobs_per_trace(build_paired_jobs(traces, cases, args.seed), args.max_operators_per_trace)
    rows = run_paired_structural(
        traces,
        cases,
        jobs,
        OpenAICompatibleClient(APIClientConfig.from_env()),
        args.output_dir,
        seed=args.seed,
        resume=args.resume,
        retry_abstained=args.retry_abstained,
    )
    print(json.dumps({"traces": len(traces), "operator_jobs": sum(map(len, jobs.values())), "labels": len(rows), "api_calls": 3 * len(rows) // 2}, sort_keys=True))


if __name__ == "__main__":
    main()
