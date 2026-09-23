from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.counterfactuals.selection import CounterfactualJob
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Trace
from carve.verifiers.spider import SpiderVerifier


SELECTOR_OPERATORS = (
    "force_candidate_a",
    "force_candidate_b",
    "force_abstain",
    "mask_public_a_status",
    "mask_public_b_status",
    "hide_candidate_a",
    "hide_candidate_b",
)


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def select_selector_jobs(trace: Trace, *, all_operators: bool = False) -> list[CounterfactualJob]:
    if trace.manifest.get("graph") != "spider_parallel_dag_v1":
        raise ValueError("selector counterfactuals require spider_parallel_dag_v1 traces")
    factual_choice = _event(trace, "selector").content
    if factual_choice not in {"candidate_a", "candidate_b", "abstain"}:
        factual_choice = "abstain"
    if all_operators:
        names = SELECTOR_OPERATORS
    elif factual_choice == "abstain":
        names = ("force_abstain", "force_candidate_a", "force_candidate_b")
    else:
        opposite = "force_candidate_b" if factual_choice == "candidate_a" else "force_candidate_a"
        unselected = "b" if factual_choice == "candidate_a" else "a"
        public_unselected = bool(_event(trace, f"public_sql_verifier_{unselected}").metadata.get("verifier_success"))
        information_operator = f"hide_candidate_{unselected}" if public_unselected else f"mask_public_{'a' if factual_choice == 'candidate_a' else 'b'}_status"
        names = ("force_abstain", opposite, information_operator)
    return [
        CounterfactualJob(
            "e6",
            "aggregate",
            name,
            3.0,
            metadata={
                "operator_set": "spider_dag_selector_v2",
                "factual_choice": factual_choice,
                "requires_api": name.startswith("mask_") or name.startswith("hide_"),
                "selection_reason": "fixed_selector_budget" if not all_operators else "selector_operator_enumeration",
            },
        )
        for name in names
    ]


def _write_atomic(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows.values():
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _context(trace: Trace, operator: str) -> tuple[dict[str, Any], list[str]]:
    candidate_a = _sql(_event(trace, "sql_writer_a").content)
    candidate_b = _sql(_event(trace, "sql_writer_b").content)
    public_a = bool(_event(trace, "public_sql_verifier_a").metadata.get("verifier_success"))
    public_b = bool(_event(trace, "public_sql_verifier_b").metadata.get("verifier_success"))
    parents = ["e1", "e2", "e3", "e4", "e5"]
    context: dict[str, Any] = {"plan": _event(trace, "planner").content, "candidate_a": candidate_a, "candidate_b": candidate_b, "public_a": public_a, "public_b": public_b}
    if operator == "mask_public_a_status":
        context.pop("public_a")
        parents.remove("e4")
    elif operator == "mask_public_b_status":
        context.pop("public_b")
        parents.remove("e5")
    elif operator == "hide_candidate_a":
        context.pop("candidate_a")
        context.pop("public_a")
        parents = [parent for parent in parents if parent not in {"e2", "e4"}]
    elif operator == "hide_candidate_b":
        context.pop("candidate_b")
        context.pop("public_b")
        parents = [parent for parent in parents if parent not in {"e3", "e5"}]
    return context, parents


def replay_selector_job(trace: Trace, case: SpiderCase, job: CounterfactualJob, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    verifier = SpiderVerifier()
    context, parents = _context(trace, job.operator_name)
    if job.operator_name.startswith("force_"):
        choice = job.operator_name.removeprefix("force_")
        telemetry: dict[str, Any] = {"api_calls": 0, "api_request_attempts": 0, "token_source": "none"}
    else:
        choice = str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)).strip().splitlines()[0].lower()
        telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {}
    available = {name for name in ("candidate_a", "candidate_b") if name in context}
    if choice not in available | {"abstain"}:
        choice = "abstain"
    final = context.get(choice, "")
    score = verifier.verify(final, case).score if final else 0.0
    return score, {
        "operator": job.operator_name,
        "factual_choice": job.metadata["factual_choice"],
        "selector_choice": choice,
        "available_candidates": sorted(available),
        "selector_parents": parents,
        "final_sql": final,
        "api_telemetry": telemetry,
    }


def run_selector_jobs(
    traces: list[Trace],
    cases: dict[str, SpiderCase],
    client: Any,
    output_dir: Path,
    *,
    seed: int,
    resume: bool = False,
    retry_abstained: bool = False,
    all_operators: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume:
        raise ValueError("selector v2 output must be create-once")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    by_key = {(row["trace_id"], row["operator_name"]): row for row in existing}
    produced = []
    for trace in traces:
        for job in select_selector_jobs(trace, all_operators=all_operators):
            key = (trace.trace_id, job.operator_name)
            previous = by_key.get(key)
            if previous and not (retry_abstained and previous["abstained"]):
                continue
            try:
                score, evidence = replay_selector_job(trace, cases[trace.task_id], job, client, seed)
                row = CreditLabel(trace.trace_id, "e6", "selector", job.operator_name, float(trace.verifier_score or 0.0), [score], score - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"operator_set": "spider_dag_selector_v2", "credit_scheme": "selector_direct_delta", "job": job.__dict__, "replay_evidence": evidence}).__dict__
            except (RuntimeError, TimeoutError) as error:
                row = CreditLabel(trace.trace_id, "e6", "selector", job.operator_name, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "spider_dag_selector_v2", "job": job.__dict__, "error": f"{type(error).__name__}: {error}"}).__dict__
            by_key[key] = row
            _write_atomic(path, by_key)
            produced.append(row)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Spider DAG selector/stopper counterfactual labels")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    parser.add_argument("--all-selector-operators", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        traces = traces[:args.limit]
    cases = {row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in (json.loads(line) for line in args.input_cases.read_text(encoding="utf-8").splitlines() if line.strip())}
    rows = run_selector_jobs(traces, cases, OpenAICompatibleClient(APIClientConfig.from_env()), args.output_dir, seed=args.seed, resume=args.resume, retry_abstained=args.retry_abstained, all_operators=args.all_selector_operators)
    print(json.dumps({"traces": len(traces), "labels": len(rows), "api_calls": sum(int(row.get("metadata", {}).get("replay_evidence", {}).get("api_telemetry", {}).get("api_calls", 0)) for row in rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
