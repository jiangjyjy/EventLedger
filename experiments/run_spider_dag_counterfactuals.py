from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.counterfactuals.selection import CounterfactualJob
from carve.counterfactuals.spider_dag_replay import replay_branch_dropout
from carve.counterfactuals.spider_operators import mutate_spider_event
from carve.datasets.spider import SpiderCase
from carve.schemas import CreditLabel, Trace
from carve.verifiers.spider import SpiderVerifier


def select_dag_jobs(trace: Trace) -> list[CounterfactualJob]:
    if trace.manifest.get("graph") != "spider_parallel_dag_v1":
        raise ValueError("DAG counterfactuals require spider_parallel_dag_v1 traces")
    return [
        CounterfactualJob("e1", "assign", "plan_ablation", 1.0, metadata={"operator_set": "spider_dag_v1"}),
        CounterfactualJob("e2", "revise", "projection_swap", 2.0, metadata={"operator_set": "spider_dag_v1"}),
        CounterfactualJob("e3", "revise", "projection_swap", 2.0, metadata={"operator_set": "spider_dag_v1"}),
        CounterfactualJob("e2", "revise", "drop_branch", 3.0, metadata={"operator_set": "spider_dag_v1", "dropped_branch": "a"}),
    ]


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_labels_atomic(path: Path, labels: dict[tuple[str, str, str], CreditLabel]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for label in labels.values():
            handle.write(json.dumps(label.__dict__, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _selector_continuation(trace: Trace, case: SpiderCase, client: Any, seed: int, *, plan: str, candidate_a: str, candidate_b: str, public_a: bool, public_b: bool, parents: list[str], intervention: str) -> Trace:
    verifier = SpiderVerifier()
    context = {"plan": plan, "candidate_a": candidate_a, "candidate_b": candidate_b, "public_a": public_a, "public_b": public_b}
    choice = str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)).strip().splitlines()[0].lower()
    telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {}
    if choice not in {"candidate_a", "candidate_b", "abstain"}:
        choice = "abstain"
    selector = _event(trace, "selector").clone(content=choice, parents=parents, metadata={"choice": choice, "counterfactual": intervention, "counterfactual_reexecuted": True})
    final = candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""
    final_event = _event(trace, "final_resolver").clone(content=final, parents=[selector.event_id], metadata={"choice": choice, "counterfactual_reexecuted": True})
    hidden_score = verifier.verify(final, case) if final else None
    hidden = _event(trace, "hidden_sql_verifier").clone(content=json.dumps(hidden_score.details if hidden_score else {"tests_passed": False}, sort_keys=True), parents=[final_event.event_id], metadata={"verifier_score": hidden_score.score if hidden_score else 0.0, "verifier_success": bool(hidden_score and hidden_score.success), "counterfactual_reexecuted": True})
    prefix = [event for event in trace.events if event.event_id not in {selector.event_id, final_event.event_id, hidden.event_id}]
    return trace.clone_with_events(prefix + [selector, final_event, hidden], final_answer=final, verifier_score=hidden_score.score if hidden_score else 0.0, success=bool(hidden_score and hidden_score.success), manifest={**trace.manifest, "replay_mode": "spider_dag_selector_continuation", "reexecuted_api_calls": int(telemetry.get("api_calls", 1)), "reexecuted_telemetry": telemetry, "intervention": intervention})


def replay_dag_job(trace: Trace, job: CounterfactualJob, case: SpiderCase, client: Any, seed: int) -> Trace:
    if job.operator_name == "drop_branch":
        return replay_branch_dropout(trace, str(job.metadata["dropped_branch"]), case, client, seed)
    planner = _event(trace, "planner")
    writer_a = _event(trace, "sql_writer_a")
    writer_b = _event(trace, "sql_writer_b")
    public_a = _event(trace, "public_sql_verifier_a")
    public_b = _event(trace, "public_sql_verifier_b")
    plan = "Join relationship evidence withheld." if job.event_id == "e1" else planner.content
    candidate_a, candidate_b = writer_a.content, writer_b.content
    parents = ["e1", "e2", "e3", "e4", "e5"]
    prefix = list(trace.events[:5])
    if job.event_id == "e1":
        prefix[0] = planner.clone(content=plan, metadata={**planner.metadata, "counterfactual": job.operator_name})
    if job.event_id in {"e2", "e3"}:
        original = writer_a if job.event_id == "e2" else writer_b
        mutated = mutate_spider_event(original.clone(agent_role="sql_writer"), job.operator_name, random.Random(seed)).clone(agent_role=original.agent_role)
        score = SpiderVerifier().verify(mutated.content, case)
        if job.event_id == "e2":
            candidate_a = _sql(mutated.content)
            prefix[1] = mutated
            prefix[3] = public_a.clone(content=json.dumps(score.details, sort_keys=True), metadata={**public_a.metadata, "verifier_score": score.score, "verifier_success": score.success, "counterfactual_reexecuted": True})
            public_a = prefix[3]
        else:
            candidate_b = _sql(mutated.content)
            prefix[2] = mutated
            prefix[4] = public_b.clone(content=json.dumps(score.details, sort_keys=True), metadata={**public_b.metadata, "verifier_score": score.score, "verifier_success": score.success, "counterfactual_reexecuted": True})
            public_b = prefix[4]
    replayed = _selector_continuation(trace, case, client, seed, plan=plan, candidate_a=candidate_a, candidate_b=candidate_b, public_a=bool(public_a.metadata.get("verifier_success")), public_b=bool(public_b.metadata.get("verifier_success")), parents=parents, intervention=job.operator_name)
    return replayed.clone_with_events(prefix + replayed.events[-3:], final_answer=replayed.final_answer, verifier_score=replayed.verifier_score, success=replayed.success, manifest=replayed.manifest)


def replay_evidence(trace: Trace, replayed: Trace, job: CounterfactualJob) -> dict[str, Any]:
    replayed_events = {event.agent_role: event for event in replayed.events}
    mutated_role = "sql_writer_a" if job.operator_name == "projection_swap" and job.event_id == "e2" else "sql_writer_b" if job.operator_name == "projection_swap" and job.event_id == "e3" else None
    original_sql = _event(trace, mutated_role).content if mutated_role else None
    replayed_sql = replayed_events[mutated_role].content if mutated_role and mutated_role in replayed_events else None
    return {
        "operator_name": job.operator_name,
        "selector_choice": replayed_events["selector"].content,
        "final_sql": replayed.final_answer,
        "public_a": replayed_events["public_sql_verifier_a"].metadata.get("verifier_success") if "public_sql_verifier_a" in replayed_events else None,
        "public_b": replayed_events["public_sql_verifier_b"].metadata.get("verifier_success") if "public_sql_verifier_b" in replayed_events else None,
        "mutated_sql_before": original_sql,
        "mutated_sql_after": replayed_sql,
        "replay_verifier_score": float(replayed.verifier_score or 0.0),
    }


def run_dag_jobs(traces: list[Trace], cases_by_id: dict[str, SpiderCase], client: Any, output_dir: Path, *, seed: int, resume: bool = False, retry_abstained: bool = False) -> list[CreditLabel]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path, label_path = output_dir / "traces.jsonl", output_dir / "credit_labels.jsonl"
    if (trace_path.exists() or label_path.exists()) and not resume:
        raise ValueError("DAG counterfactual output must be create-once")
    existing_tasks = {json.loads(line)["task_id"] for line in trace_path.read_text().splitlines() if line.strip()} if trace_path.exists() else set()
    with trace_path.open("a", encoding="utf-8") as handle:
        for trace in traces:
            if trace.task_id in existing_tasks:
                continue
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            handle.flush(); os.fsync(handle.fileno())
    previous = [CreditLabel(**json.loads(line)) for line in label_path.read_text().splitlines() if line.strip()] if label_path.exists() else []
    by_key = {(label.trace_id, label.event_id, label.operator_name): label for label in previous}
    labels: list[CreditLabel] = []
    for trace in traces:
        case = cases_by_id[trace.task_id]
        for job in select_dag_jobs(trace):
            key = (trace.trace_id, job.event_id, job.operator_name)
            if key in by_key and not (retry_abstained and by_key[key].abstained):
                continue
            try:
                replayed = replay_dag_job(trace, job, case, client, seed)
                label = CreditLabel(trace.trace_id, job.event_id, trace.get_event(job.event_id).agent_role, job.operator_name, float(trace.verifier_score or 0.0), [float(replayed.verifier_score or 0.0)], float(replayed.verifier_score or 0.0) - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"job": job.__dict__, "replay_manifest": replayed.manifest, "replay_evidence": replay_evidence(trace, replayed, job)})
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(trace.trace_id, job.event_id, trace.get_event(job.event_id).agent_role, job.operator_name, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"job": job.__dict__, "error": f"{type(error).__name__}: {error}"})
            by_key[key] = label
            labels.append(label)
            _write_labels_atomic(label_path, by_key)
    return labels


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Collect Spider DAG counterfactual credit labels")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    traces = traces[args.offset:] if args.limit is None else traces[args.offset:args.offset + args.limit]
    cases = {
        row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])})
        for row in (json.loads(line) for line in args.input_cases.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    labels = run_dag_jobs(traces, cases, OpenAICompatibleClient(APIClientConfig.from_env()), args.output_dir, seed=args.seed, resume=args.resume, retry_abstained=args.retry_abstained)
    print(json.dumps({"traces": len(traces), "labels": len(labels), "api_calls": len(labels)}, sort_keys=True))


if __name__ == "__main__":
    main()
