from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from carve.counterfactuals.selection import CounterfactualJob
from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from experiments.run_nq_openqa_traces import configure_openqa_api
from carve.schemas import CreditLabel, Trace
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.agents.nq_openqa_runner import _evidence_check, _format_contexts, _parse_reader_output


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def select_selector_jobs(trace: Trace) -> list[CounterfactualJob]:
    if trace.manifest.get("workflow") != "nq_openqa_dpr_dag_v2":
        raise ValueError("OpenQA selector counterfactuals require nq_openqa_dpr_dag_v2 traces")
    factual_choice = _event(trace, "final_resolver").metadata.get("choice", "abstain")
    if factual_choice not in {"candidate_a", "candidate_b", "abstain"}:
        factual_choice = "abstain"
    if factual_choice == "candidate_a":
        names = ("force_abstain", "force_candidate_b", "hide_candidate_b")
    elif factual_choice == "candidate_b":
        names = ("force_abstain", "force_candidate_a", "hide_candidate_a")
    else:
        names = ("force_abstain", "force_candidate_a", "force_candidate_b")
    return [
        CounterfactualJob(
            "e7",
            "aggregate",
            name,
            3.0,
            metadata={
                "operator_set": "nq_openqa_selector_v1",
                "factual_choice": factual_choice,
                "requires_api": name.startswith("hide_"),
                "selection_reason": "fixed_selector_budget",
            },
        )
        for name in names
    ]


def _write_atomic(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _replay(trace: Trace, case: Any, job: CounterfactualJob, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    candidate_a = _event(trace, "reader_a").metadata["answer"]
    candidate_b = _event(trace, "reader_b").metadata["answer"]
    choices = {"candidate_a": candidate_a, "candidate_b": candidate_b}
    if job.operator_name.startswith("force_"):
        choice = job.operator_name.removeprefix("force_")
        telemetry = {"api_calls": 0, "api_request_attempts": 0, "token_source": "none"}
    else:
        hidden = job.operator_name.removeprefix("hide_")
        choices.pop(hidden)
        prompt = "Question: " + case.question + "\n\n" + "\n".join(f"{name}: {answer}" for name, answer in choices.items()) + "\n\nReturn exactly candidate_a, candidate_b, or abstain."
        choice = str(client.complete("selector", prompt, seed)).strip().splitlines()[0].lower()
        telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {"api_calls": 1}
    if choice not in set(choices) | {"abstain"}:
        choice = "abstain"
    final = choices.get(choice, "")
    return OpenQAExactMatchVerifier().verify(final, case.answers).score, {"selector_choice": choice, "final_answer": final, "api_telemetry": telemetry}


def run_selector_jobs(traces: list[Trace], cases: dict[str, Any], client: Any, output_dir: Path, *, seed: int, resume: bool = False, retry_abstained: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume:
        raise ValueError("OpenQA selector output is create-once; pass resume=True")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    rows = {(row["trace_id"], row["event_id"], row["operator_name"]): row for row in existing}
    produced = []
    for trace in traces:
        for job in select_selector_jobs(trace):
            key = (trace.trace_id, job.event_id, job.operator_name)
            if key in rows and not (retry_abstained and rows[key].get("abstained")):
                continue
            try:
                score, evidence = _replay(trace, cases[trace.task_id], job, client, seed)
                label = CreditLabel(trace.trace_id, "e7", "selector", job.operator_name, float(trace.verifier_score or 0.0), [score], score - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"operator_set": "nq_openqa_selector_v1", "credit_scheme": "selector_direct_delta", "replay_evidence": evidence}).__dict__
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(trace.trace_id, "e7", "selector", job.operator_name, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "nq_openqa_selector_v1", "error": f"{type(error).__name__}: {error}"}).__dict__
            rows[key] = label
            _write_atomic(path, rows)
            produced.append(label)
    return produced


def select_reader_jobs(trace: Trace) -> list[CounterfactualJob]:
    if trace.manifest.get("workflow") != "nq_openqa_dpr_dag_v2":
        raise ValueError("OpenQA reader counterfactuals require nq_openqa_dpr_dag_v2 traces")
    return [
        CounterfactualJob(event_id, "revise", operator, 2.0, metadata={"operator_set": "nq_openqa_reader_v1", "requires_api": True})
        for event_id in ("e3", "e4")
        for operator in ("citation_remove", "answer_truncate")
    ]


def select_router_jobs(trace: Trace) -> list[CounterfactualJob]:
    if trace.manifest.get("workflow") != "nq_openqa_dpr_dag_v2":
        raise ValueError("OpenQA router counterfactuals require nq_openqa_dpr_dag_v2 traces")
    router = _event(trace, "evidence_router").metadata
    names = ["drop_passage_a", "drop_passage_b"]
    if router.get("reader_a_indices") != router.get("reader_b_indices"):
        names.append("swap_branch_evidence")
    return [
        CounterfactualJob("e2", "assign", name, 2.5, metadata={"operator_set": "nq_openqa_router_v1", "requires_api": True})
        for name in names
    ]


def _replay_reader(trace: Trace, case: Any, job: CounterfactualJob, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    target = "a" if job.event_id == "e3" else "b"
    answers = {"a": _event(trace, "reader_a").metadata["answer"], "b": _event(trace, "reader_b").metadata["answer"]}
    supports = {"a": bool(_event(trace, "evidence_check_a").metadata["answer_supported"]), "b": bool(_event(trace, "evidence_check_b").metadata["answer_supported"])}
    if job.operator_name == "citation_remove":
        supports[target] = False
    else:
        answers[target] = answers[target].split()[0] if answers[target].split() else ""
        supports[target] = False
    prompt = f"Question: {case.question}\nCandidate A: {answers['a']}\nA support: {supports['a']}\nCandidate B: {answers['b']}\nB support: {supports['b']}\nReturn exactly candidate_a, candidate_b, or abstain."
    choice = str(client.complete("selector", prompt, seed)).strip().splitlines()[0].lower()
    if choice not in {"candidate_a", "candidate_b", "abstain"}:
        choice = "abstain"
    final = answers.get(choice.removeprefix("candidate_"), "") if choice != "abstain" else ""
    score = OpenQAExactMatchVerifier().verify(final, case.answers).score
    telemetry = client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {"api_calls": 1}
    return score, {"selector_choice": choice, "final_answer": final, "mutated_reader": target, "api_telemetry": telemetry}


def run_reader_jobs(traces: list[Trace], cases: dict[str, Any], client: Any, output_dir: Path, *, seed: int, resume: bool = False, retry_abstained: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume:
        raise ValueError("OpenQA reader output is create-once; pass resume=True")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    rows = {(row["trace_id"], row["event_id"], row["operator_name"]): row for row in existing}
    produced = []
    for trace in traces:
        for job in select_reader_jobs(trace):
            key = (trace.trace_id, job.event_id, job.operator_name)
            if key in rows and not (retry_abstained and rows[key].get("abstained")):
                continue
            try:
                score, evidence = _replay_reader(trace, cases[trace.task_id], job, client, seed)
                label = CreditLabel(trace.trace_id, job.event_id, "reader", job.operator_name, float(trace.verifier_score or 0.0), [score], score - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"operator_set": "nq_openqa_reader_v1", "credit_scheme": "reader_structural_selector_delta", "replay_evidence": evidence}).__dict__
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(trace.trace_id, job.event_id, "reader", job.operator_name, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "nq_openqa_reader_v1", "error": f"{type(error).__name__}: {error}"}).__dict__
            rows[key] = label
            _write_atomic(path, rows)
            produced.append(label)
    return produced


def _replay_router(trace: Trace, case: Any, job: CounterfactualJob, client: Any, seed: int) -> tuple[float, dict[str, Any]]:
    router = _event(trace, "evidence_router").metadata
    indices = {"a": list(router["reader_a_indices"]), "b": list(router["reader_b_indices"])}
    if job.operator_name == "drop_passage_a":
        indices["a"] = indices["a"][1:]
        targets = ("a",)
    elif job.operator_name == "drop_passage_b":
        indices["b"] = indices["b"][1:]
        targets = ("b",)
    else:
        indices["a"], indices["b"] = indices["b"], indices["a"]
        targets = ("a", "b")
    answers = {key: _event(trace, f"reader_{key}").metadata["answer"] for key in ("a", "b")}
    supports = {key: bool(_event(trace, f"evidence_check_{key}").metadata["answer_supported"]) for key in ("a", "b")}
    mutated_readers: dict[str, dict[str, Any]] = {}
    calls: list[dict[str, Any]] = []
    for key in targets:
        evidence = _format_contexts(case, indices[key], 900)
        prompt = f"Question: {case.question}\n\nEvidence:\n{evidence}\n\nReturn exactly:\nAnswer: <short answer>\nEvidence: <passage index or none>\nConfidence: <0-1>"
        raw = str(client.complete(f"reader_{key}", prompt, seed)).strip()
        answers[key], citation, _ = _parse_reader_output(raw, indices[key])
        supports[key] = bool(_evidence_check(answers[key], citation, case)["answer_supported"])
        calls.append(client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {"api_calls": 1})
        mutated_readers[key] = {"answer": answers[key], "citation": citation, "answer_supported": supports[key], "context_indices": indices[key]}
    selector_prompt = f"Question: {case.question}\nCandidate A: {answers['a']}\nA support: {supports['a']}\nCandidate B: {answers['b']}\nB support: {supports['b']}\nReturn exactly candidate_a, candidate_b, or abstain."
    choice = str(client.complete("selector", selector_prompt, seed)).strip().splitlines()[0].lower()
    if choice not in {"candidate_a", "candidate_b", "abstain"}: choice = "abstain"
    final = answers.get(choice.removeprefix("candidate_"), "") if choice != "abstain" else ""
    calls.append(client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {"api_calls": 1})
    telemetry = {"api_calls": sum(int(call.get("api_calls", 0)) for call in calls), "api_request_attempts": sum(int(call.get("api_request_attempts", call.get("api_calls", 0))) for call in calls), "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls), "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls), "wall_clock_latency_ms": sum(float(call.get("wall_clock_latency_ms", 0.0)) for call in calls), "calls": calls}
    return OpenQAExactMatchVerifier().verify(final, case.answers).score, {"selector_choice": choice, "final_answer": final, "reexecuted_roles": [f"reader_{key}" for key in targets] + ["selector"], "mutated_readers": mutated_readers, "api_telemetry": telemetry}


def run_router_jobs(traces: list[Trace], cases: dict[str, Any], client: Any, output_dir: Path, *, seed: int, resume: bool = False, retry_abstained: bool = False) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True); path = output_dir / "credit_labels.jsonl"
    if path.exists() and not resume: raise ValueError("OpenQA router output is create-once; pass resume=True")
    existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    rows = {(row["trace_id"], row["event_id"], row["operator_name"]): row for row in existing}; produced = []
    for trace in traces:
        for job in select_router_jobs(trace):
            key = (trace.trace_id, job.event_id, job.operator_name)
            if key in rows and not (retry_abstained and rows[key].get("abstained")): continue
            try:
                score, evidence = _replay_router(trace, cases[trace.task_id], job, client, seed)
                label = CreditLabel(trace.trace_id, "e2", "router", job.operator_name, float(trace.verifier_score or 0.0), [score], score - float(trace.verifier_score or 0.0), 0.0, 1, False, "verifier", {"operator_set": "nq_openqa_router_v1", "credit_scheme": "router_reexecute_delta", "replay_evidence": evidence}).__dict__
            except (RuntimeError, TimeoutError) as error:
                label = CreditLabel(trace.trace_id, "e2", "router", job.operator_name, float(trace.verifier_score or 0.0), [], 0.0, 0.0, 0, True, "verifier", {"operator_set": "nq_openqa_router_v1", "error": f"{type(error).__name__}: {error}"}).__dict__
            rows[key] = label; _write_atomic(path, rows); produced.append(label)
    return produced


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable NQ-open selector, reader, and router counterfactuals")
    parser.add_argument("--input-traces", required=True, type=Path)
    parser.add_argument("--input-cases", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        traces = traces[:args.limit]
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.input_cases)}
    client = OpenAICompatibleClient(configure_openqa_api(APIClientConfig.from_env(), max_tokens=args.max_tokens))
    selector = run_selector_jobs(traces, cases, client, args.output_dir, seed=args.seed, resume=args.resume, retry_abstained=args.retry_abstained)
    reader = run_reader_jobs(traces, cases, client, args.output_dir, seed=args.seed, resume=True, retry_abstained=args.retry_abstained)
    router = run_router_jobs(traces, cases, client, args.output_dir, seed=args.seed, resume=True, retry_abstained=args.retry_abstained)
    print(json.dumps({"traces": len(traces), "selector_labels": len(selector), "reader_labels": len(reader), "router_labels": len(router)}, sort_keys=True))


if __name__ == "__main__":
    main()
