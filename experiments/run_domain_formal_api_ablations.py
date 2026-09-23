"""Domain-verifier API counterfactuals for the formal Table 2 subset.

This runner deliberately does not use the generic replay engine. Spider outcomes
are rechecked with SQLite and OpenQA outcomes with answer-alias exact match.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.nq_openqa_runner import _evidence_check, _format_contexts
from carve.agents.spider_dag_runner import _sql, spider_dag_prompt
from carve.counterfactuals.spider_operators import applicable_spider_operators, mutate_spider_event
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.datasets.spider import SpiderCase, load_spider_dev
from carve.schemas import Trace
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier


def independent_continuation_seeds(seed: int) -> tuple[int, int]:
    return seed, seed + 1


def should_run_job(previous: dict[str, Any] | None, *, resume: bool, retry_abstained: bool) -> bool:
    if previous is None:
        return True
    return bool(resume and retry_abstained and previous.get("abstained"))


def is_complete_job_set(scores_by_trace: dict[str, list[float]], *, trace_count: int) -> bool:
    return len(scores_by_trace) == trace_count and all(len(scores) == 2 for scores in scores_by_trace.values())


def spider_message_only_context(*, plan: str, candidate_a: str, candidate_b: str, public_a: bool, public_b: bool, hidden_branch: str) -> dict[str, Any]:
    context: dict[str, Any] = {"plan": plan, "candidate_a": candidate_a, "candidate_b": candidate_b, "public_a": public_a, "public_b": public_b}
    context.pop(f"candidate_{hidden_branch}")
    context.pop(f"public_{hidden_branch}")
    return context


def openqa_message_only_candidates(*, answer_a: str, answer_b: str, hidden_branch: str) -> dict[str, str]:
    choices = {"candidate_a": answer_a, "candidate_b": answer_b}
    choices.pop(f"candidate_{hidden_branch}")
    return choices


def _event(trace: Trace, role: str):
    return next(event for event in trace.events if event.agent_role == role)


def _choice(raw: str, available: set[str]) -> str:
    value = raw.strip().splitlines()[0].lower() if raw.strip() else "abstain"
    return value if value in available | {"abstain"} else "abstain"


def _telemetry(client: Any) -> dict[str, Any]:
    return client.last_completion_telemetry() if hasattr(client, "last_completion_telemetry") else {"api_calls": 1}


def _spider_case(spider_root: Path, task_id: str) -> SpiderCase:
    _, suffix = task_id.rsplit("-dev-", 1)
    case = load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
    if case.case_id != task_id:
        raise ValueError(f"Spider case mismatch for {task_id}")
    return case


def _spider_replay(trace: Trace, case: SpiderCase, client: Any, *, seed: int, context: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    choice = _choice(str(client.complete("selector", spider_dag_prompt("selector", case, context), seed=seed)), {name for name in ("candidate_a", "candidate_b") if name in context})
    final = context.get(choice, "")
    result = SpiderVerifier().verify(final, case) if final else None
    return float(result.score if result else 0.0), {"selector_choice": choice, "final_sql": final, "api_telemetry": _telemetry(client)}


def select_non_noop_spider_mutation(writer: Any, operators: list[str], *, seed: int) -> tuple[str, str]:
    """Choose the first stable typed mutation that actually changes the SQL."""
    original = _sql(writer.content)
    for offset, operator in enumerate(sorted(operators)):
        mutated = _sql(mutate_spider_event(writer.clone(agent_role="sql_writer"), operator, random.Random(seed + offset)).content)
        if mutated != original:
            return mutated, operator
    raise ValueError("no non-noop typed Spider mutation")


def _openqa_prompt(case: Any, choices: dict[str, str], supports: dict[str, bool]) -> str:
    blocks = [f"Question: {case.question}"]
    for name in ("candidate_a", "candidate_b"):
        if name in choices:
            branch = name[-1]
            blocks.append(f"{name}: {choices[name]}\n{branch.upper()} support: {supports[name]}")
    return "\n\n".join(blocks) + "\n\nReturn exactly candidate_a, candidate_b, or abstain. Missing candidates are unavailable."


def _openqa_replay(trace: Trace, case: Any, client: Any, *, seed: int, choices: dict[str, str], supports: dict[str, bool]) -> tuple[float, dict[str, Any]]:
    choice = _choice(str(client.complete("selector", _openqa_prompt(case, choices, supports), seed=seed)), set(choices))
    final = choices.get(choice, "")
    result = OpenQAExactMatchVerifier().verify(final, case.answers)
    return float(result.score), {"selector_choice": choice, "final_answer": final, "api_telemetry": _telemetry(client)}


def _spider_typed_context(trace: Trace, case: SpiderCase, branch: str, seed: int) -> tuple[dict[str, Any], str]:
    role = f"sql_writer_{branch}"
    writer = _event(trace, role)
    operators = applicable_spider_operators(writer.clone(agent_role="sql_writer"))
    if not operators:
        raise ValueError(f"no typed Spider operator applicable to {trace.trace_id}/{branch}")
    try:
        mutated, operator = select_non_noop_spider_mutation(writer, operators, seed=seed)
    except ValueError as error:
        raise ValueError(f"{error} for {trace.trace_id}/{branch}") from error
    a = _sql(_event(trace, "sql_writer_a").content)
    b = _sql(_event(trace, "sql_writer_b").content)
    if branch == "a":
        a = mutated
    else:
        b = mutated
    verifier = SpiderVerifier()
    return {"plan": _event(trace, "planner").content, "candidate_a": a, "candidate_b": b, "public_a": verifier.verify(a, case).success, "public_b": verifier.verify(b, case).success}, operator


def _openqa_typed_candidates(trace: Trace, case: Any, branch: str) -> tuple[dict[str, str], dict[str, bool], str]:
    answer_a = str(_event(trace, "reader_a").metadata["answer"])
    answer_b = str(_event(trace, "reader_b").metadata["answer"])
    answer = answer_a if branch == "a" else answer_b
    mutated = answer.split()[0] if answer.split() else ""
    if mutated == answer:
        mutated = ""
    choices = {"candidate_a": answer_a, "candidate_b": answer_b}
    choices[f"candidate_{branch}"] = mutated
    supports = {
        "candidate_a": bool(_event(trace, "evidence_check_a").metadata.get("answer_supported")),
        "candidate_b": bool(_event(trace, "evidence_check_b").metadata.get("answer_supported")),
    }
    supports[f"candidate_{branch}"] = False
    return choices, supports, "answer_truncate"


def _write_atomic(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _label(trace: Trace, branch: str, variant: str, factual: float, counterfactual: float, evidence: dict[str, Any], *, abstained: bool = False, error: str | None = None) -> dict[str, Any]:
    event_id = "e2" if trace.dataset == "spider" and branch == "a" else "e3" if trace.dataset == "spider" else "e3" if branch == "a" else "e4"
    family = "sql_writer_a" if trace.dataset == "spider" and branch == "a" else "sql_writer_b" if trace.dataset == "spider" else "reader"
    metadata = {"formal_variant": variant, "replay_evidence": evidence}
    if error:
        metadata["error"] = error
    return {"trace_id": trace.trace_id, "event_id": event_id, "operator_family": family, "operator_name": "message_nullify" if variant == "no_typed_operators" else evidence.get("operator", "typed"), "factual_score": factual, "counterfactual_scores": [] if abstained else [counterfactual], "delta_mean": 0.0 if abstained else counterfactual - factual, "delta_std": 0.0, "num_rollouts": 0 if abstained else 1, "abstained": abstained, "score_source": "verifier", "metadata": metadata}


def run_domain_jobs(*, domain: str, variant: str, traces: list[Trace], cases: dict[str, Any], client: Any, output_dir: Path, seed: int, resume: bool, retry_abstained: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "credit_labels.jsonl"
    existing = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()] if labels_path.exists() else []
    rows = {(row["trace_id"], row["event_id"]): row for row in existing}
    for trace in traces:
        case = cases[trace.task_id]
        for branch in ("a", "b"):
            key = (trace.trace_id, "e2" if domain == "sql" and branch == "a" else "e3" if domain == "sql" else "e3" if branch == "a" else "e4")
            if not should_run_job(rows.get(key), resume=resume, retry_abstained=retry_abstained):
                continue
            try:
                factual_seed, cf_seed = independent_continuation_seeds(seed)
                if domain == "sql":
                    base = {"plan": _event(trace, "planner").content, "candidate_a": _sql(_event(trace, "sql_writer_a").content), "candidate_b": _sql(_event(trace, "sql_writer_b").content), "public_a": bool(_event(trace, "public_sql_verifier_a").metadata.get("verifier_success")), "public_b": bool(_event(trace, "public_sql_verifier_b").metadata.get("verifier_success"))}
                    if variant == "no_typed_operators":
                        cf_context = spider_message_only_context(**base, hidden_branch=branch)
                        factual = float(trace.verifier_score or 0.0)
                        cf, evidence = _spider_replay(trace, case, client, seed=factual_seed, context=cf_context)
                    else:
                        factual, factual_evidence = _spider_replay(trace, case, client, seed=factual_seed, context=base)
                        cf_context, operator = _spider_typed_context(trace, case, branch, cf_seed)
                        cf, evidence = _spider_replay(trace, case, client, seed=cf_seed, context=cf_context)
                        evidence.update({"factual_replay": factual_evidence, "factual_seed": factual_seed, "counterfactual_seed": cf_seed, "operator": operator})
                else:
                    base_choices = {"candidate_a": str(_event(trace, "reader_a").metadata["answer"]), "candidate_b": str(_event(trace, "reader_b").metadata["answer"])}
                    supports = {"candidate_a": bool(_event(trace, "evidence_check_a").metadata.get("answer_supported")), "candidate_b": bool(_event(trace, "evidence_check_b").metadata.get("answer_supported"))}
                    if variant == "no_typed_operators":
                        choices = openqa_message_only_candidates(answer_a=base_choices["candidate_a"], answer_b=base_choices["candidate_b"], hidden_branch=branch)
                        support_view = {name: supports[name] for name in choices}
                        factual = float(trace.verifier_score or 0.0)
                        cf, evidence = _openqa_replay(trace, case, client, seed=factual_seed, choices=choices, supports=support_view)
                    else:
                        factual, factual_evidence = _openqa_replay(trace, case, client, seed=factual_seed, choices=base_choices, supports=supports)
                        choices, support_view, operator = _openqa_typed_candidates(trace, case, branch)
                        cf, evidence = _openqa_replay(trace, case, client, seed=cf_seed, choices=choices, supports=support_view)
                        evidence.update({"factual_replay": factual_evidence, "factual_seed": factual_seed, "counterfactual_seed": cf_seed, "operator": operator})
                rows[key] = _label(trace, branch, variant, factual, cf, evidence)
            except (RuntimeError, TimeoutError, ValueError, KeyError, StopIteration) as error:
                rows[key] = _label(trace, branch, variant, 0.0, 0.0, {}, abstained=True, error=f"{type(error).__name__}: {error}")
            _write_atomic(labels_path, rows)
    valid = [row for row in rows.values() if not row["abstained"]]
    by_trace: dict[str, list[float]] = {}
    for row in valid:
        by_trace.setdefault(row["trace_id"], []).append(float(row["counterfactual_scores"][0]))
    summary = {"regime": domain, "variant": variant, "status": "measured_api_counterfactual" if is_complete_job_set(by_trace, trace_count=len(traces)) else "partial_api_counterfactual", "tasks": len(traces), "success_rate": sum(sum(scores) / len(scores) for scores in by_trace.values()) / len(traces) if traces else 0.0, "verifier_score": sum(sum(scores) / len(scores) for scores in by_trace.values()) / len(traces) if traces else 0.0, "provenance": {"api_counterfactual_jobs": len(valid), "abstained_jobs": len(rows) - len(valid), "domain_verifier": "SQLite" if domain == "sql" else "OpenQAExactMatch"}}
    (output_dir / f"{variant}_{domain}_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal domain-specific Table 2 API counterfactuals")
    parser.add_argument("--domain", choices=("sql", "openqa"), required=True)
    parser.add_argument("--variant", choices=("no_typed_operators", "no_crn_pairing"), required=True)
    parser.add_argument("--input-traces", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-cases", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    traces = [Trace.from_dict(json.loads(line)) for line in args.input_traces.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        traces = traces[: args.limit]
    if args.domain == "sql":
        if args.spider_root is None:
            parser.error("--spider-root is required for sql")
        cases = {trace.task_id: _spider_case(args.spider_root, trace.task_id) for trace in traces}
    else:
        if args.openqa_cases is None:
            parser.error("--openqa-cases is required for openqa")
        all_cases = {case.task_id: case for case in load_nq_openqa_jsonl(args.openqa_cases)}
        cases = {trace.task_id: all_cases[trace.task_id] for trace in traces}
    print(json.dumps(run_domain_jobs(domain=args.domain, variant=args.variant, traces=traces, cases=cases, client=OpenAICompatibleClient(APIClientConfig.from_env()), output_dir=args.output_dir, seed=args.seed, resume=args.resume, retry_abstained=args.retry_abstained), sort_keys=True))


if __name__ == "__main__":
    main()
