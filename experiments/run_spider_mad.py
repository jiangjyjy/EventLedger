from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.spider import SpiderCase
from carve.verifiers.spider import SpiderVerifier
from experiments.run_spider_single_cot import source_cases


def _contract(question: str, schema: str) -> str:
    return f"Question:\n{question}\n\nSQLite schema:\n{schema}"


def spider_solver_prompt(question: str, schema: str, role: str) -> str:
    return (
        f"You are {role} in a two-solver SQL debate. Independently write one executable SQLite SELECT or WITH query "
        "that answers the question using the schema. Return SQL only, with no Markdown or explanation.\n\n"
        + _contract(question, schema)
    )


def spider_critic_prompt(question: str, schema: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "You are a SQL debate critic. Compare the two candidate queries against the question and SQLite schema. "
        "Check table and column validity, joins, filters, grouping, aggregation, and ordering. Do not execute SQL. "
        "Return a concise critique.\n\n"
        + _contract(question, schema)
        + f"\n\nCandidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}"
    )


def spider_judge_prompt(question: str, schema: str, candidate_a: str, candidate_b: str, critique: str) -> str:
    return (
        "You are the final judge in a multi-agent SQL debate. Produce the best executable SQLite SELECT or WITH query "
        "for the question using the schema, candidate queries, and critique. Return SQL only, with no Markdown or explanation.\n\n"
        + _contract(question, schema)
        + f"\n\nCandidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\nCritique:\n{critique}"
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _task_ids(path: Path) -> set[str]:
    return {str(row["task_id"]) for row in _read_jsonl(path)}


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _call(client: OpenAICompatibleClient, role: str, prompt: str, seed: int) -> tuple[str, dict]:
    return client.complete(role, prompt, seed).strip(), client.last_completion_telemetry()


def run(cases: list[SpiderCase], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [case for case in cases if case.case_id not in completed and (retry_errors or case.case_id not in errors)]
    verifier = SpiderVerifier()
    for index, case in enumerate(pending):
        try:
            task_seed = seed + 4 * index
            candidate_a, telemetry_a = _call(client, "mad_sql_solver_a", spider_solver_prompt(case.question, case.schema, "solver A"), task_seed)
            candidate_b, telemetry_b = _call(client, "mad_sql_solver_b", spider_solver_prompt(case.question, case.schema, "solver B"), task_seed + 1)
            critique, telemetry_c = _call(client, "mad_sql_critic", spider_critic_prompt(case.question, case.schema, candidate_a, candidate_b), task_seed + 2)
            answer, telemetry_j = _call(client, "mad_sql_judge", spider_judge_prompt(case.question, case.schema, candidate_a, candidate_b, critique), task_seed + 3)
            score = verifier.verify(answer, case)
            _append(output / "traces.jsonl", {
                "task_id": case.case_id, "dataset": "spider", "method": "multi_agent_debate_mad4", "model": model,
                "candidate_a": candidate_a, "candidate_b": candidate_b, "critique": critique, "final_answer": answer,
                "success": score.success, "verifier_score": score.score, "verifier_details": score.details, "stderr": None,
                "telemetry": {"api_calls": 4, "stages": {"solver_a": telemetry_a, "solver_b": telemetry_b, "critic": telemetry_c, "judge": telemetry_j}},
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
    rows = _read_jsonl(output / "traces.jsonl")
    summary = {
        "method": "multi_agent_debate_mad4", "model": model, "source_tasks": len(cases), "completed_tasks": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "api_calls": sum(int((row.get("telemetry") or {}).get("api_calls", 0)) for row in rows),
        "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MAD-4 baseline on the CARVE Spider-100 subset")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--spider-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    cases = source_cases(args.source_run, args.spider_root)
    if args.limit is not None:
        cases = cases[:args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    config.max_tokens, config.timeout, config.retries_per_url = 512, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(cases, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model, resume=args.resume, retry_errors=args.retry_errors), indent=2))


if __name__ == "__main__":
    main()
