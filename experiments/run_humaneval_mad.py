from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.code import CodeVerifier
from experiments.run_humaneval_single_cot import humaneval_tests, source_tasks


def code_solver_prompt(contract: str, role: str) -> str:
    return (
        f"You are {role} in a two-coder debate. Implement the function contract independently. "
        "Return complete executable Python code only, with no Markdown or explanation.\n\nFunction contract:\n"
        + contract
    )


def code_critic_prompt(contract: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "You are a code-review critic. Compare both candidate implementations against the function contract. "
        "Check API compliance, edge cases, and likely logic errors. Do not execute code and do not invent tests. "
        "Return a concise review.\n\nFunction contract:\n"
        + contract
        + "\n\nCandidate A:\n"
        + candidate_a
        + "\n\nCandidate B:\n"
        + candidate_b
    )


def code_judge_prompt(contract: str, candidate_a: str, candidate_b: str, critique: str) -> str:
    return (
        "You are the final judge in a multi-agent code debate. Produce the best complete implementation for the "
        "function contract using the candidate code and critique. Return Python code only, with no Markdown or explanation.\n\n"
        "Function contract:\n"
        + contract
        + "\n\nCandidate A:\n"
        + candidate_a
        + "\n\nCandidate B:\n"
        + candidate_b
        + "\n\nCritique:\n"
        + critique
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


def run(tasks: list[dict], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [task for task in tasks if str(task["task_id"]) not in completed and (retry_errors or str(task["task_id"]) not in errors)]
    verifier = CodeVerifier()
    for index, task in enumerate(pending):
        try:
            contract, task_seed = str(task["prompt"]), seed + 4 * index
            candidate_a, telemetry_a = _call(client, "mad_coder_a", code_solver_prompt(contract, "coder A"), task_seed)
            candidate_b, telemetry_b = _call(client, "mad_coder_b", code_solver_prompt(contract, "coder B"), task_seed + 1)
            critique, telemetry_c = _call(client, "mad_code_critic", code_critic_prompt(contract, candidate_a, candidate_b), task_seed + 2)
            answer, telemetry_j = _call(client, "mad_code_judge", code_judge_prompt(contract, candidate_a, candidate_b, critique), task_seed + 3)
            score = verifier.verify(answer, humaneval_tests(task))
            _append(output / "traces.jsonl", {
                "task_id": task["task_id"], "dataset": "humaneval", "method": "multi_agent_debate_mad4", "model": model,
                "candidate_a": candidate_a, "candidate_b": candidate_b, "critique": critique, "final_answer": answer,
                "success": score.success, "verifier_score": score.score, "verifier_details": score.details, "stderr": score.stderr,
                "telemetry": {"api_calls": 4, "stages": {"solver_a": telemetry_a, "solver_b": telemetry_b, "critic": telemetry_c, "judge": telemetry_j}},
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    rows = _read_jsonl(output / "traces.jsonl")
    summary = {
        "method": "multi_agent_debate_mad4", "model": model, "source_tasks": len(tasks), "completed_tasks": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "api_calls": sum(int((row.get("telemetry") or {}).get("api_calls", 0)) for row in rows),
        "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MAD-4 baseline on the CARVE HumanEval-164 subset")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    tasks = source_tasks(args.source_root)
    if len(tasks) != 164:
        raise RuntimeError(f"expected 164 CARVE HumanEval tasks, found {len(tasks)}")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    config.max_tokens, config.timeout, config.retries_per_url = 1024, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model, resume=args.resume, retry_errors=args.retry_errors), indent=2))


if __name__ == "__main__":
    main()
