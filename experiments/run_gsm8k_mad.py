from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.math import MathVerifier
from experiments.run_gsm8k_single_cot import source_tasks


def _problem(task):
    return f"Problem:\n{task['prompt']}"


def solver_prompt(task, role):
    return (f"You are {role} in a two-solver math debate. Solve independently. Show concise reasoning, then put the final numeric answer on its own line exactly as `#### <number>`.\n\n" + _problem(task))


def critic_prompt(task, candidate_a, candidate_b):
    return ("You are a math debate critic. Compare both solutions against the problem. Check arithmetic, units, and reasoning. State which candidate is better justified.\n\n" + _problem(task) + "\n\nCandidate A:\n" + candidate_a + "\n\nCandidate B:\n" + candidate_b)


def judge_prompt(task, candidate_a, candidate_b, critique):
    return ("You are the final judge in a multi-agent math debate. Resolve the problem using the candidates and critique. Return concise reasoning followed by the final numeric answer on its own line exactly as `#### <number>`.\n\n" + _problem(task) + "\n\nCandidate A:\n" + candidate_a + "\n\nCandidate B:\n" + candidate_b + "\n\nCritique:\n" + critique)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _task_ids(path):
    return {str(row["task_id"]) for row in _read_jsonl(path)}


def _append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _call(client, role, prompt, seed):
    return client.complete(role, prompt, seed).strip(), client.last_completion_telemetry()


def run(tasks, output, client, *, seed, model, resume, retry_errors):
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [t for t in tasks if str(t["task_id"]) not in completed and (retry_errors or str(t["task_id"]) not in errors)]
    verifier = MathVerifier()
    for index, task in enumerate(pending):
        try:
            task_seed = seed + 4 * index
            a, ta = _call(client, "mad_math_solver_a", solver_prompt(task, "solver A"), task_seed)
            b, tb = _call(client, "mad_math_solver_b", solver_prompt(task, "solver B"), task_seed + 1)
            critique, tc = _call(client, "mad_math_critic", critic_prompt(task, a, b), task_seed + 2)
            answer, tj = _call(client, "mad_math_judge", judge_prompt(task, a, b, critique), task_seed + 3)
            score = verifier.verify(answer, str(task["reference"]))
            _append(output / "traces.jsonl", {"task_id": task["task_id"], "dataset": "gsm8k", "method": "multi_agent_debate_mad4", "model": model, "candidate_a": a, "candidate_b": b, "critique": critique, "final_answer": answer, "success": score.success, "verifier_score": score.score, "verifier_details": score.details, "telemetry": {"api_calls": 4, "stages": {"solver_a": ta, "solver_b": tb, "critic": tc, "judge": tj}}})
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    rows = _read_jsonl(output / "traces.jsonl")
    summary = {"method": "multi_agent_debate_mad4", "model": model, "source_tasks": len(tasks), "completed_tasks": len(rows), "successes": sum(int(r["success"]) for r in rows), "success_rate": sum(int(r["success"]) for r in rows) / len(rows) if rows else 0.0, "api_calls": sum(int((r.get("telemetry") or {}).get("api_calls", 0)) for r in rows), "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl"))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="MAD-4 baseline on CARVE GSM8K-900")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    tasks = source_tasks(args.source_run)
    if len(tasks) != 900:
        raise RuntimeError(f"expected 900 CARVE GSM8K tasks, found {len(tasks)}")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    config.max_tokens, config.timeout, config.retries_per_url = 1024, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model, resume=args.resume, retry_errors=args.retry_errors), indent=2))


if __name__ == "__main__":
    main()
