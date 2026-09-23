from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from experiments.run_openqa_single_cot import _read_jsonl, source_cases, source_tasks


def solver_prompt(question: str, evidence: str, role: str) -> str:
    return (
        f"You are {role} in a two-solver debate. Answer using only the retrieved evidence. "
        "State a candidate answer and one brief evidence-based reason. Do not use outside knowledge.\n\n"
        f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}\n\n"
        "Return exactly:\nCandidate answer: <short answer>\nReason: <brief reason>"
    )


def critic_prompt(question: str, evidence: str, candidate_a: str, candidate_b: str) -> str:
    return (
        "You are a debate critic. Compare the two candidate answers against the retrieved evidence. "
        "Identify which answer is better supported, or explain why neither is supported.\n\n"
        f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}\n\n"
        f"Candidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\n"
        "Return a concise evidence-grounded critique."
    )


def judge_prompt(question: str, evidence: str, candidate_a: str, candidate_b: str, critique: str) -> str:
    return (
        "You are the final judge in a multi-agent debate. Use only the retrieved evidence and the critique "
        "to resolve the question. Return only the shortest answer span, with no explanation, citation, label, or Markdown.\n\n"
        f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}\n\n"
        f"Candidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\nCritique:\n{critique}"
    )


def _task_ids(path: Path) -> set[str]:
    return {str(row["task_id"]) for row in _read_jsonl(path)} if path.exists() else set()


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _call(client: OpenAICompatibleClient, role: str, prompt: str, seed: int) -> tuple[str, dict]:
    content = client.complete(role, prompt, seed).strip()
    return content, client.last_completion_telemetry()


def run(tasks: list[dict], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [task for task in tasks if task["task_id"] not in completed and (retry_errors or task["task_id"] not in errors)]
    verifier = OpenQAExactMatchVerifier()
    for index, task in enumerate(pending):
        try:
            task_seed = seed + index * 4
            candidate_a, telemetry_a = _call(client, "mad_solver_a", solver_prompt(task["question"], task["evidence"], "solver A"), task_seed)
            candidate_b, telemetry_b = _call(client, "mad_solver_b", solver_prompt(task["question"], task["evidence"], "solver B"), task_seed + 1)
            critique, telemetry_c = _call(client, "mad_critic", critic_prompt(task["question"], task["evidence"], candidate_a, candidate_b), task_seed + 2)
            answer, telemetry_j = _call(client, "mad_judge", judge_prompt(task["question"], task["evidence"], candidate_a, candidate_b, critique), task_seed + 3)
            score = verifier.verify(answer, task["aliases"])
            telemetry = {"api_calls": 4, "stages": {"solver_a": telemetry_a, "solver_b": telemetry_b, "critic": telemetry_c, "judge": telemetry_j}}
            _append(output / "traces.jsonl", {
                "task_id": task["task_id"], "dataset": "natural_questions_open_dpr_dev", "method": "multi_agent_debate_mad4", "model": model,
                "candidate_a": candidate_a, "candidate_b": candidate_b, "critique": critique, "final_answer": answer,
                "success": score.success, "verifier_score": score.score, "verifier_details": score.details,
                "telemetry": telemetry,
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    rows = _read_jsonl(output / "traces.jsonl") if (output / "traces.jsonl").exists() else []
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
    parser = argparse.ArgumentParser(description="MAD-4 baseline on the CARVE NQ-Open-100 subset")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    tasks = source_tasks(args.source_run, source_cases(args.data))
    if len(tasks) != 100:
        raise RuntimeError(f"expected 100 CARVE NQ-Open tasks, found {len(tasks)}")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    config.max_tokens, config.timeout, config.retries_per_url = 128, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model, resume=args.resume, retry_errors=args.retry_errors), indent=2))


if __name__ == "__main__":
    main()
