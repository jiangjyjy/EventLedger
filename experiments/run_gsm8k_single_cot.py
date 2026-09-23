from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.schemas import Trace
from carve.verifiers.math import MathVerifier


def source_tasks(source_run: Path) -> list[dict]:
    tasks: dict[str, dict] = {}
    for line in (source_run / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        trace = Trace.from_dict(json.loads(line))
        task = trace.manifest.get("task")
        if isinstance(task, dict) and task.get("task_id") and task.get("prompt") and task.get("reference"):
            tasks.setdefault(str(task["task_id"]), task)
    return [tasks[task_id] for task_id in sorted(tasks)]


def gsm8k_prompt(task: dict) -> str:
    return (
        "Solve the grade-school math problem. Show concise reasoning, then put the final numeric answer "
        "on its own line exactly as `#### <number>`.\n\nProblem:\n"
        + str(task["prompt"])
    )


def _task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(json.loads(line)["task_id"]) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(tasks: list[dict], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed = _task_ids(output / "traces.jsonl")
    errors = _task_ids(output / "errors.jsonl")
    pending = [task for task in tasks if str(task["task_id"]) not in completed and (retry_errors or str(task["task_id"]) not in errors)]
    verifier = MathVerifier()
    for index, task in enumerate(pending):
        try:
            answer = client.complete("single_agent_cot", gsm8k_prompt(task), seed + index).strip()
            score = verifier.verify(answer, str(task["reference"]))
            _append(output / "traces.jsonl", {
                "task_id": task["task_id"], "dataset": "gsm8k", "method": "single_agent_cot", "model": model,
                "final_answer": answer, "success": score.success, "verifier_score": score.score,
                "verifier_details": score.details, "stderr": score.stderr, "telemetry": client.last_completion_telemetry(),
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    rows = [json.loads(line) for line in (output / "traces.jsonl").read_text(encoding="utf-8").splitlines()] if (output / "traces.jsonl").exists() else []
    summary = {
        "method": "single_agent_cot", "model": model, "source_tasks": len(tasks), "completed_tasks": len(rows),
        "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent CoT baseline on the CARVE GSM8K-900 subset")
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
