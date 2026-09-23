from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.schemas import Trace
from carve.verifiers.code import CodeVerifier


def source_tasks(source_root: Path) -> list[dict]:
    records: dict[str, dict] = {}
    for path in sorted(source_root.glob("humaneval_glm52_code_v2_full_k3_top8_ops3_task*/traces.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            trace = Trace.from_dict(json.loads(line))
            task = trace.manifest.get("task")
            if isinstance(task, dict) and task.get("task_id") and task.get("prompt") and task.get("tests"):
                records.setdefault(str(task["task_id"]), task)
    return list(records.values())


def prompt_for(task: dict) -> str:
    return "Write complete Python code only, with no Markdown or explanation.\n\n" + str(task["prompt"])


def humaneval_tests(task: dict) -> str:
    """Attach the official HumanEval check call when a task stores it separately."""
    tests = str(task["tests"])
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    entry_point = task.get("entry_point") or metadata.get("entry_point")
    if entry_point and "def check(" in tests:
        return f"{tests}\ncheck({entry_point})\n"
    return tests


def run(tasks: list[dict], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    successes = 0
    with (output / "traces.jsonl").open("a", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            try:
                answer = client.complete("single_agent_cot", prompt_for(task), seed + index).strip()
                score = CodeVerifier().verify(answer, humaneval_tests(task))
                row = {"task_id": task["task_id"], "dataset": "humaneval", "method": "single_agent_cot", "model": model, "final_answer": answer, "success": score.success, "verifier_score": score.score, "verifier_details": score.details, "stderr": score.stderr, "telemetry": client.last_completion_telemetry()}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
                successes += int(score.success)
            except (RuntimeError, TimeoutError) as exc:
                with (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}) + "\n"); errors.flush(); os.fsync(errors.fileno())
    summary = {"method": "single_agent_cot", "model": model, "tasks": len(tasks), "successes": successes, "success_rate": successes / len(tasks) if tasks else 0.0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args()
    tasks = source_tasks(args.source_root)
    if len(tasks) != 164:
        raise RuntimeError(f"expected 164 CARVE HumanEval tasks, found {len(tasks)}")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    config.max_tokens, config.timeout, config.retries_per_url = 1024, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model), indent=2))


if __name__ == "__main__":
    main()
