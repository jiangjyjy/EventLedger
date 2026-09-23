from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.openqa import OpenQAExactMatchVerifier


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_cases(data_path: Path) -> dict[str, dict]:
    cases = {str(row["task_id"]): row for row in _read_jsonl(data_path)}
    return cases


def source_tasks(source_run: Path, cases: dict[str, dict]) -> list[dict]:
    tasks: list[dict] = []
    seen: set[str] = set()
    for trace in _read_jsonl(source_run / "traces.jsonl"):
        task_id = str(trace["task_id"])
        if task_id in seen:
            raise RuntimeError(f"duplicate factual trace for {task_id}")
        events = trace.get("events") or []
        e1 = next((event for event in events if event.get("event_id") == "e1"), None)
        if not e1 or not isinstance(e1.get("content"), str) or not e1["content"].strip():
            raise RuntimeError(f"missing DPR e1 evidence for {task_id}")
        case = cases.get(task_id)
        if case is None:
            raise RuntimeError(f"source trace task absent from NQ-Open data: {task_id}")
        answers = tuple(str(answer) for answer in (case.get("reference") or {}).get("answers", []))
        if not answers:
            raise RuntimeError(f"NQ-Open case has no answer aliases: {task_id}")
        tasks.append({"task_id": task_id, "question": str(case["prompt"]), "aliases": answers, "evidence": e1["content"]})
        seen.add(task_id)
    if set(cases) != seen:
        raise RuntimeError(f"source/data task mismatch: {len(tasks)} factual traces vs {len(cases)} cases")
    return tasks


def openqa_prompt(question: str, evidence: str) -> str:
    return (
        "Answer the question using only the retrieved evidence. Return only the shortest answer span, "
        "with no explanation, citation, label, or Markdown.\n\nQuestion:\n"
        + question
        + "\n\nRetrieved evidence:\n"
        + evidence
    )


def _task_ids(path: Path) -> set[str]:
    return {str(row["task_id"]) for row in _read_jsonl(path)} if path.exists() else set()


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run(tasks: list[dict], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [task for task in tasks if task["task_id"] not in completed and (retry_errors or task["task_id"] not in errors)]
    verifier = OpenQAExactMatchVerifier()
    for index, task in enumerate(pending):
        try:
            answer = client.complete("single_agent_cot", openqa_prompt(task["question"], task["evidence"]), seed + index).strip()
            score = verifier.verify(answer, task["aliases"])
            _append(output / "traces.jsonl", {
                "task_id": task["task_id"], "dataset": "natural_questions_open_dpr_dev", "method": "single_agent_cot", "model": model,
                "final_answer": answer, "success": score.success, "verifier_score": score.score,
                "verifier_details": score.details, "stderr": score.stderr, "telemetry": client.last_completion_telemetry(),
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    rows = _read_jsonl(output / "traces.jsonl") if (output / "traces.jsonl").exists() else []
    summary = {
        "method": "single_agent_cot", "model": model, "source_tasks": len(tasks), "completed_tasks": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent CoT baseline on the CARVE NQ-Open-100 subset")
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
    config.max_tokens, config.timeout, config.retries_per_url = 64, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), seed=args.seed, model=config.model, resume=args.resume, retry_errors=args.retry_errors), indent=2))


if __name__ == "__main__":
    main()
