from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.spider import SpiderCase, load_spider_dev
from carve.student_lora.data import load_traces
from carve.verifiers.spider import SpiderVerifier


def source_cases(source_run: Path, spider_root: Path) -> list[SpiderCase]:
    task_ids = [trace.task_id for trace in load_traces(source_run / "traces.jsonl")]
    if len(task_ids) != 100 or len(set(task_ids)) != 100:
        raise RuntimeError(f"expected 100 unique Spider source tasks, found {len(task_ids)}")
    cases = []
    for task_id in task_ids:
        _, index_text = task_id.rsplit("-dev-", 1)
        case = load_spider_dev(spider_root, limit=1, offset=int(index_text))[0]
        if case.case_id != task_id:
            raise RuntimeError(f"Spider task mismatch: {task_id} != {case.case_id}")
        cases.append(case)
    return cases


def spider_prompt(question: str, schema: str) -> str:
    return (
        "Write one executable SQLite SELECT or WITH query that answers the question using the schema. "
        "Return SQL only, with no Markdown or explanation.\n\nQuestion:\n"
        + question
        + "\n\nSQLite schema:\n"
        + schema
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


def run(cases: list[SpiderCase], output: Path, client: OpenAICompatibleClient, *, seed: int, model: str, resume: bool, retry_errors: bool) -> dict:
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; pass --resume to continue: {output}")
    output.mkdir(parents=True, exist_ok=True)
    completed, errors = _task_ids(output / "traces.jsonl"), _task_ids(output / "errors.jsonl")
    pending = [case for case in cases if case.case_id not in completed and (retry_errors or case.case_id not in errors)]
    verifier = SpiderVerifier()
    for index, case in enumerate(pending):
        try:
            answer = client.complete("single_agent_cot", spider_prompt(case.question, case.schema), seed + index).strip()
            score = verifier.verify(answer, case)
            _append(output / "traces.jsonl", {
                "task_id": case.case_id, "dataset": "spider", "method": "single_agent_cot", "model": model,
                "final_answer": answer, "success": score.success, "verifier_score": score.score,
                "verifier_details": score.details, "stderr": score.stderr, "telemetry": client.last_completion_telemetry(),
            })
        except (RuntimeError, TimeoutError) as exc:
            _append(output / "errors.jsonl", {"task_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
    rows = [json.loads(line) for line in (output / "traces.jsonl").read_text(encoding="utf-8").splitlines()] if (output / "traces.jsonl").exists() else []
    summary = {
        "method": "single_agent_cot", "model": model, "source_tasks": len(cases), "completed_tasks": len(rows),
        "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "pending_errors": len(_task_ids(output / "errors.jsonl") - _task_ids(output / "traces.jsonl")),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent CoT baseline on the CARVE Spider-100 subset")
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
