from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_dag_runner import SpiderDAGRunner, SpiderDAGRunnerConfig
from carve.datasets.spider import SpiderCase
from carve.schemas import Task


def run_cases(cases: list[SpiderCase], output: Path, client: object, *, seed: int = 0, model: str = "glm-5.2", resume: bool = False) -> dict[str, int]:
    if output.exists() and not resume:
        raise ValueError("output already exists; pass resume=True")
    runner = SpiderDAGRunner(client)
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output.exists():
        existing = {json.loads(line)["task_id"]: line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [case for case in cases if case.case_id not in existing]
    with output.open("a", encoding="utf-8") as handle:
        for case in pending:
            trace = runner.run(Task(case.case_id, "spider", case.question), case, SpiderDAGRunnerConfig(seed=seed, model=model))
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"input_rows": len(cases), "trace_rows": len(pending), "api_calls": 4 * len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect independent-branch Spider DAG factual traces")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[args.offset:args.offset + args.limit]
    cases = [SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in selected]
    config = APIClientConfig.from_env()
    config.model = args.model
    print(json.dumps(run_cases(cases, args.output, OpenAICompatibleClient(config), seed=args.seed, model=args.model, resume=args.resume), sort_keys=True))


if __name__ == "__main__":
    main()
