from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.spider_runner import SpiderRunner, SpiderRunnerConfig
from carve.datasets.spider import SpiderCase
from carve.schemas import Task


def run_cases(cases: list[SpiderCase], output: Path, client: object, *, seed: int = 0, model: str = "glm-5.2") -> dict[str, int]:
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    runner = SpiderRunner(client)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for case in cases:
            trace = runner.run(Task(case.case_id, "spider", case.question), case, SpiderRunnerConfig(seed=seed, model=model))
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"input_rows": len(cases), "trace_rows": len(cases), "api_calls": 4 * len(cases)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="glm-5.2")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output already exists")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[args.offset:args.offset + args.limit]
    config = APIClientConfig.from_env()
    config.model = args.model
    cases = [SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in selected]
    print(json.dumps(run_cases(cases, args.output, OpenAICompatibleClient(config), seed=args.seed, model=args.model), sort_keys=True))


if __name__ == "__main__":
    main()
