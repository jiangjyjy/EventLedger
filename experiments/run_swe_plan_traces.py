from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.swe_plan_runner import PlanRunnerConfig, SWEPlanRunner
from carve.schemas import Task
from carve.swe_derived.plan_only import RepairSpec


def _spec(row: dict[str, Any]) -> RepairSpec:
    value = row["spec"]
    return RepairSpec(
        case_id=row["task_id"],
        files=tuple(value["files"]),
        symbols=tuple(value["symbols"]),
        diagnosis_terms=tuple(value["diagnosis_terms"]),
        change_terms=tuple(value["change_terms"]),
        test_terms=tuple(value["test_terms"]),
    )


def run_rows(rows: Iterable[dict[str, Any]], output: Path, client: Any, *, seed: int = 0, model: str = "glm-5.2") -> dict[str, int]:
    if output.exists() or output.is_symlink():
        raise ValueError("output already exists")
    materialized = list(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    runner = SWEPlanRunner(client)
    api_calls = 0
    with output.open("x", encoding="utf-8") as handle:
        for row in materialized:
            task = Task(row["task_id"], "swe_plan", row["prompt"])
            trace = runner.run(task, _spec(row), PlanRunnerConfig(seed=seed, model=model))
            api_calls += int(trace.manifest["telemetry"]["api_calls"])
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return {"input_rows": len(materialized), "trace_rows": len(materialized), "api_calls": api_calls}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="glm-5.2")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    config = APIClientConfig.from_env()
    config.model = args.model
    print(json.dumps(run_rows(rows, args.output, OpenAICompatibleClient(config), seed=args.seed, model=args.model), sort_keys=True))


if __name__ == "__main__":
    main()
