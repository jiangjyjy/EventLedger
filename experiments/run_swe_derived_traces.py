from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.swe_derived_runner import RunnerConfig, SWEDerivedRunner, task_from_case
from carve.datasets.swe_derived import load_swe_derived_jsonl


def append_trace(path: Path, trace: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--api-retries", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists")
    client = OpenAICompatibleClient(APIClientConfig.from_env())
    runner = SWEDerivedRunner(client)
    cases = load_swe_derived_jsonl(args.input, limit=args.limit, offset=args.offset)
    for case in cases:
        trace = runner.run(
            task_from_case(case),
            case,
            RunnerConfig(seed=args.seed, model=args.model, api_retries=args.api_retries),
        )
        append_trace(args.output, trace)


if __name__ == "__main__":
    main()
