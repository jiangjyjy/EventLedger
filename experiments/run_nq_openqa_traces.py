from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.agents.nq_openqa_runner import NQOpenQARunner
from carve.datasets.nq_openqa import load_nq_openqa_jsonl


def configure_openqa_api(config: APIClientConfig, *, max_tokens: int) -> APIClientConfig:
    """Use an explicit OpenQA response budget without mutating shared config."""
    if max_tokens < 64:
        raise ValueError("OpenQA max_tokens must be at least 64 to leave room for reasoning and final output")
    return replace(config, max_tokens=max_tokens, retries_per_url=max(2, config.retries_per_url))


def run_cases(cases: list[Any], output: Path, client: Any, *, seed: int = 0, model: str = "glm-5.2", reader_context_mode: str = "split", top_k: int = 8, resume: bool = False, retry_abstained: bool = False) -> dict[str, int]:
    if output.exists() and not resume:
        raise ValueError("OpenQA factual output is create-once; pass resume=True")
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    errors_path = output.with_name("errors.jsonl")
    failed = set()
    if output.exists():
        completed = {json.loads(line)["task_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
    if errors_path.exists():
        failed = {json.loads(line)["task_id"] for line in errors_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    pending = [case for case in cases if case.task_id not in completed and (retry_abstained or case.task_id not in failed)]
    runner = NQOpenQARunner(client, reader_context_mode=reader_context_mode, top_k=top_k)
    with output.open("a", encoding="utf-8") as handle:
        for case in pending:
            try:
                trace = runner.run(case, seed=seed, model=model, split="factual")
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            except (RuntimeError, TimeoutError) as error:
                with errors_path.open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps({"task_id": case.task_id, "abstained": True, "error": f"{type(error).__name__}: {error}"}) + "\n")
                    errors.flush(); os.fsync(errors.fileno())
    return {"input_rows": len(cases), "trace_rows": len(pending), "api_calls": 3 * len(pending), "failed_rows": len(failed)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resumable NQ-open DPR factual traces")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--reader-context-mode", choices=("split", "full_top_k"), default="split")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-abstained", action="store_true")
    args = parser.parse_args()
    cases = load_nq_openqa_jsonl(args.input, limit=args.limit, offset=args.offset)
    config = configure_openqa_api(APIClientConfig.from_env(), max_tokens=args.max_tokens)
    print(json.dumps(run_cases(cases, args.output, OpenAICompatibleClient(config), seed=args.seed, model=args.model, reader_context_mode=args.reader_context_mode, top_k=args.top_k, resume=args.resume, retry_abstained=args.retry_abstained), sort_keys=True))


if __name__ == "__main__":
    main()
