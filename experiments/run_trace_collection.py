from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

from carve.agents import APIClientConfig, MultiAgentRunner, OpenAICompatibleClient, RunnerConfig, get_role_specs
from carve.datasets import load_gsm8k_sample, load_humaneval_sample, load_mbpp_sample, load_openqa_sample, load_swebench_sample
from carve.schemas import Trace
from carve.schemas import Trace
from carve.verifiers import CodeVerifier, MathVerifier, RubricVerifier, SWEBenchVerifier

LOADERS = {
    "humaneval": load_humaneval_sample,
    "mbpp": load_mbpp_sample,
    "gsm8k": load_gsm8k_sample,
    "swebench_lite": load_swebench_sample,
    "research_synthesis_qa": load_openqa_sample,
}


def score_trace_for_task(trace, task):
    if task.dataset in {"humaneval", "mbpp"}:
        return CodeVerifier().verify(trace.final_answer, task.tests)
    if task.dataset == "gsm8k":
        return MathVerifier().verify(trace.final_answer, task.reference)
    if task.dataset == "research_synthesis_qa":
        return RubricVerifier().verify(trace.final_answer)
    if task.dataset == "swebench_lite":
        return SWEBenchVerifier().verify(trace.final_answer, task.tests, repo_path=task.metadata.get("repo_path"), base_commit=task.metadata.get("base_commit"), setup_patch=task.metadata.get("test_patch"))
    return RubricVerifier().verify(trace.final_answer, task.reference or task.prompt)


def load_trace_records(path: Path) -> list[Trace]:
    if not path.exists():
        return []
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_trace_record(path: Path, trace: Trace) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gsm8k", choices=LOADERS)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--api", action="store_true", help="Use CARVE_API_KEY and OpenAI-compatible API endpoints")
    parser.add_argument("--planner-mode", choices=["static", "dynamic"], default="static")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0)
    parser.add_argument("--token-cost", type=float, default=0.00001)
    parser.add_argument("--prompt-version", default="default", choices=["default", "code_v2", "mbpp_v1", "gsm8k_stable_v1", "swebench_v1"])
    args = parser.parse_args()
    if args.dataset == "swebench_lite":
        if args.prompt_version == "default":
            args.prompt_version = "swebench_v1"
        elif args.prompt_version != "swebench_v1":
            parser.error("swebench_lite requires --prompt-version swebench_v1")

    out_dir = Path("artifacts/runs") / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAICompatibleClient(APIClientConfig.from_env()) if args.api else None
    model_name = os.environ.get("CARVE_MODEL", "glm-5.1") if args.api else "deterministic"
    runner = MultiAgentRunner(client=client, roles=get_role_specs(args.prompt_version))
    path = out_dir / "traces.jsonl"
    traces = load_trace_records(path)
    completed_task_ids = {trace.task_id for trace in traces}
    loader = LOADERS[args.dataset]
    try:
        tasks = loader(args.limit, offset=args.offset)
    except TypeError:
        tasks = loader(args.limit)
        if args.offset:
            tasks = tasks[args.offset : args.offset + args.limit]
    for task in tasks:
        if task.task_id in completed_task_ids:
            continue
        trace = runner.run(
            task,
            RunnerConfig(
                seed=args.seed,
                model=model_name,
                split="smoke",
                planner_mode=args.planner_mode,
                max_retries=args.max_retries,
                max_cost=args.max_cost,
                early_stop_threshold=args.early_stop_threshold,
                token_cost=args.token_cost,
                prompt_version=args.prompt_version,
            ),
        )
        trace.manifest["task"] = {
            "task_id": task.task_id,
            "dataset": task.dataset,
            "prompt": task.prompt,
            "reference": task.reference,
            "tests": task.tests,
            "metadata": task.metadata,
            "repo_path": task.metadata.get("repo_path"),
            "base_commit": task.metadata.get("base_commit"),
            "instance_id": task.metadata.get("instance_id", task.task_id),
        }
        score = score_trace_for_task(trace, task)
        trace.verifier_score = score.score
        trace.success = score.success
        append_trace_record(path, trace)
        traces.append(trace)
        completed_task_ids.add(task.task_id)
    manifest = {
        "run_id": args.run_id,
        "dataset": args.dataset,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "planner_mode": args.planner_mode,
        "max_retries": args.max_retries,
        "max_cost": args.max_cost,
        "early_stop_threshold": args.early_stop_threshold,
        "token_cost": args.token_cost,
        "prompt_version": args.prompt_version,
        "output": str(path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
