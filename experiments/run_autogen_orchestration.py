from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_gsm8k_single_cot import source_tasks as gsm8k_tasks
from experiments.run_humaneval_single_cot import humaneval_tests, source_tasks as humaneval_tasks
from experiments.run_mbpp_single_cot import source_tasks as mbpp_tasks
from experiments.run_openqa_single_cot import source_cases as openqa_cases, source_tasks as openqa_tasks
from experiments.run_spider_single_cot import source_cases as spider_cases


def debate_prompts(dataset: str, question: str, context: str, candidate_a: str, candidate_b: str, critique: str) -> list[str]:
    output_rule = "For OpenQA, output only the shortest answer span, never a sentence or explanation." if dataset == "openqa" else "Return only the answer required by the task."
    return [
        f"You are the AutoGen solver for {dataset}. Solve the task independently using only the supplied context.\n\nTask:\n{question}\n\nContext:\n{context}\n\nReturn the best answer only.",
        f"You are the AutoGen critic for {dataset}. Compare these two candidate answers against the task and context. Do not execute tools and do not use outside information.\n\nTask:\n{question}\n\nContext:\n{context}\n\nCandidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\nReturn a concise critique and recommendation.",
        f"You are the AutoGen finalizer for {dataset}. Produce the final answer using the task, context, candidates, and critique. {output_rule} Do not add Markdown or explanation.\n\nTask:\n{question}\n\nContext:\n{context}\n\nCandidate A:\n{candidate_a}\n\nCandidate B:\n{candidate_b}\n\nCritique:\n{critique}",
    ]


def extract_text(result: Any) -> str:
    messages = getattr(result, "messages", None) or []
    content = getattr(messages[-1], "content", "") if messages else ""
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    return str(content).strip()


def telemetry_from_results(results: list[Any], started_at: float, finished_at: float) -> dict[str, Any]:
    """Collect provider usage when AutoGen exposes it and always record wall-clock time."""
    input_tokens = 0
    output_tokens = 0
    usage_records = 0
    tool_calls = 0
    for result in results:
        usage = getattr(result, "usage", None)
        messages = getattr(result, "messages", None) or []
        if usage is None and messages:
            usage = getattr(messages[-1], "models_usage", None)
        if usage is not None:
            input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
            usage_records += 1
        for message in messages:
            content = getattr(message, "content", None)
            if isinstance(content, list):
                tool_calls += sum(1 for item in content if "tool" in type(item).__name__.lower())
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "usage_records": usage_records,
        "tool_calls": tool_calls,
        "latency_ms": round((finished_at - started_at) * 1000, 3),
        "telemetry_source": "autogen_result_usage_and_wall_clock",
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_tasks(dataset: str, source_run: Path, spider_root: Path | None, openqa_data: Path | None) -> list[dict]:
    if dataset == "gsm8k":
        return [{"task_id": str(x["task_id"]), "question": str(x["prompt"]), "context": "", "reference": x["reference"], "kind": dataset} for x in gsm8k_tasks(source_run)]
    if dataset == "mbpp":
        return [{"task_id": str(x["task_id"]), "question": str(x["prompt"]), "context": str(x["tests"]), "reference": x["tests"], "kind": dataset} for x in mbpp_tasks(source_run)]
    if dataset == "humaneval":
        return [{"task_id": str(x["task_id"]), "question": str(x["prompt"]), "context": humaneval_tests(x), "reference": humaneval_tests(x), "kind": dataset} for x in humaneval_tasks(source_run)]
    if dataset == "spider":
        if spider_root is None:
            raise ValueError("--spider-root is required for Spider")
        return [{"task_id": c.case_id, "question": c.question, "context": c.schema, "reference": c, "kind": dataset} for c in spider_cases(source_run, spider_root)]
    if dataset == "openqa":
        if openqa_data is None:
            raise ValueError("--openqa-data is required for OpenQA")
        cases = openqa_cases(openqa_data)
        return [{"task_id": str(x["task_id"]), "question": str(x["question"]), "context": str(x["evidence"]), "reference": tuple(x["aliases"]), "kind": dataset} for x in openqa_tasks(source_run, cases)]
    raise ValueError(dataset)


def _verify(task: dict, answer: str):
    if task["kind"] == "gsm8k":
        return MathVerifier().verify(answer, str(task["reference"]))
    if task["kind"] in {"mbpp", "humaneval"}:
        return CodeVerifier().verify(answer, str(task["reference"]))
    if task["kind"] == "spider":
        return SpiderVerifier().verify(answer, task["reference"])
    return OpenQAExactMatchVerifier().verify(answer, task["reference"])


async def _debate(client, dataset: str, task: dict) -> tuple[str, str, str]:
    started_at = time.perf_counter()
    solver = AssistantAgent("solver", model_client=client, system_message="Return a candidate solution for the assigned task.")
    solver_result_a = await solver.run(task=debate_prompts(dataset, task["question"], task["context"], "", "", "")[0])
    candidate_a = extract_text(solver_result_a)
    solver_result_b = await solver.run(task=debate_prompts(dataset, task["question"], task["context"], "", "", "")[0])
    candidate_b = extract_text(solver_result_b)
    critic = AssistantAgent("critic", model_client=client, system_message="Review candidate solutions and recommend one.")
    critique_result = await critic.run(task=debate_prompts(dataset, task["question"], task["context"], candidate_a, candidate_b, "")[1])
    critique = extract_text(critique_result)
    finalizer = AssistantAgent("finalizer", model_client=client, system_message="Return only the final task answer.")
    answer_result = await finalizer.run(task=debate_prompts(dataset, task["question"], task["context"], candidate_a, candidate_b, critique)[2])
    answer = extract_text(answer_result)
    finished_at = time.perf_counter()
    return candidate_a, candidate_b, answer, telemetry_from_results([solver_result_a, solver_result_b, critique_result, answer_result], started_at, finished_at)


def run(tasks: list[dict], output: Path, dataset: str, model: str, base_url: str, api_key: str, limit: int | None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    completed = {row["task_id"] for row in _jsonl(output / "traces.jsonl")}
    selected = tasks[:limit] if limit is not None else tasks
    client = OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/") + "/v1",
        model_info={"vision": False, "function_calling": False, "json_output": False, "family": "unknown"},
    )
    async def loop():
        for task in selected:
            if task["task_id"] in completed:
                continue
            try:
                a, b, answer, telemetry = await _debate(client, dataset, task)
                score = _verify(task, answer)
                _append(output / "traces.jsonl", {"task_id": task["task_id"], "dataset": dataset, "method": "autogen_orchestration", "model": model, "candidate_a": a, "candidate_b": b, "final_answer": answer, "success": bool(score.success), "verifier_score": score.score, "verifier_details": score.details, "api_calls": 4, **telemetry})
            except Exception as exc:
                _append(output / "errors.jsonl", {"task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"})
        await client.close()
    asyncio.run(loop())
    rows = _jsonl(output / "traces.jsonl")
    summary = {"dataset": dataset, "method": "autogen_orchestration", "model": model, "source_tasks": len(tasks), "completed_tasks": len(rows), "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0, "api_calls": sum(int(row.get("api_calls", 0)) for row in rows), "input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows), "output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows), "total_tokens": sum(int(row.get("total_tokens", 0)) for row in rows), "tool_calls": sum(int(row.get("tool_calls", 0)) for row in rows), "latency_ms": round(sum(float(row.get("latency_ms", 0.0)) for row in rows), 3), "telemetry_records": sum(int(row.get("usage_records", 0)) for row in rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "humaneval", "mbpp", "spider", "openqa"], required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-data", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="resume the output directory and skip completed task IDs")
    parser.add_argument("--model", default=os.environ.get("CARVE_MODEL", "glm-5.2"))
    parser.add_argument("--base-url", default=os.environ.get("CARVE_BASE_URLS", "https://api.openai.com/v1"))
    args = parser.parse_args()
    api_key = os.environ["CARVE_API_KEY"]
    tasks = load_tasks(args.dataset, args.source_run, args.spider_root, args.openqa_data)
    print(json.dumps(run(tasks, args.output, args.dataset, args.model, args.base_url, api_key, args.limit), indent=2))


if __name__ == "__main__":
    main()
