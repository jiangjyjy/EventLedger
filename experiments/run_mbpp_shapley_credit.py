"""Independent MBPP Shapley-credit baseline over saved factual traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from carve.datasets.spider import load_spider_dev
from carve.datasets.nq_openqa import load_nq_openqa_jsonl


def terminal_answer(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "aggregate":
            return str(event.get("content", ""))
    for event in reversed(events):
        if event.get("type") in {"revise", "msg"}:
            return str(event.get("content", ""))
    return str(events[-1].get("content", "")) if events else ""


def coalition_value(events: list[dict], reference, coalition: set[int], verifier) -> float:
    selected = [event for index, event in enumerate(events) if index in coalition]
    answer = terminal_answer(selected)
    return 1.0 if verifier.verify(answer, reference).success else 0.0


def shapley_values(events: list[dict], reference, verifier, permutations: int, seed: int) -> list[float]:
    values = [0.0] * len(events)
    rng = random.Random(seed)
    for _ in range(permutations):
        order = list(range(len(events)))
        rng.shuffle(order)
        coalition: set[int] = set()
        before = 0.0
        for index in order:
            coalition.add(index)
            after = coalition_value(events, reference, coalition, verifier)
            values[index] += after - before
            before = after
    return [value / max(1, permutations) for value in values]


def run(source: Path, output: Path, dataset: str, spider_root: Path | None, openqa_data: Path | None, limit: int | None, permutations: int, seed: int) -> dict:
    traces = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        traces = traces[:limit]
    dataset_key = dataset.lower()
    verifier = CodeVerifier() if dataset_key in {"mbpp", "humaneval"} else MathVerifier() if dataset_key == "gsm8k" else SpiderVerifier() if dataset_key == "spider" else OpenQAExactMatchVerifier()
    spider_cases = {}
    openqa_cases = {}
    if dataset_key == "spider":
        if spider_root is None:
            raise ValueError("--spider-root is required for Spider")
        for trace in traces:
            _, suffix = trace["task_id"].rsplit("-dev-", 1)
            case = load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
            if case.case_id != trace["task_id"]:
                raise ValueError(f"Spider task mismatch: {trace['task_id']} != {case.case_id}")
            spider_cases[trace["task_id"]] = case
    if dataset_key == "openqa":
        if openqa_data is None:
            raise ValueError("--openqa-data is required for OpenQA")
        openqa_cases = {case.task_id: case for case in load_nq_openqa_jsonl(openqa_data)}
    rows = []
    task_records = []
    for trace in traces:
        if dataset_key in {"mbpp", "humaneval"}:
            reference = str(trace["manifest"]["task"]["tests"])
        elif dataset_key == "gsm8k":
            reference = str(trace["manifest"]["task"].get("reference", trace.get("final_answer", "")))
        elif dataset_key == "spider":
            reference = spider_cases[trace["task_id"]]
        else:
            reference = openqa_cases[trace["task_id"]].answers
        events = trace.get("events", [])
        values = shapley_values(events, reference, verifier, permutations, seed + len(task_records))
        keep = {index for index, value in enumerate(values) if value > 0.0}
        answer = terminal_answer([event for index, event in enumerate(events) if index in keep])
        result = verifier.verify(answer, reference)
        task_records.append({"task_id": trace["task_id"], "success": bool(result.success), "events": len(events), "events_kept": len(keep), "answer": answer})
        for index, event in enumerate(events):
            rows.append({"trace_id": trace["trace_id"], "task_id": trace["task_id"], "event_id": event["event_id"], "event_index": index, "shapley_value": values[index], "permutations": permutations, "method": "shapley_credit_local"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "task_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in task_records), encoding="utf-8")
    summary = {"method": "shapley_credit_local", "dataset": dataset, "traces": len(task_records), "success_rate": sum(int(row["success"]) for row in task_records) / len(task_records) if task_records else 0.0, "events_scored": len(rows), "permutations_per_trace": permutations, "api_calls": 0, "gpu": False, "implementation": "Monte Carlo Shapley over verifier-backed factual traces"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("MBPP", "HumanEval", "GSM8K", "Spider", "OpenQA"))
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-data", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--permutations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args()
    run(args.source, args.output, args.dataset, args.spider_root, args.openqa_data, args.limit, args.permutations, args.seed)


if __name__ == "__main__":
    main()
