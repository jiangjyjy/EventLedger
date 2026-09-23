"""Unprotected structural replay: terminal answer events are not forced to survive."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from statistics import mean

from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from carve.datasets.spider import load_spider_dev
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from experiments.run_mbpp_rq3_baselines import (
    _event_evidence,
    _events,
    _keep_indices,
    _load_jsonl,
    masprm_event_scores,
)


def terminal_answer(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "aggregate":
            return str(event.get("content", ""))
    for event in reversed(events):
        if event.get("type") in {"revise", "msg"}:
            return str(event.get("content", ""))
    return str(events[-1].get("content", "")) if events else ""


def _telemetry(events: list[dict]) -> dict[str, float]:
    api = tokens = latency = 0.0
    tools = 0
    for event in events:
        t = event.get("metadata", {}).get("telemetry", {}) or {}
        api += float(t.get("api_calls", 0) or 0)
        tokens += float(t.get("input_tokens", 0) or 0) + float(t.get("output_tokens", 0) or 0)
        latency += float(t.get("wall_clock_latency_ms", 0) or 0)
        tools += int(event.get("type") == "tool")
    return {"api_calls": api, "tokens": tokens, "tool_calls": float(tools), "latency_ms": latency}


def run(source: Path, output: Path, method: str, dataset: str, spider_root: Path | None = None, openqa_data: Path | None = None) -> dict:
    traces = _load_jsonl(source)
    key = dataset.lower()
    verifier = CodeVerifier() if key in {"mbpp", "humaneval"} else MathVerifier() if key == "gsm8k" else SpiderVerifier() if key == "spider" else OpenQAExactMatchVerifier()
    references = {}
    if key == "spider":
        if spider_root is None: raise ValueError("--spider-root is required")
        for trace in traces:
            _, suffix = trace["task_id"].rsplit("-dev-", 1)
            references[trace["task_id"]] = load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
    elif key == "openqa":
        if openqa_data is None: raise ValueError("--openqa-data is required")
        references = {case.task_id: case.answers for case in load_nq_openqa_jsonl(openqa_data)}
    rows = []
    for trace in traces:
        events = trace.get("events", [])
        keep = _keep_indices(method, trace)
        controlled = [copy.deepcopy(event) for index, event in enumerate(events) if index in keep]
        answer = terminal_answer(controlled)
        if key in {"mbpp", "humaneval"}: reference = str(trace["manifest"]["task"]["tests"])
        elif key == "gsm8k": reference = str(trace["manifest"]["task"].get("reference", trace.get("final_answer", "")))
        else: reference = references[trace["task_id"]]
        result = verifier.verify(answer, reference)
        rows.append({
            "task_id": trace["task_id"],
            "method": method,
            "success_original": bool(trace.get("success", False)),
            "success_unprotected": bool(result.success),
            "answer": answer,
            "events_original": len(events),
            "events_kept": len(controlled),
            "events_removed": len(events) - len(controlled),
            **_telemetry(controlled),
        })
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "dataset": dataset,
        "method": method,
        "protocol": "unprotected_structural_replay",
        "traces": len(rows),
        "original_success_rate": sum(int(row["success_original"]) for row in rows) / len(rows) if rows else 0.0,
        "unprotected_success_rate": sum(int(row["success_unprotected"]) for row in rows) / len(rows) if rows else 0.0,
        "answer_lost_or_invalid": sum(int(not row["success_unprotected"]) for row in rows),
        "mean_events_kept": mean(row["events_kept"] for row in rows) if rows else 0.0,
        "mean_events_removed": mean(row["events_removed"] for row in rows) if rows else 0.0,
        "mean_tokens": mean(row["tokens"] for row in rows) if rows else 0.0,
        "api_calls": 0,
        "gpu": False,
        "note": "Final answer event was not protected; answer was re-extracted and locally re-verified.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=("MBPP", "HumanEval", "GSM8K", "Spider", "OpenQA"))
    parser.add_argument("--spider-root", type=Path)
    parser.add_argument("--openqa-data", type=Path)
    args = parser.parse_args()
    for method in ("outcome_reward_rl", "ruler_style_local", "masprm_style_local"):
        run(args.source, args.output_root / method, method, args.dataset, args.spider_root, args.openqa_data)


if __name__ == "__main__":
    main()
