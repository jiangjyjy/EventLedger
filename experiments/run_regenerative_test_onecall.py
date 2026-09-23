from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.datasets.spider import load_spider_dev
from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from experiments.run_mbpp_rq3_baselines import _keep_indices


def _safe_regeneration_events(trace: dict, keep: set[int], *, dataset: str = "") -> list[dict]:
    """Remove answer sinks and any intermediate event that leaks the final answer."""
    safe = []
    for index, event in enumerate(trace.get("events", [])):
        if index not in keep or event.get("type") == "aggregate":
            continue
        if dataset.lower() == "spider" and event.get("type") == "revise":
            continue
        content = str(event.get("content", ""))
        if re.search(r"\bfinal\s+answer\s*:", content, flags=re.IGNORECASE):
            continue
        safe.append(event)
    return safe


def _answer(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "aggregate":
            return str(event.get("content", ""))
    for event in reversed(events):
        if event.get("type") in {"revise", "msg"}:
            return str(event.get("content", ""))
    return str(events[-1].get("content", "")) if events else ""


def _task_and_reference(trace: dict, dataset: str, spider_root: Path | None, openqa_data: Path | None):
    task = trace.get("manifest", {}).get("task", {})
    key = dataset.lower()
    if key in {"mbpp", "humaneval"}:
        return str(task.get("prompt", trace.get("task_id"))), str(task.get("tests", "")), CodeVerifier()
    if key == "gsm8k":
        return str(task.get("prompt", trace.get("task_id"))), str(task.get("reference", trace.get("final_answer", ""))), MathVerifier()
    if key == "spider":
        if spider_root is None: raise ValueError("--spider-root is required")
        _, suffix = str(trace["task_id"]).rsplit("-dev-", 1)
        case = load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
        return case.question, case, SpiderVerifier()
    if openqa_data is None: raise ValueError("--openqa-data is required")
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(openqa_data)}
    case = cases[str(trace["task_id"])]
    return case.question, case.answers, OpenQAExactMatchVerifier()


def run(source: Path, split_file: Path, dataset: str, output: Path, model: str, spider_root: Path | None, openqa_data: Path | None, method: str, context_mode: str):
    traces = [json.loads(x) for x in source.joinpath("traces.jsonl").read_text().splitlines() if x]
    split = json.loads(split_file.read_text())
    ids = set(split["test"])
    traces = [trace for trace in traces if str(trace["task_id"]) in ids]
    config = APIClientConfig.from_env(); config.model = model; config.max_tokens = 1024; config.timeout = 90.0; config.retries_per_url = max(2, config.retries_per_url)
    client = OpenAICompatibleClient(config); output.mkdir(parents=True, exist_ok=True); result_path = output / "results.jsonl"; error_path = output / "errors.jsonl"
    completed = {json.loads(x)["task_id"] for x in result_path.read_text().splitlines() if x.strip()} if result_path.exists() else set()
    rows = [json.loads(x) for x in result_path.read_text().splitlines() if x.strip()] if result_path.exists() else []
    for index, trace in enumerate(traces):
        if str(trace["task_id"]) in completed: continue
        if completed:
            time.sleep(float(os.environ.get("CARVE_BETWEEN_TASKS_SECONDS", "0")))
        question, reference, verifier = _task_and_reference(trace, dataset, spider_root, openqa_data)
        keep = set(range(len(trace.get("events", [])))) if context_mode == "full_context" and dataset.lower() == "spider" else _keep_indices(method, trace)
        safe_events = _safe_regeneration_events(trace, keep, dataset=dataset)
        retained = "\n".join(str(event.get("content", "")) for event in safe_events)
        if dataset.lower() == "gsm8k": instruction = "Solve the math problem and return the final numeric answer with concise reasoning."
        elif dataset.lower() in {"mbpp", "humaneval"}: instruction = "Return complete executable Python code only."
        elif dataset.lower() == "spider": instruction = "Return one executable SQLite SELECT or WITH query only."
        else: instruction = "Return only the shortest exact answer span, with no explanation."
        prompt = f"{instruction} Reconstruct the answer from the retained trajectory; do not copy an old final answer blindly.\n\nQuestion:\n{question}\n\nRetained trajectory:\n{retained}"
        try:
            answer = str(client.complete("regenerative_test_solver", prompt, index)).strip()
            score = verifier.verify(answer, reference)
            row = {"task_id": trace["task_id"], "dataset": dataset, "original_success": bool(trace.get("success")), "success": bool(score.success), "answer": answer, "events_kept": len(safe_events), "events_removed": len(trace.get("events", [])) - len(safe_events), "api_calls": 1, "telemetry": client.last_completion_telemetry()}
            with result_path.open("a") as handle: handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
            rows.append(row); completed.add(str(trace["task_id"]))
        except (RuntimeError, TimeoutError) as exc:
            with error_path.open("a") as handle: handle.write(json.dumps({"task_id":trace["task_id"],"error":f"{type(exc).__name__}: {exc}"})+"\n"); handle.flush(); os.fsync(handle.fileno())
    summary={"method":f"{method}_regenerative_test_onecall","dataset":dataset,"context_mode":context_mode,"traces":len(rows),"successes":sum(int(r["success"]) for r in rows),"success_rate":sum(int(r["success"]) for r in rows)/len(rows) if rows else 0.0,"original_successes":sum(int(r["original_success"]) for r in rows),"recovery":sum(int(r["success"] and not r["original_success"]) for r in rows),"regression":sum(int((not r["success"]) and r["original_success"]) for r in rows),"api_calls":len(rows),"protocol":"held_out_test_only_onecall_regenerative_replay"}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True,type=Path); p.add_argument("--split-file",required=True,type=Path); p.add_argument("--dataset",required=True); p.add_argument("--output",required=True,type=Path); p.add_argument("--model",default="glm-5.2"); p.add_argument("--method",default="masprm_style_local"); p.add_argument("--context-mode",choices=("selective","full_context"),default="selective"); p.add_argument("--spider-root",type=Path); p.add_argument("--openqa-data",type=Path); a=p.parse_args(); run(a.source,a.split_file,a.dataset,a.output,a.model,a.spider_root,a.openqa_data,a.method,a.context_mode)


if __name__ == "__main__": main()
