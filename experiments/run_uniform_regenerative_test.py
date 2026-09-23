from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.datasets.spider import load_spider_dev


def _safe_events(trace: dict, kept_ids: set[str]) -> list[dict]:
    events = []
    for event in trace.get("events", []):
        if event.get("event_id") not in kept_ids or event.get("type") == "aggregate":
            continue
        if re.search(r"\bfinal\s+answer\s*:", str(event.get("content", "")), re.I):
            continue
        events.append(event)
    return events


def _reference(trace: dict, dataset: str, spider_root: Path | None, openqa_data: Path | None):
    task = trace.get("manifest", {}).get("task", {})
    key = dataset.lower()
    if key in {"gsm8k", "math"}:
        return MathVerifier(), str(task.get("reference", trace.get("final_answer", "")))
    if key in {"mbpp", "humaneval"}:
        return CodeVerifier(), str(task.get("tests", ""))
    if key == "spider":
        if spider_root is None: raise ValueError("--spider-root is required")
        _, suffix = trace["task_id"].rsplit("-dev-", 1)
        return SpiderVerifier(), load_spider_dev(spider_root, limit=1, offset=int(suffix))[0]
    if openqa_data is None: raise ValueError("--openqa-data is required")
    cases = {x.task_id: x for x in load_nq_openqa_jsonl(openqa_data)}
    return OpenQAExactMatchVerifier(), cases[trace["task_id"]].answers


def _question(trace: dict, dataset: str, openqa_data: Path | None) -> str:
    prompt = trace.get("manifest", {}).get("task", {}).get("prompt")
    if prompt:
        return str(prompt)
    if dataset.lower() == "openqa" and openqa_data:
        return next(x.question for x in load_nq_openqa_jsonl(openqa_data) if x.task_id == trace["task_id"])
    return trace["task_id"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path); p.add_argument("--controlled", required=True, type=Path); p.add_argument("--split-file", required=True, type=Path)
    p.add_argument("--dataset", required=True); p.add_argument("--output", required=True, type=Path); p.add_argument("--model", default="glm-5.2"); p.add_argument("--spider-root", type=Path); p.add_argument("--openqa-data", type=Path); p.add_argument("--sleep", type=float, default=3.0)
    a=p.parse_args(); all_traces=[json.loads(x) for x in a.source.read_text().splitlines() if x.strip()]; selected=set(json.loads(a.split_file.read_text())["test"]); traces=[x for x in all_traces if x["task_id"] in selected]
    controls={x["task_id"]:x for x in (json.loads(y) for y in a.controlled.read_text().splitlines() if y.strip())}
    cfg=APIClientConfig.from_env(); cfg.model=a.model; cfg.max_tokens=1024; cfg.timeout=120; cfg.retries_per_url=max(3,cfg.retries_per_url); client=OpenAICompatibleClient(cfg)
    a.output.mkdir(parents=True, exist_ok=True); result_path=a.output/"results.jsonl"; error_path=a.output/"errors.jsonl"; rows=[json.loads(x) for x in result_path.read_text().splitlines() if x.strip()] if result_path.exists() else []; done={x["task_id"] for x in rows}
    for trace in traces:
        if trace["task_id"] in done: continue
        control=controls.get(trace["task_id"],{}); kept={e.get("event_id") for e in control.get("events",[])}; events=_safe_events(trace,kept); context="\n".join(f"[{e.get('type')}] {e.get('content','')}" for e in events)
        if a.dataset.lower() == "gsm8k":
            instruction = "Solve the math problem and return the final numeric answer with concise reasoning."
        elif a.dataset.lower() in {"mbpp", "humaneval"}:
            instruction = "Return complete executable Python code only, with the requested function definition."
        elif a.dataset.lower() == "spider":
            instruction = "Return one executable SQLite SELECT or WITH query only."
        else:
            instruction = "Return only the shortest exact answer span, with no explanation, markdown, code, or search script."
        task = trace.get("manifest", {}).get("task", {})
        tests = str(task.get("tests", "")) if a.dataset.lower() in {"mbpp", "humaneval"} else ""
        prompt=f"{instruction} Solve independently; do not copy an old final answer or candidate.\n\nQuestion:\n{_question(trace,a.dataset,a.openqa_data)}\n\nRequired tests (use only to infer the function contract):\n{tests[:5000]}\n\nRetained evidence:\n{context[:9000]}"
        try:
            answer=str(client.complete("uniform_credit_regenerative_solver",prompt,len(rows))).strip(); verifier,reference=_reference(trace,a.dataset,a.spider_root,a.openqa_data); score=verifier.verify(answer,reference); telemetry=client.last_completion_telemetry(); row={"task_id":trace["task_id"],"dataset":a.dataset,"original_success":bool(trace.get("success")),"success":bool(score.success),"answer":answer,"events_kept":len(events),"events_removed":len(trace.get("events",[]))-len(events),"api_calls":1,"telemetry":telemetry}; rows.append(row); result_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows)); done.add(trace["task_id"])
        except (RuntimeError,TimeoutError) as exc:
            with error_path.open("a") as h: h.write(json.dumps({"task_id":trace["task_id"],"abstained":True,"error":str(exc)})+"\n")
        time.sleep(a.sleep)
    summary={"method":"uniform_credit_regenerative_test","dataset":a.dataset,"traces":len(rows),"successes":sum(int(x["success"]) for x in rows),"success_rate":sum(int(x["success"]) for x in rows)/len(rows) if rows else 0.0,"original_successes":sum(int(x["original_success"]) for x in rows),"api_calls":sum(int(x.get("api_calls",0)) for x in rows),"protocol":"held_out_test_only_regenerative_replay_from_uniform_policy_actions"}; (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__": main()
