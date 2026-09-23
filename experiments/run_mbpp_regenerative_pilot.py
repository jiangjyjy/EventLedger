from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.code import CodeVerifier
from experiments.run_mbpp_rq3_baselines import _keep_indices


def _task_prompt(trace: dict) -> tuple[str, str]:
    task = trace["manifest"]["task"]
    return str(task["prompt"]), str(task["tests"])


def _retained(trace: dict, keep: set[int]) -> str:
    return "\n".join(str(event.get("content", "")) for i, event in enumerate(trace.get("events", [])) if i in keep)


def run(source: Path, output: Path, limit: int, model: str, split_file: Path | None = None) -> dict:
    traces = [json.loads(x) for x in source.read_text().splitlines() if x][:limit]
    if split_file is not None:
        split = json.loads(split_file.read_text(encoding="utf-8"))
        test_ids = set(split["test"])
        traces = [trace for trace in [json.loads(x) for x in source.read_text().splitlines() if x] if str(trace["task_id"]) in test_ids]
    config = APIClientConfig.from_env(); config.model = model; config.max_tokens = 1024; config.timeout = 60.0; config.retries_per_url = max(2, config.retries_per_url)
    client = OpenAICompatibleClient(config); verifier = CodeVerifier()
    output.parent.mkdir(parents=True, exist_ok=True); errors = output.with_name("errors.jsonl")
    completed = {json.loads(x)["task_id"] for x in output.read_text().splitlines() if x.strip()} if output.exists() else set()
    rows = [json.loads(x) for x in output.read_text().splitlines() if x.strip()] if output.exists() else []
    for index, trace in enumerate(traces):
        if trace["task_id"] in completed: continue
        prompt, tests = _task_prompt(trace); keep = _keep_indices("masprm_style_local", trace); retained = _retained(trace, keep)
        base = f"Write complete executable Python code only, with no Markdown or explanation. Implement the function required by the task and public tests. Use the retained prior trajectory only as context; correct its mistakes.\n\nTask:\n{prompt}\n\nPublic tests:\n{tests}\n\nRetained trajectory:\n{retained}\n"
        try:
            final = client.complete("mbpp_solver_regenerative", base + "\nReturn the complete code. Check the required function signature and public tests before answering.", index).strip()
            score = verifier.verify(final, tests); row={"task_id":trace["task_id"],"original_success":bool(trace.get("success")),"success":bool(score.success),"final_answer":final,"events_kept":len(keep),"api_calls":3}
            with output.open("a") as handle: handle.write(json.dumps(row,ensure_ascii=False)+"\n"); handle.flush(); os.fsync(handle.fileno())
            rows.append(row); completed.add(trace["task_id"])
        except (RuntimeError, TimeoutError) as exc:
            with errors.open("a") as handle: handle.write(json.dumps({"task_id":trace["task_id"],"error":f"{type(exc).__name__}: {exc}"})+"\n"); handle.flush(); os.fsync(handle.fileno())
    summary={"method":"masprm_style_regenerative_replay","dataset":"MBPP","model":model,"traces":len(rows),"original_successes":sum(int(r["original_success"]) for r in rows),"successes":sum(int(r["success"]) for r in rows),"success_rate":sum(int(r["success"]) for r in rows)/len(rows) if rows else 0.0,"api_calls":3*len(rows)}
    output.with_name("summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2)); return summary


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True,type=Path); p.add_argument("--output",required=True,type=Path); p.add_argument("--limit",type=int,default=20); p.add_argument("--split-file",type=Path); p.add_argument("--model",default="glm-5.2"); a=p.parse_args(); run(a.source,a.output,a.limit,a.model,a.split_file)


if __name__ == "__main__": main()
