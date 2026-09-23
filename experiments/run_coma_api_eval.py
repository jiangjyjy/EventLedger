"""Regenerate COMA-selected GSM8K answers through the provider API.

The COMA policy/training artifacts stay fixed. Only the final answer is
regenerated from events selected by the saved COMA actions, so this run is
separate from the local verifier-replay evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.math import MathVerifier


def _safe_events(events: list[dict], actions: list[int]) -> list[dict]:
    kept = []
    for i, event in enumerate(events):
        if i >= len(actions) or not actions[i] or event.get("type") in {"aggregate", "stop"}:
            continue
        content = str(event.get("content", ""))
        if re.search(r"\bfinal\s+answer\s*:", content, re.IGNORECASE):
            continue
        kept.append(event)
    return kept


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--split-file", required=True, type=Path)
    p.add_argument("--coma-run", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", default="glm-5.2")
    p.add_argument("--max-tokens", type=int, default=1024)
    args = p.parse_args()

    traces = {
        str(row["task_id"]): row
        for row in (json.loads(line) for line in (args.source_run / "traces.jsonl").read_text().splitlines())
        if row
    }
    split = json.loads(args.split_file.read_text())
    test_ids = [str(x) for x in split["test"]]
    actions = {
        str(row["task_id"]): row["actions"]
        for row in (json.loads(line) for line in (args.coma_run / "rollouts.jsonl").read_text().splitlines())
        if row
    }
    missing = [task_id for task_id in test_ids if task_id not in traces]
    if missing:
        raise ValueError(f"missing source traces: {missing[:3]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "results.jsonl"
    error_path = args.output_dir / "errors.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()] if result_path.exists() else []
    completed = {str(row["task_id"]) for row in rows}

    config = APIClientConfig.from_env()
    config.model = args.model
    config.max_tokens = args.max_tokens
    config.timeout = 90.0
    config.retries_per_url = max(3, config.retries_per_url)
    client = OpenAICompatibleClient(config)
    verifier = MathVerifier()

    for index, task_id in enumerate(test_ids):
        if task_id in completed:
            continue
        trace = traces[task_id]
        if task_id not in actions:
            raise ValueError(f"no saved COMA actions for test task {task_id}")
        safe = _safe_events(trace.get("events", []), actions[task_id])
        task = trace.get("manifest", {}).get("task", {})
        question = str(task.get("prompt", task_id))
        reference = str(task.get("reference", trace.get("final_answer", "")))
        retained = "\n".join(str(event.get("content", "")) for event in safe)
        prompt = (
            "Solve the GSM8K math problem. Return the final numeric answer with concise reasoning. "
            "Do not copy an old final answer from the trajectory.\n\n"
            f"Question:\n{question}\n\nCOMA-selected evidence:\n{retained}"
        )
        try:
            started = time.perf_counter()
            answer = str(client.complete("coma_api_test_solver", prompt, index)).strip()
            score = verifier.verify(answer, reference)
            telemetry = client.last_completion_telemetry()
            row = {
                "task_id": task_id,
                "success": bool(score.success),
                "verifier_score": float(score.score) if score.score is not None else None,
                "answer": answer,
                "events_kept": len(safe),
                "events_removed": len(trace.get("events", [])) - len(safe),
                "api_calls": 1,
                "elapsed_seconds": time.perf_counter() - started,
                "telemetry": telemetry,
            }
            with result_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
            completed.add(task_id)
        except (RuntimeError, TimeoutError) as exc:
            with error_path.open("a") as handle:
                handle.write(json.dumps({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    summary = {
        "method": "coma_style_api_regenerative_eval",
        "dataset": "GSM8K",
        "split": "held_out_test",
        "traces": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0,
        "abstained": len(test_ids) - len(rows),
        "api_calls": len(rows),
        "source_coma_run": str(args.coma_run),
        "protocol": "saved COMA actions + answer-sink-free provider regeneration",
        "model": args.model,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
