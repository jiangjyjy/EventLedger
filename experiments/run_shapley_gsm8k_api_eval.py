"""Resumable GSM8K API evaluation using Monte Carlo Shapley event credit."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.math import MathVerifier


def terminal_answer(events: list[dict]) -> str:
    for event in reversed(events):
        if event.get("type") == "aggregate":
            return str(event.get("content", ""))
    for event in reversed(events):
        if event.get("type") in {"revise", "msg"}:
            return str(event.get("content", ""))
    return str(events[-1].get("content", "")) if events else ""


def shapley_values(events: list[dict], reference: str, permutations: int, seed: int) -> list[float]:
    verifier = MathVerifier()
    values = [0.0] * len(events)
    rng = random.Random(seed)
    for _ in range(permutations):
        order = list(range(len(events)))
        rng.shuffle(order)
        coalition: set[int] = set()
        before = 0.0
        for index in order:
            coalition.add(index)
            after = float(verifier.verify(terminal_answer([event for i, event in enumerate(events) if i in coalition]), reference).success)
            values[index] += after - before
            before = after
    return [value / permutations for value in values]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--split-file", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--permutations", type=int, default=128)
    p.add_argument("--seed", type=int, default=81)
    p.add_argument("--model", default="glm-5.2")
    args = p.parse_args()
    traces = {row["task_id"]: row for row in (json.loads(line) for line in (args.source_run / "traces.jsonl").read_text().splitlines()) if row}
    test_ids = json.loads(args.split_file.read_text())["test"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path, error_path, label_path = args.output_dir / "results.jsonl", args.output_dir / "errors.jsonl", args.output_dir / "shapley_labels.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()] if result_path.exists() else []
    done = {row["task_id"] for row in rows}
    known_labels = {row["task_id"]: row for line in label_path.read_text().splitlines() if line.strip() for row in [json.loads(line)]} if label_path.exists() else {}
    cfg = APIClientConfig.from_env(); cfg.model = args.model; cfg.max_tokens = 1024; cfg.timeout = 90.0; cfg.retries_per_url = max(3, cfg.retries_per_url)
    client, verifier = OpenAICompatibleClient(cfg), MathVerifier()
    for index, task_id in enumerate(test_ids):
        if task_id in done: continue
        trace = traces[task_id]; task = trace["manifest"]["task"]; events = trace["events"]
        label = known_labels.get(task_id)
        if label is None:
            values = shapley_values(events, str(task.get("reference", trace.get("final_answer", ""))), args.permutations, args.seed + index)
            label = {"task_id": task_id, "permutations": args.permutations, "values": values}
            with label_path.open("a") as handle:
                handle.write(json.dumps(label) + "\n"); handle.flush(); os.fsync(handle.fileno())
        kept = [event for event, value in zip(events, label["values"]) if value > 0 and event.get("type") not in {"aggregate", "stop"} and not re.search(r"final\s+answer\s*:", str(event.get("content", "")), re.I)]
        evidence = "\n".join(str(event.get("content", "")) for event in kept)
        prompt = "Solve the GSM8K math problem and return the final numeric answer with concise reasoning. Do not copy any old final answer.\n\nQuestion:\n" + str(task.get("prompt", task_id)) + "\n\nShapley-selected evidence:\n" + evidence
        try:
            started = time.perf_counter(); answer = str(client.complete("shapley_api_test", prompt, index)).strip(); score = verifier.verify(answer, str(task.get("reference", trace.get("final_answer", ""))))
            row = {"task_id": task_id, "success": bool(score.success), "answer": answer, "api_calls": 1, "events_kept": len(kept), "elapsed_seconds": time.perf_counter() - started, "telemetry": client.last_completion_telemetry()}
            with result_path.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n"); handle.flush(); os.fsync(handle.fileno())
            rows.append(row); done.add(task_id)
        except (RuntimeError, TimeoutError) as exc:
            with error_path.open("a") as handle:
                handle.write(json.dumps({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n"); handle.flush(); os.fsync(handle.fileno())
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()] if result_path.exists() else []
    summary = {"method": "monte_carlo_shapley_api_regeneration", "dataset": "GSM8K", "test_tasks": len(test_ids), "completed": len(rows), "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0, "abstained": len(test_ids) - len(rows), "api_calls": len(rows), "permutations": args.permutations, "resumable": True}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
