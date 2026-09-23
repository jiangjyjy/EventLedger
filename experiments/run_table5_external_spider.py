"""Generate resumable two-candidate Spider traces for Table 5 transfer runs."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def complete(*, endpoint: str, api_key: str, model: str, prompt: str, seed: int) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512,
        "seed": seed,
    }
    started = time.perf_counter()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    message = parsed["choices"][0]["message"]
    text = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not text:
        raise ValueError("empty completion")
    usage = parsed.get("usage") or {}
    return text, {
        "api_calls": 1,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="https://api.openai.com/v1")
    parser.add_argument("--max-calls", type=int, default=200)
    parser.add_argument("--api-key-env", default="TABLE5_PROVIDER_KEY")
    args = parser.parse_args()
    api_key = os.environ[args.api_key_env]
    tasks = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    records_path, errors_path = args.output / "traces.jsonl", args.output / "errors.jsonl"
    done = {json.loads(line)["task_id"] for line in records_path.read_text().splitlines() if line.strip()} if records_path.exists() else set()
    used_calls = sum(int(json.loads(line).get("api_calls", 0)) for line in records_path.read_text().splitlines() if line.strip()) if records_path.exists() else 0
    used_calls += sum(int(json.loads(line).get("api_calls", 0)) for line in errors_path.read_text().splitlines() if line.strip()) if errors_path.exists() else 0
    for index, task in enumerate(tasks):
        if task["task_id"] in done:
            continue
        if used_calls + 2 > args.max_calls:
            break
        # Reserve both solver calls before issuing either request, so failed
        # tasks cannot accidentally exceed the provider budget.
        used_calls += 2
        prompt = (
            "Return one executable SQLite SELECT or WITH query only.\n\n"
            f"Question:\n{task['question']}\n\nSchema:\n{task['schema']}"
        )
        try:
            candidate_a, telemetry_a = complete(endpoint=args.endpoint, api_key=api_key, model=args.model, prompt=prompt, seed=index * 2)
            candidate_b, telemetry_b = complete(endpoint=args.endpoint, api_key=api_key, model=args.model, prompt=prompt, seed=index * 2 + 1)
            append(records_path, {
                "task_id": task["task_id"], "question": task["question"], "schema": task["schema"],
                "candidate_a": candidate_a, "candidate_b": candidate_b, "model": args.model,
                "api_calls": 2, "telemetry": [telemetry_a, telemetry_b],
            })
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError) as exc:
            append(errors_path, {"task_id": task["task_id"], "api_calls": 2, "error": f"{type(exc).__name__}: {exc}"})
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    error_rows = [json.loads(line) for line in errors_path.read_text().splitlines() if line.strip()] if errors_path.exists() else []
    summary = {"model": args.model, "tasks": len(tasks), "completed": len(rows), "failed": len(error_rows), "api_calls": sum(row["api_calls"] for row in rows) + sum(row.get("api_calls", 0) for row in error_rows), "max_calls": args.max_calls}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
