"""Strict Wang et al. (2022) self-consistency baseline for GSM8K."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.verifiers.math import MathVerifier
from experiments.run_gsm8k_single_cot import source_tasks


def prompt(task: dict) -> str:
    return (
        "Solve this grade-school math problem independently. Show concise reasoning, "
        "then put only the final numeric answer on its own line exactly as `#### <number>`.\n\n"
        f"Problem:\n{task['prompt']}"
    )


def answer_key(text: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run(tasks, output: Path, client, *, samples: int, seed: int, model: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    verifier = MathVerifier()
    for index, task in enumerate(tasks):
        samples_out = []
        for sample in range(samples):
            text = client.complete("self_consistency_solver", prompt(task), seed + index * samples + sample).strip()
            samples_out.append({"text": text, "answer": answer_key(text), "telemetry": client.last_completion_telemetry()})
        valid = [row["answer"] for row in samples_out if row["answer"] is not None]
        counts = Counter(valid)
        voted = counts.most_common(1)[0][0] if counts else None
        voted_text = f"#### {voted}" if voted is not None else ""
        score = verifier.verify(voted_text, str(task["reference"]))
        traces.append({"task_id": task["task_id"], "method": "self_consistency", "model": model, "samples": samples_out, "vote": dict(counts), "voted_answer": voted, "success": score.success, "verifier_score": score.score})
        (output / "traces.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces), encoding="utf-8")
    summary = {"method": "self_consistency", "model": model, "tasks": len(traces), "samples_per_task": samples, "api_calls": len(traces) * samples, "successes": sum(int(row["success"]) for row in traces), "success_rate": sum(int(row["success"]) for row in traces) / len(traces) if traces else 0.0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args()
    tasks = source_tasks(args.source_run)[: args.limit]
    config = APIClientConfig.from_env()
    config.model = os.environ.get("CARVE_MODEL", "glm-5.2")
    if any(url.rstrip("/").endswith("/v1") for url in config.base_urls):
        config.chat_completions_path = "/chat/completions"
    config.max_tokens, config.timeout, config.retries_per_url = 1024, 180.0, max(2, config.retries_per_url)
    print(json.dumps(run(tasks, args.output, OpenAICompatibleClient(config), samples=args.samples, seed=args.seed, model=config.model), indent=2))


if __name__ == "__main__":
    main()
