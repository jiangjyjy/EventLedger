"""Run a five-dataset Self-Consistency test suite with independent samples."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.datasets.spider import load_spider_dev
from carve.schemas import Trace
from carve.verifiers.code import CodeVerifier
from carve.verifiers.math import MathVerifier
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from carve.verifiers.spider import SpiderVerifier


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def traces(path: Path) -> list[Trace]:
    return [Trace.from_dict(row) for row in jsonl(path)]


def task_from_trace(trace: Trace) -> dict:
    task = trace.manifest.get("task")
    if not isinstance(task, dict):
        raise ValueError(f"missing task manifest for {trace.task_id}")
    return task


def normalize(text: str, regime: str) -> str:
    text = text.strip()
    if regime in {"humaneval", "mbpp"}:
        text = re.sub(r"^```(?:python)?\s*|\s*```$", "", text, flags=re.I | re.S)
        return re.sub(r"\s+", " ", text).strip()
    if regime == "spider":
        text = re.sub(r"```(?:sql)?|```", "", text, flags=re.I)
        return re.sub(r"\s+", " ", text).strip().lower().rstrip(";")
    if regime == "openqa":
        return re.sub(r"\s+", " ", text).strip().lower()
    answer = re.findall(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    return answer[-1].replace(",", "") if answer else ""


def prompt_for(regime: str, item: dict) -> str:
    if regime == "code_math":
        return "Solve independently. Show concise reasoning and put the final numeric answer on its own line exactly as `#### <number>`.\n\nProblem:\n" + str(item["prompt"])
    if regime in {"humaneval", "mbpp"}:
        extra = "\n\nPublic tests:\n" + str(item.get("tests", "")) if regime == "mbpp" else ""
        return "Write complete executable Python code only, with no Markdown or explanation.\n\nTask:\n" + str(item["prompt"]) + extra
    if regime == "spider":
        return "Write one executable SQLite SELECT or WITH query that answers the question. Return SQL only, with no Markdown or explanation.\n\nQuestion:\n" + str(item["question"]) + "\n\nSQLite schema:\n" + str(item["schema"])
    return "Answer using only the retrieved evidence. Return only the shortest exact answer span, with no explanation.\n\nQuestion:\n" + str(item["question"]) + "\n\nRetrieved evidence:\n" + str(item["evidence"])


def vote(samples: list[dict], regime: str) -> tuple[str, dict[str, int]]:
    keys = [normalize(row["text"], regime) for row in samples]
    counts = Counter(key for key in keys if key)
    winner = counts.most_common(1)[0][0] if counts else ""
    return winner, dict(counts)


def run_dataset(regime: str, items: list[dict], output: Path, client: OpenAICompatibleClient, samples: int, seed: int, model: str, resume: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "traces.jsonl"
    existing = {str(row["task_id"]): row for row in jsonl(path)} if path.exists() and resume else {}
    # Error records are transient provider failures and must not block a
    # later resume after the endpoint recovers.
    error_path = output / "errors.jsonl"
    verifier_math = MathVerifier() if regime == "code_math" else None
    verifier_code = CodeVerifier() if regime in {"humaneval", "mbpp"} else None
    verifier_spider = SpiderVerifier() if regime == "spider" else None
    verifier_openqa = OpenQAExactMatchVerifier() if regime == "openqa" else None
    records = []
    for index, item in enumerate(items):
        task_id = str(item["task_id"])
        if task_id in existing and len(existing[task_id].get("samples", [])) == samples:
            records.append(existing[task_id]); continue
        sampled = []
        try:
            for sample in range(samples):
                text = client.complete("self_consistency_solver", prompt_for(regime, item), seed + index * samples + sample).strip()
                sampled.append({"text": text, "normalized": normalize(text, regime), "telemetry": client.last_completion_telemetry()})
        except (RuntimeError, TimeoutError) as exc:
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            continue
        winner, counts = vote(sampled, regime)
        if regime == "code_math":
            result = verifier_math.verify("#### " + winner, str(item["reference"]))
        elif regime == "humaneval":
            tests = str(item["tests"]); entry = item.get("entry_point")
            if entry and "def check(" in tests: tests += f"\ncheck({entry})\n"
            result = verifier_code.verify(next((s["text"] for s in sampled if s["normalized"] == winner), ""), tests)
        elif regime == "mbpp":
            result = verifier_code.verify(next((s["text"] for s in sampled if s["normalized"] == winner), ""), str(item["tests"]))
        elif regime == "spider":
            result = verifier_spider.verify(winner, item["case"])
        else:
            result = verifier_openqa.verify(winner, item["aliases"])
        row = {"task_id": task_id, "dataset": regime, "method": "self_consistency", "model": model, "samples": sampled, "vote": counts, "voted_answer": winner, "success": bool(result.success), "verifier_score": result.score, "verifier_details": result.details}
        records.append(row)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = {"method": "self_consistency", "dataset": regime, "model": model, "tasks": len(records), "samples_per_task": samples, "api_calls": len(records) * samples, "successes": sum(int(r["success"]) for r in records), "success_rate": sum(int(r["success"]) for r in records) / len(records) if records else 0.0}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root; out = args.output_root
    config = APIClientConfig.from_env(); config.model = os.environ.get("CARVE_MODEL", "deepseek-v4-flash"); config.max_tokens = 2048; config.timeout = 180.0; config.chat_completions_path = "/chat/completions" if any(u.endswith("/v1") for u in config.base_urls) else config.chat_completions_path
    client = OpenAICompatibleClient(config)
    sources = {
        "code_math": (root / "artifacts/runs/GSM8K/gsm8k_distill_merged_full900_v1/traces.jsonl", root / "artifacts/student_runs/gsm8k_qwen35_9b_bf16_lora_full720_v1/split.json"),
        "humaneval": (root / "artifacts/runs", root / "artifacts/student_runs/humaneval_qwen35_9b_bf16_lora_full_seed0_v1/split.json"),
        "mbpp": (root / "artifacts/runs/MBPP/mbpp_distill_merged_shard00_02_clean_v1/traces.jsonl", root / "artifacts/student_runs/mbpp_qwen35_9b_bf16_lora_merged321_seed0_v1/split.json"),
        "spider": (root / "artifacts/formal_ablation_subset_20260819/ranking_beta0_inputs/sql", root / "artifacts/formal_ablation_subset_20260819/ranking_beta0_inputs/sql/split.json"),
        "openqa": (root / "artifacts/formal_ablation_subset_20260819/ranking_beta0_inputs/openqa", root / "artifacts/formal_ablation_subset_20260819/ranking_beta0_inputs/openqa/split.json"),
    }
    # Materialize task objects from the existing factual traces and fixed test splits.
    for regime, (source, split_path) in sources.items():
        split = json.loads(split_path.read_text())
        test_ids = set(str(x) for x in split["test"])
        if regime == "humaneval":
            raw = []
            for p in sorted(source.glob("humaneval_glm52_code_v2_full_k3_top8_ops3_task*/traces.jsonl")):
                raw.extend(traces(p))
            items = [task_from_trace(t) for t in raw if t.task_id in test_ids]
        elif regime == "spider":
            raw = traces(source / "traces.jsonl"); spider_root = root / "artifacts/spider_probe/source/spider_data"
            items = []
            for t in raw:
                if t.task_id in test_ids:
                    index = int(t.task_id.rsplit("-dev-", 1)[1]); case = load_spider_dev(spider_root, limit=1, offset=index)[0]
                    items.append({"task_id": t.task_id, "question": case.question, "schema": case.schema, "case": case})
        elif regime == "openqa":
            raw = traces(source / "traces.jsonl"); cases = {c.task_id: c for c in load_nq_openqa_jsonl(root / ".worktrees/swe-derived-pilot/data/openqa/nq_open_dpr_dev100.jsonl")}
            items = []
            for t in raw:
                if t.task_id in test_ids:
                    c = cases[t.task_id]
                    e = next(e for e in t.events if e.event_id == "e1")
                    items.append({"task_id": t.task_id, "question": c.question, "aliases": list(c.answers), "evidence": e.content})
        else:
            raw = traces(source); items = [task_from_trace(t) for t in raw if t.task_id in test_ids]
        run_dataset(regime, items, out / regime, client, args.samples, args.seed, config.model, args.resume)


if __name__ == "__main__":
    main()
