from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from experiments.run_mbpp_rq3_baselines import _keep_indices


def _kept_context(trace: dict, keep: set[int]) -> str:
    parts = []
    for index, event in enumerate(trace.get("events", [])):
        if index in keep:
            parts.append(f"[{event.get('event_id')}/{event.get('type')}] {event.get('content', '')}")
    return "\n".join(parts)


def run(traces_path: Path, cases_path: Path, output: Path, limit: int, model: str) -> dict:
    traces = [json.loads(x) for x in traces_path.read_text().splitlines() if x][:limit]
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(cases_path)}
    config = APIClientConfig.from_env()
    config.model = model
    config.max_tokens = 512
    config.timeout = float(os.environ.get("CARVE_REGENERATIVE_TIMEOUT", "60"))
    config.retries_per_url = max(2, config.retries_per_url)
    client = OpenAICompatibleClient(config)
    verifier = OpenQAExactMatchVerifier()
    output.parent.mkdir(parents=True, exist_ok=True)
    error_path = output.with_name("errors.jsonl")
    completed = {json.loads(line)["task_id"] for line in output.read_text().splitlines() if line.strip()} if output.exists() else set()
    failed = {json.loads(line)["task_id"] for line in error_path.read_text().splitlines() if line.strip()} if error_path.exists() else set()
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()] if output.exists() else []
    for trace in traces:
        if trace["task_id"] in completed:
            continue
        case = cases[trace["task_id"]]
        keep = _keep_indices("masprm_style_local", trace)
        context = _kept_context(trace, keep)
        base = (
            "Use only the retained trajectory and evidence. Answer the question with the shortest exact answer span. "
            "Do not explain or answer Unknown when the retained evidence supports an answer.\n\n"
            f"Question: {case.question}\nRetained trajectory:\n{context}\n"
        )
        try:
            a = str(client.complete("reader_a_regenerative", base + "\nReturn only the answer.", len(rows))).strip()
            b = str(client.complete("reader_b_regenerative", base + f"\nCandidate A: {a}\nReturn the best corrected answer only.", len(rows) + 1000)).strip()
            selector_prompt = (
                f"Question: {case.question}\nCandidate A: {a}\nCandidate B: {b}\n"
                "Choose the candidate that exactly answers the question. Return only the chosen answer, no explanation."
            )
            answer = str(client.complete("selector_regenerative", selector_prompt, len(rows) + 2000)).strip()
            score = verifier.verify(answer, case.answers)
            row = {"task_id": trace["task_id"], "original_success": bool(trace.get("success")), "success": bool(score.success), "answer": answer, "candidate_a": a, "candidate_b": b, "events_kept": len(keep), "api_calls": 3}
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(row)
            completed.add(trace["task_id"])
        except (RuntimeError, TimeoutError) as exc:
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"task_id": trace["task_id"], "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            failed.add(trace["task_id"])
    summary = {"method": "masprm_style_regenerative_replay", "model": model, "traces": len(rows), "original_successes": sum(int(r["original_success"]) for r in rows), "successes": sum(int(r["success"]) for r in rows), "success_rate": sum(int(r["success"]) for r in rows) / len(rows) if rows else 0.0, "api_calls": 3 * len(rows), "protocol": "MASPRM-style unprotected replay with regenerated readers and selector"}
    output.with_name("summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, type=Path)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--model", default="glm-5.2")
    a = p.parse_args()
    run(a.traces, a.cases, a.output, a.limit, a.model)


if __name__ == "__main__":
    main()
