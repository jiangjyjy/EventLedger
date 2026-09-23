from __future__ import annotations

import argparse
import json
from pathlib import Path

from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.datasets.nq_openqa import load_nq_openqa_jsonl
from carve.verifiers.openqa import OpenQAExactMatchVerifier


def run(traces_path: Path, cases_path: Path, output: Path, model: str, limit: int | None) -> dict:
    traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {case.task_id: case for case in load_nq_openqa_jsonl(cases_path)}
    traces = [trace for trace in traces if not trace.get("success")]
    if limit is not None:
        traces = traces[:limit]
    config = APIClientConfig.from_env()
    config.model = model
    config.max_tokens = 256
    client = OpenAICompatibleClient(config)
    verifier = OpenQAExactMatchVerifier()
    rows = []
    for trace in traces:
        case = cases[trace["task_id"]]
        events = {event["event_id"]: event for event in trace["events"]}
        a = events["e3"].get("metadata", {}).get("answer", "")
        b = events["e4"].get("metadata", {}).get("answer", "")
        ca = events["e3"].get("metadata", {}).get("citation")
        cb = events["e4"].get("metadata", {}).get("citation")
        prompt = (
            "Choose the candidate that answers the question exactly. Preserve required qualifiers, dates, "
            "editions, and list members. Do not infer from outside knowledge. Return exactly candidate_a, "
            "candidate_b, or abstain.\n\n"
            f"Question: {case.question}\nCandidate A: {a}\nCitation A: {ca}\n"
            f"Candidate B: {b}\nCitation B: {cb}\n"
        )
        choice = str(client.complete("second_judge", prompt, len(rows))).strip().splitlines()[0].lower()
        if choice not in {"candidate_a", "candidate_b"}:
            choice = "abstain"
        answer = a if choice == "candidate_a" else b if choice == "candidate_b" else ""
        score = verifier.verify(answer, case.answers)
        rows.append({"task_id": trace["task_id"], "choice": choice, "answer": answer, "success": bool(score.success), "gold": list(case.answers)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"model": model, "input_failed_traces": len(rows), "successes": sum(int(row["success"]) for row in rows), "success_rate": sum(int(row["success"]) for row in rows) / len(rows) if rows else 0.0, "api_calls": len(rows), "role": "second_selector_judge_only"}
    output.with_name("summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="glm-5.1")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.traces, args.cases, args.output, args.model, args.limit)


if __name__ == "__main__":
    main()
