from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from carve.agents import APIClientConfig, OpenAICompatibleClient
from carve.scoring.oracle import APIJudge, DeterministicJudge, JudgeCommittee
from carve.schemas import Trace


def task_prompt_for_trace(trace: Trace) -> str:
    task = trace.manifest.get("task", {})
    return str(task.get("prompt") or task.get("question") or trace.task_id)


def score_traces_with_committee(traces: list[Trace], committee: JudgeCommittee) -> dict[str, dict[str, Any]]:
    scored: dict[str, dict[str, Any]] = {}
    for trace in traces:
        scores = committee.score(task_prompt_for_trace(trace), trace.final_answer)
        scored[trace.trace_id] = {
            "trace_id": trace.trace_id,
            "task_id": trace.task_id,
            "raw_mean": mean(scores) if scores else 0.0,
            "dispersion": pstdev(scores) if len(scores) > 1 else 0.0,
            "committee_scores": scores,
            "success": trace.success,
            "verifier_score": trace.verifier_score,
            "total_tokens": trace.total_tokens,
            "total_cost_usd": trace.total_cost_usd,
        }
    return scored


def summarize_judge_reranking(traces: list[Trace], scored: dict[str, dict[str, Any]], mode: str) -> dict[str, Any]:
    by_task: dict[str, list[Trace]] = {}
    for trace in traces:
        by_task.setdefault(trace.task_id, []).append(trace)
    eligible = {task_id: candidates for task_id, candidates in by_task.items() if len(candidates) >= 2}
    rankings = {
        task_id: sorted(candidates, key=lambda trace: scored.get(trace.trace_id, {}).get("raw_mean", float("-inf")), reverse=True)
        for task_id, candidates in eligible.items()
    }
    selected = {task_id: candidates[0] for task_id, candidates in rankings.items()}
    single_selected = next(iter(selected.values())) if len(selected) == 1 else None
    return {
        "mode": mode,
        "num_traces": len(traces),
        "num_tasks": len(by_task),
        "eligible_tasks": len(eligible),
        "status": "ok" if eligible else "insufficient_candidates",
        "selected_trace_ids": {task_id: trace.trace_id for task_id, trace in selected.items()},
        "selected_success_rate": mean(int(bool(trace.success)) for trace in selected.values()) if selected else None,
        "selected_trace_id": single_selected.trace_id if single_selected else None,
        "selected_task_id": single_selected.task_id if single_selected else None,
        "selected_score": scored[single_selected.trace_id]["raw_mean"] if single_selected else None,
        "selected_success": single_selected.success if single_selected else None,
        "selected_verifier_score": single_selected.verifier_score if single_selected else None,
        "mean_trace_score": mean(row["raw_mean"] for row in scored.values()) if scored else 0.0,
        "rankings": {task_id: [trace.trace_id for trace in candidates] for task_id, candidates in rankings.items()},
        "ranking": [trace.trace_id for candidates in rankings.values() for trace in candidates],
    }


def load_traces(run_dir: Path) -> list[Trace]:
    trace_path = run_dir / "traces.jsonl"
    return [Trace.from_dict(json.loads(line)) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_committee(api_judges: bool, judge_names: str, samples_per_judge: int) -> JudgeCommittee:
    if api_judges:
        client = OpenAICompatibleClient(APIClientConfig.from_env())
        judges = [APIJudge(name.strip(), client) for name in judge_names.split(",") if name.strip()]
    else:
        judges = [DeterministicJudge("strict"), DeterministicJudge("lenient")]
    return JudgeCommittee(judges, samples_per_judge=samples_per_judge)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--samples-per-judge", type=int, default=3)
    parser.add_argument("--api-judges", action="store_true", help="Use API-backed judges for holistic trajectory reranking")
    parser.add_argument("--judge-names", default="glm_strict,glm_lenient")
    args = parser.parse_args()

    run_dir = Path("artifacts/runs") / args.run_id
    traces = load_traces(run_dir)
    committee = build_committee(args.api_judges, args.judge_names, args.samples_per_judge)
    scored = score_traces_with_committee(traces, committee)
    mode = "api" if args.api_judges else "deterministic"
    summary = summarize_judge_reranking(traces, scored, mode=mode)
    summary["judges"] = [judge.name for judge in committee.judges]
    summary["samples_per_judge"] = args.samples_per_judge

    rows_path = run_dir / "judge_reranking_scores.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for trace_id in summary["ranking"] or scored:
            handle.write(json.dumps(scored[trace_id], ensure_ascii=False) + "\n")
    summary["scores_path"] = str(rows_path)
    out_path = run_dir / "judge_reranking.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
