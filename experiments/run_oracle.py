from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from carve.agents import APIClientConfig, OpenAICompatibleClient
from carve.scoring.calibration import CalibratedOracle
from carve.scoring.oracle import APIJudge, DeterministicJudge, JudgeCommittee, load_anchor_scores, write_default_anchor_file
from carve.schemas import Trace
from experiments.prepare_openqa_anchors import calibration_report_from_anchors, read_jsonl


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    family: str
    method_model: str
    runtime_model: str


def default_method_judge_specs(runtime_model: str | None = None) -> list[JudgeSpec]:
    active_glm = runtime_model or os.environ.get("CARVE_MODEL", "GLM-5.1")
    return [
        JudgeSpec(name="glm_primary", family="glm", method_model="GLM-5.1", runtime_model=active_glm),
        JudgeSpec(name="gpt4o_reference", family="openai", method_model="gpt-4o", runtime_model=os.environ.get("CARVE_OPENAI_JUDGE_MODEL", "gpt-4o")),
    ]


def compute_oracle_metrics(
    rows: list[dict],
    anchor_report: dict,
    alpha: float,
    factual_counterfactual_pairs: list[tuple[str, str]] | None = None,
) -> dict:
    retained_rows = [row for row in rows if not row.get("abstain", False)]
    retained_ids = {str(row.get("trace_id")) for row in retained_rows}
    retained_rate = len(retained_rows) / max(1, len(rows))
    abstention_rate = 1.0 - retained_rate
    score_reliability = max(0.0, 1.0 - float(anchor_report.get("expected_calibration_error", 0.0)))
    epsilon = float(anchor_report.get("bias_epsilon", anchor_report.get("expected_calibration_error", 0.0)))
    pair_rate = None
    if factual_counterfactual_pairs is not None:
        both = sum(1 for factual_id, cf_id in factual_counterfactual_pairs if factual_id in retained_ids and cf_id in retained_ids)
        pair_rate = both / max(1, len(factual_counterfactual_pairs))
    return {
        "coverage_target": 1.0 - alpha,
        "conformal_threshold": anchor_report.get("conformal_threshold"),
        "judge_human_corr": anchor_report.get("judge_human_corr", 0.0),
        "expected_calibration_error": anchor_report.get("expected_calibration_error", 0.0),
        "bias_epsilon": epsilon,
        "delta_bias_bound_2epsilon": 2.0 * epsilon,
        "retained": len(retained_rows),
        "abstention_rate": abstention_rate,
        "retained_set_score_reliability": score_reliability,
        "factual_counterfactual_both_retained_rate": pair_rate,
    }


def build_judge_committee(
    api_judges: bool,
    samples_per_judge: int,
    judge_names: str | None = None,
) -> tuple[JudgeCommittee, dict]:
    specs = default_method_judge_specs()
    names = [name.strip() for name in judge_names.split(",")] if judge_names else []
    names = [name for name in names if name]
    if api_judges:
        client = OpenAICompatibleClient(APIClientConfig.from_env())
        judge_specs = specs
        if names:
            judge_specs = [
                JudgeSpec(name=name, family=name.split("_", 1)[0], method_model=specs[min(idx, len(specs) - 1)].method_model, runtime_model=os.environ.get("CARVE_MODEL", "GLM-5.1"))
                for idx, name in enumerate(names)
            ]
        judges = [APIJudge(spec.name, client) for spec in judge_specs]
        manifest = {
            "judge_mode": "api",
            "judges": [judge.name for judge in judges],
            "families": [spec.family for spec in judge_specs],
            "method_default_models": [spec.method_model for spec in specs],
            "runtime_models": [spec.runtime_model for spec in judge_specs],
            "samples_per_judge": samples_per_judge,
            "api_fallback_reason": None,
        }
        return JudgeCommittee(judges, samples_per_judge=samples_per_judge), manifest

    judges = [DeterministicJudge("strict"), DeterministicJudge("lenient")]
    manifest = {
        "judge_mode": "deterministic",
        "judges": [judge.name for judge in judges],
        "families": [spec.family for spec in specs],
        "method_default_models": [spec.method_model for spec in specs],
        "runtime_models": [spec.runtime_model for spec in specs],
        "samples_per_judge": samples_per_judge,
        "api_fallback_reason": "api_judges_disabled",
    }
    return JudgeCommittee(judges, samples_per_judge=samples_per_judge), manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--anchor-path", default="data/openqa/anchors.jsonl")
    parser.add_argument("--samples-per-judge", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--api-judges", action="store_true", help="Use CARVE_API_KEY OpenAI-compatible judge calls instead of deterministic judges")
    parser.add_argument("--judge-names", default=None, help="Comma-separated judge names for API mode")
    args = parser.parse_args()

    run_dir = Path("artifacts/runs") / args.run_id
    trace_path = run_dir / "traces.jsonl"
    traces = [Trace.from_dict(json.loads(line)) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    anchor_path = Path(args.anchor_path)
    if not anchor_path.exists():
        write_default_anchor_file(anchor_path)
    anchors = load_anchor_scores(anchor_path)
    oracle = CalibratedOracle(alpha=args.alpha)
    oracle.fit(anchors.raw_scores, anchors.targets, anchors.dispersions)
    committee, judge_manifest = build_judge_committee(
        api_judges=args.api_judges,
        samples_per_judge=args.samples_per_judge,
        judge_names=args.judge_names,
    )

    rows = []
    retained = 0
    for trace in traces:
        scores = committee.score(trace.manifest.get("task", {}).get("prompt", trace.task_id), trace.final_answer)
        oracle_score = oracle.score(scores, calibration_version=anchor_path.stem)
        retained += int(not oracle_score.abstain)
        rows.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "raw_mean": oracle_score.raw_mean,
                "calibrated_score": oracle_score.calibrated_score,
                "committee_scores": oracle_score.committee_scores,
                "dispersion": oracle_score.dispersion,
                "abstain": oracle_score.abstain,
                "calibration_version": oracle_score.calibration_version,
            }
        )
    out = run_dir / "oracle_scores.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    try:
        anchor_report = calibration_report_from_anchors(read_jsonl(anchor_path), alpha=args.alpha)
    except Exception as exc:
        anchor_report = {"calibration_report_error": str(exc)}
    metrics = {
        "run_id": args.run_id,
        "num_traces": len(rows),
        "anchor_path": str(anchor_path),
        "output": str(out),
        **judge_manifest,
        **compute_oracle_metrics(rows, anchor_report, args.alpha),
    }
    metrics.update(anchor_report)
    (run_dir / "oracle_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
