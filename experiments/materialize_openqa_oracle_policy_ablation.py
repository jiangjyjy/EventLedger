"""Materialize Table 2 OpenQA oracle-policy ablations from aligned Table 4 scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def materialize(*, table4_path: Path, scores_path: Path, traces_path: Path, split_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    report = json.loads(table4_path.read_text(encoding="utf-8"))
    scores = _read_jsonl(scores_path)
    traces = _read_jsonl(traces_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    trace_truth = {str(row["task_id"]): bool(row.get("success")) for row in traces}
    score_truth = {str(row["task_id"]): bool(row["truth"]) for row in scores}
    test_ids = {str(task_id) for task_id in split["test"]}
    if len(trace_truth) != 100 or len(score_truth) != 100 or set(trace_truth) != set(score_truth):
        raise ValueError("Table 4 judge scores must align to exactly the formal OpenQA 100-task subset")
    if any(trace_truth[task_id] != score_truth[task_id] for task_id in trace_truth):
        raise ValueError("Table 4 truth labels do not match formal OpenQA factual traces")
    if len(test_ids) != 15 or not test_ids <= set(trace_truth):
        raise ValueError("formal OpenQA split must provide 15 aligned held-out tasks")
    metrics = report["metrics"]
    entries = (
        ("no_oracle_calibration", "Committee (no calibration)"),
        ("no_conformal_abstention", "+ Isotonic calibration"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for variant, metric_name in entries:
        metric = metrics[metric_name]
        result = {
            "regime": "openqa",
            "variant": variant,
            "status": "measured_oracle_policy",
            "subset_tasks": 100,
            "tasks": 15,
            "oracle_agreement": float(metric["accuracy"]),
            "verifier_score": float(metric["accuracy"]),
            "calibration": {"corr_rho": float(metric["corr_rho"]), "ece": float(metric["ece"]), "coverage": float(metric["coverage"]), "abstain_pct": float(metric["abstain_pct"])},
            "provenance": {"judge_scores": 100, "aligned_truth_labels": 100, "held_out_tasks": sorted(test_ids), "table4_metric": metric_name},
        }
        path = output_dir / f"{variant}_openqa_evaluation.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append(result)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table4", required=True, type=Path)
    parser.add_argument("--judge-scores", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(materialize(table4_path=args.table4, scores_path=args.judge_scores, traces_path=args.traces, split_path=args.split, output_dir=args.output_dir), indent=2))


if __name__ == "__main__":
    main()
