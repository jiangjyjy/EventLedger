from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from carve.scoring.calibration import CalibratedOracle
from carve.scoring.oracle import DeterministicJudge, JudgeCommittee
from carve.schemas import Trace


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0.0 or vy == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / math.sqrt(vx * vy)


def build_anchor_template(traces: list[Trace], committee: JudgeCommittee) -> list[dict[str, Any]]:
    rows = []
    for trace in traces:
        task_prompt = trace.manifest.get("task", {}).get("prompt", trace.task_id)
        scores = committee.score(task_prompt, trace.final_answer)
        raw = mean(scores)
        rows.append(
            {
                "trace_id": trace.trace_id,
                "task_id": trace.task_id,
                "task_prompt": task_prompt,
                "final_answer": trace.final_answer,
                "raw_score": raw,
                "dispersion": pstdev(scores) if len(scores) > 1 else 0.0,
                "target": None,
                "rubric_notes": "",
            }
        )
    return rows


def calibration_report_from_anchors(rows: list[dict[str, Any]], alpha: float = 0.1) -> dict[str, Any]:
    labeled = [row for row in rows if row.get("target") is not None]
    raw = [float(row["raw_score"]) for row in labeled]
    targets = [float(row["target"]) for row in labeled]
    dispersions = [float(row.get("dispersion", 0.0)) for row in labeled]
    if not labeled:
        return {
            "num_anchors": 0,
            "judge_human_corr": 0.0,
            "expected_calibration_error": 0.0,
            "conformal_threshold": math.inf,
            "retained_rate": 0.0,
        }
    oracle = CalibratedOracle(alpha=alpha)
    oracle.fit(raw, targets, dispersions)
    calibrated = oracle.iso.predict(raw)
    retained = [disp <= oracle.q for disp in dispersions]
    abs_biases = [abs(pred - target) for pred, target in zip(calibrated, targets, strict=True)]
    bias_epsilon = mean(abs_biases) if abs_biases else 0.0
    return {
        "num_anchors": len(labeled),
        "judge_human_corr": _pearson(raw, targets),
        "expected_calibration_error": CalibratedOracle.expected_calibration_error(calibrated, targets),
        "conformal_threshold": oracle.q,
        "coverage_target": 1.0 - alpha,
        "retained_rate": sum(int(v) for v in retained) / len(retained),
        "abstention_rate": 1.0 - sum(int(v) for v in retained) / len(retained),
        "bias_epsilon": bias_epsilon,
        "delta_bias_bound_2epsilon": 2.0 * bias_epsilon,
        "retained_set_score_reliability": 1.0 - CalibratedOracle.expected_calibration_error(
            [pred for pred, keep in zip(calibrated, retained, strict=True) if keep],
            [target for target, keep in zip(targets, retained, strict=True) if keep],
        )
        if any(retained)
        else 0.0,
        "raw_mean": mean(raw),
        "target_mean": mean(targets),
    }


def read_traces(path: Path) -> list[Trace]:
    return [Trace.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Run whose traces should become anchor candidates")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--out", default="data/openqa/anchor_template.jsonl")
    parser.add_argument("--labeled", default=None, help="Existing labeled anchor JSONL for calibration report")
    parser.add_argument("--report-out", default="data/openqa/calibration_report.json")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    if args.run_id or args.traces:
        trace_path = Path(args.traces) if args.traces else Path("artifacts/runs") / args.run_id / "traces.jsonl"
        committee = JudgeCommittee([DeterministicJudge("strict"), DeterministicJudge("lenient")], samples_per_judge=3)
        rows = build_anchor_template(read_traces(trace_path), committee)
        write_jsonl(Path(args.out), rows)
        print(json.dumps({"template": args.out, "rows": len(rows)}, indent=2))

    if args.labeled:
        rows = read_jsonl(Path(args.labeled))
        report = calibration_report_from_anchors(rows, alpha=args.alpha)
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"report": args.report_out, **report}, indent=2))


if __name__ == "__main__":
    main()
