from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from carve.schemas import Trace
from carve.scoring.credit_value import MAX_CONSERVATION_SCALE


REQUIRED_MANIFEST_KEYS = {"run_id", "dataset", "seed", "model", "operator_config"}


def operator_coverage_policy(manifest: dict[str, Any]) -> str:
    config = manifest.get("operator_config", {})
    explicit = config.get("coverage_policy")
    if explicit in {"strict", "sampled"}:
        return explicit
    if (
        config.get("k") == 1
        and config.get("top_m") in {5, 8}
        and config.get("operators_per_event") == 2
        and config.get("stop_counterfactual") is True
    ):
        return "sampled"
    return "strict"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_trace_graphs(traces: list[Trace]) -> tuple[int, int]:
    resolved = 0
    total = 0
    for trace in traces:
        seen: set[str] = set()
        for event in trace.events:
            for parent in event.parents:
                total += 1
                if parent in seen:
                    resolved += 1
            seen.add(event.event_id)
    return resolved, total


def validate_run_dir(run_dir: Path) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    missing_manifest = sorted(REQUIRED_MANIFEST_KEYS - set(manifest))
    if missing_manifest:
        issues.append(f"manifest missing keys: {missing_manifest}")

    traces = [Trace.from_dict(row) for row in read_jsonl(run_dir / "traces.jsonl")]
    if not traces:
        issues.append("no traces found")
    resolved, parent_total = validate_trace_graphs(traces)
    parent_rate = 1.0 if parent_total == 0 else resolved / parent_total
    if parent_rate < 0.95:
        issues.append(f"parent_resolved_rate below 0.95: {parent_rate}")

    credit_rows = read_jsonl(run_dir / "credit_labels.jsonl")
    if not credit_rows:
        issues.append("no credit labels found")
    non_abstained = [row for row in credit_rows if not row.get("abstained", False)]
    for row in non_abstained:
        if not finite(row.get("delta_mean", 0.0)):
            issues.append(f"non-finite delta_mean for {row.get('event_id')}")
        metadata = row.get("metadata", {})
        scale = metadata.get("conservation_scale")
        if scale is not None and (not finite(scale) or abs(float(scale)) > MAX_CONSERVATION_SCALE):
            issues.append(f"explosive conservation scale for {row.get('trace_id')}::{row.get('event_id')}: {scale}")
        for key in ("rescaled_delta", "adjusted_delta"):
            if key in metadata and not finite(metadata[key]):
                issues.append(f"non-finite {key} for {row.get('event_id')}")
        for replay in metadata.get("replays", []):
            if replay.get("prefix_valid") is False:
                issues.append(f"invalid replay prefix for {row.get('event_id')}")

    reward_rows = read_jsonl(run_dir / "reward_labels.jsonl")
    if reward_rows:
        for row in reward_rows:
            if not finite(row.get("total_reward")):
                issues.append(f"non-finite total_reward for {row.get('event_id')}")

    control_summary_path = run_dir / "control_summary.json"
    control_summary = json.loads(control_summary_path.read_text(encoding="utf-8")) if control_summary_path.exists() else {}
    controlled_rows = read_jsonl(run_dir / "controlled_traces.jsonl")
    expected_controlled = int(control_summary.get("controlled_traces", 0))
    if expected_controlled and not controlled_rows:
        issues.append("control summary exists but controlled_traces.jsonl is missing or empty")
    if controlled_rows and len(controlled_rows) != expected_controlled:
        issues.append(f"controlled trace count mismatch: expected {expected_controlled}, found {len(controlled_rows)}")
    for row in controlled_rows:
        if row.get("manifest", {}).get("control", {}).get("verified_after_pruning") is not True:
            issues.append(f"controlled trace not reverified: {row.get('trace_id')}")

    prs_path = run_dir / "prs_summary.json"
    prs = json.loads(prs_path.read_text(encoding="utf-8")) if prs_path.exists() else {}
    coverage = prs.get("prs", {}).get("operator_coverage")
    coverage_policy = operator_coverage_policy(manifest)
    coverage_complete = coverage.get("complete_for_compatible_events") if coverage else None
    coverage_missing = coverage.get("missing_compatible_operators", []) if coverage else []
    if coverage and coverage_complete is False:
        message = f"operator coverage incomplete: {coverage_missing}"
        if coverage_policy == "strict":
            issues.append(message)
        else:
            warnings.append(message)

    report = {
        "run_dir": str(run_dir),
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "traces": len(traces),
        "events": sum(len(trace.events) for trace in traces),
        "parent_edges": parent_total,
        "parent_resolved_rate": parent_rate,
        "credit_labels": len(credit_rows),
        "non_abstained_credit_labels": len(non_abstained),
        "reward_labels": len(reward_rows),
        "controlled_traces": len(controlled_rows),
        "operator_coverage_policy": coverage_policy,
        "operator_coverage_complete": coverage_complete,
        "operator_coverage_missing": coverage_missing,
        "manifest_missing_keys": missing_manifest,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path("artifacts/runs") / str(args.run_id)
    report = validate_run_dir(run_dir)
    if args.write:
        (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
