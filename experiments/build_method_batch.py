from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


JSONL_FILES = [
    "traces.jsonl",
    "credit_labels.jsonl",
    "reward_labels.jsonl",
    "rl_samples.jsonl",
    "judge_reranking_scores.jsonl",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return 0
    with dst.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")
    return len(rows)


def clear_outputs(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in JSONL_FILES:
        path = run_dir / name
        if path.exists():
            path.unlink()
    for path in run_dir.glob("*.json"):
        path.unlink()
    for path in run_dir.glob("*.md"):
        path.unlink()


def build_batch(input_run_ids: list[str], output_run_id: str, runs_root: Path) -> dict[str, Any]:
    output_dir = runs_root / output_run_id
    clear_outputs(output_dir)
    included: list[str] = []
    skipped: dict[str, str] = {}
    validation_reports: dict[str, Any] = {}
    row_counts = {name: 0 for name in JSONL_FILES}

    for run_id in input_run_ids:
        run_dir = runs_root / run_id
        validation = read_json(run_dir / "validation_report.json")
        if not validation or not validation.get("passed"):
            skipped[run_id] = "validation_not_passed"
            continue
        included.append(run_id)
        validation_reports[run_id] = validation
        for name in JSONL_FILES:
            row_counts[name] += append_jsonl(run_dir / name, output_dir / name)

    if not included:
        raise ValueError("no validated input runs were included")

    first_manifest = read_json(runs_root / included[0] / "manifest.json") or {}
    manifest = {
        **first_manifest,
        "run_id": output_run_id,
        "source_run_ids": included,
        "skipped_source_run_ids": skipped,
        "batch_size": len(included),
        "dataset": "humaneval",
        "split": "method_batch",
        "model": "api",
        "api": True,
        "skip_student": False,
        "student_added_at_batch_level": True,
        "planner_mode": "dynamic",
        "operator_config": {
            **(first_manifest.get("operator_config") or {}),
            "replay_mode": "behavior",
            "use_crn": True,
            "reward_mode": "composed",
        },
        "method_alignment": {
            "oeg": "typed orchestration event graph with explicit state snapshots",
            "tco": "typed counterfactual operator labels from source runs",
            "prs": "behavior replay, CRN, top-m selected event labels",
            "reward": "composed reward with potential, grounding, redundancy, contradiction, cost, stop",
            "student": "CARVE-S trained on aggregated teacher reward labels",
            "deployment": "reranking/pruning/early-stop/control and PPO/GRPO export",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    aggregate = {
        "output_run_id": output_run_id,
        "included_run_ids": included,
        "skipped_run_ids": skipped,
        "row_counts": row_counts,
        "validation_reports": validation_reports,
        "source_root": str(runs_root),
    }
    (output_dir / "method_batch_sources.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run-id", action="append", default=[])
    parser.add_argument("--input-run-ids-file", default=None)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--runs-root", default="artifacts/runs")
    args = parser.parse_args()

    run_ids = list(args.input_run_id)
    if args.input_run_ids_file:
        rows = json.loads(Path(args.input_run_ids_file).read_text(encoding="utf-8"))
        run_ids.extend(rows)
    if not run_ids:
        raise ValueError("provide --input-run-id or --input-run-ids-file")
    result = build_batch(run_ids, args.output_run_id, Path(args.runs_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
