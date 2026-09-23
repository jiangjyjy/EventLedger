from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from carve.counterfactuals.operators import apply_operator
from carve.counterfactuals.replay import ReplayEngine
from carve.datasets.io import iter_jsonl
from carve.datasets.swebench import resolve_swebench_test_command, validate_prepared_swebench_row
from carve.schemas import Event, Trace
from carve.counterfactuals.swebench_operators import applicable_swebench_operators
from carve.verifiers.swebench import SWEBenchVerifier


def build_gold_patch_trace(row: dict, patch: str) -> Trace:
    trace_id = f"preflight::{row['instance_id']}"
    task_id = str(row["instance_id"])
    patch_event = Event(
        "gold-patch",
        trace_id,
        task_id,
        0,
        "msg",
        "patcher",
        "patcher-1",
        patch,
        metadata={"source": "gold_patch", "prompt_version": "swebench_v1"},
    )
    aggregate_event = Event(
        "gold-aggregate",
        trace_id,
        task_id,
        1,
        "aggregate",
        "aggregator",
        "aggregator-1",
        patch,
        ["gold-patch"],
        metadata={"source": "gold_patch", "prompt_version": "swebench_v1"},
    )
    return Trace(
        trace_id,
        task_id,
        "swebench_lite",
        "preflight",
        [patch_event, aggregate_event],
        patch,
        manifest={
            "workflow": "swebench",
            "prompt_version": "swebench_v1",
            "task": {
                "task_id": task_id,
                "dataset": "swebench_lite",
                "prompt": str(row.get("problem_statement", "")),
                "tests": resolve_swebench_test_command(row),
                "metadata": dict(row),
                "repo_path": row.get("repo_path"),
                "base_commit": row.get("base_commit"),
            },
        },
    )


def run_structural_replay(
    row: dict,
    patch: str,
    work_root: str | Path = "artifacts/swebench_preflight/work",
    verifier: SWEBenchVerifier | None = None,
) -> dict:
    trace = build_gold_patch_trace(row, patch)
    patch_operators = applicable_swebench_operators(patch)
    operator_name = patch_operators[0] if patch_operators else "delete"
    intervention = apply_operator(trace, "gold-patch", operator_name, __import__("random").Random(0), operator_set="swebench_v1")
    verifier = verifier or SWEBenchVerifier(work_root=work_root, cleanup_success=True, cleanup_failure=True, reuse_worktrees=True)

    def score(replayed: Trace, _seed: int) -> float:
        repo_path = row.get("repo_path")
        if not repo_path or not Path(repo_path).exists():
            return 0.0
        return verifier.verify(
            replayed.final_answer,
            resolve_swebench_test_command(row),
            repo_path=repo_path,
            base_commit=row.get("base_commit"),
            setup_patch=row.get("test_patch"),
        ).score

    result = ReplayEngine(score, behavior_policy="frozen_behavior_policy").replay(trace, intervention, seed=0)
    return {
        "replay_mode": result.metadata["replay_mode"],
        "api_calls": 0,
        "api_replay": False,
        "structural_score": result.score,
        "operator": intervention.operator_name,
        "metadata": result.metadata,
    }


def run_preflight(input_path: str | Path, output_dir: str | Path, timeout_s: float = 120.0) -> dict:
    rows = list(iter_jsonl(input_path))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "work"
    verifier = SWEBenchVerifier(
        work_root=work_root,
        timeout_s=timeout_s,
        cleanup_success=True,
        cleanup_failure=True,
        reuse_worktrees=True,
    )
    results_path = output_root / "results.jsonl"
    results: list[dict] = []
    for row_number, row in enumerate(rows):
        instance_id = str(row.get("instance_id", f"row-{row_number}"))
        started = time.time()
        try:
            validate_prepared_swebench_row(row, row_number=row_number)
            test_command = resolve_swebench_test_command(row)
            gold = verifier.verify(
                str(row["patch"]),
                test_command,
                repo_path=str(row["repo_path"]),
                base_commit=str(row["base_commit"]),
                setup_patch=row.get("test_patch"),
            )
            structural = run_structural_replay(row, str(row["patch"]), work_root=work_root, verifier=verifier)
            result = {
                "instance_id": instance_id,
                "status": "pass" if gold.success else "gold_patch_failed",
                "gold_patch": {"success": gold.success, "score": gold.score, "details": gold.details, "stderr": gold.stderr},
                "structural_replay": structural,
                "api_calls": 0,
                "elapsed_s": time.time() - started,
            }
        except Exception as exc:
            result = {
                "instance_id": instance_id,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "api_calls": 0,
                "elapsed_s": time.time() - started,
            }
        results.append(result)
    verifier.close()
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    summary = {
        "input": str(input_path),
        "output": str(output_root),
        "rows": len(results),
        "gold_patch_pass": sum(result.get("gold_patch", {}).get("success", False) for result in results),
        "gold_patch_failed": sum(result.get("status") == "gold_patch_failed" for result in results),
        "errors": sum(result.get("status") == "error" for result in results),
        "api_calls": 0,
        "replay_mode": "structural_event_replay",
        "results": str(results_path),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    print(json.dumps(run_preflight(args.input, args.output_dir, args.timeout_s), indent=2))


if __name__ == "__main__":
    main()
