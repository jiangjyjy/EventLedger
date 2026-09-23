from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiments.make_paper_tables import build_paper_tables, render_markdown_tables
from experiments.plan_paper_campaign import build_campaign_plan


def build_run_command(run_spec: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/run_pilot.py",
        "--dataset",
        run_spec["dataset"],
        "--limit",
        str(run_spec["limit"]),
        "--run-id",
        run_spec["run_id"],
        "--seed",
        str(run_spec.get("seed", 0)),
        "--k",
        str(run_spec.get("k", 1)),
        "--all-compatible",
        "--top-m",
        str(run_spec.get("top_m", 3)),
        "--operators-per-event",
        str(run_spec.get("operators_per_event", 1)),
        "--planner-mode",
        run_spec.get("planner_mode", "static"),
        "--max-retries",
        str(run_spec.get("max_retries", 1)),
        "--max-cost",
        str(run_spec.get("max_cost", 10.0)),
        "--prompt-version",
        run_spec.get("prompt_version", "default"),
        "--replay-mode",
        run_spec.get("replay_mode", "behavior"),
        "--event-selection",
        run_spec.get("event_selection", "top_m"),
    ]
    if run_spec.get("api"):
        cmd.append("--api")
    if run_spec.get("api_replay"):
        cmd.append("--api-replay")
    if run_spec.get("oracle"):
        cmd.append("--oracle")
    if run_spec.get("api_judges"):
        cmd.append("--api-judges")
    if run_spec.get("ppo_smoke"):
        cmd.append("--ppo-smoke")
    if run_spec.get("skip_student"):
        cmd.append("--skip-student")
    return cmd


def run_command(cmd: list[str]) -> None:
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def campaign_run_status(run_spec: dict[str, Any], runs_root: Path) -> str:
    run_dir = runs_root / run_spec["run_id"]
    if not run_dir.exists():
        return "missing"
    report_path = run_dir / "validation_report.json"
    if not report_path.exists():
        return "incomplete"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "invalid"
    return "passed" if report.get("passed") else "failed"


def execute_campaign_runs(
    plan: dict[str, Any],
    runs_root: Path,
    skip_existing: bool = False,
    fail_fast: bool = True,
    command_runner=run_command,
) -> dict[str, Any]:
    result = {
        "attempted": [],
        "skipped": [],
        "failed": {},
        "statuses_before": {},
    }
    for run_spec in plan["runs"]:
        run_id = run_spec["run_id"]
        status = campaign_run_status(run_spec, runs_root)
        result["statuses_before"][run_id] = status
        if skip_existing and status == "passed":
            print(f"+ skip {run_id} already validated", flush=True)
            result["skipped"].append(run_id)
            continue
        cmd = build_run_command(run_spec)
        result["attempted"].append(run_id)
        try:
            command_runner(cmd)
        except Exception as exc:
            result["failed"][run_id] = str(exc)
            if fail_fast:
                raise
    return result


def summarize_campaign(plan: dict[str, Any], runs_root: Path, output_dir: Path) -> dict[str, Any]:
    run_dirs = [runs_root / run["run_id"] for run in plan["runs"] if (runs_root / run["run_id"]).exists()]
    validations = {}
    for run in plan["runs"]:
        report_path = runs_root / run["run_id"] / "validation_report.json"
        if report_path.exists():
            validations[run["run_id"]] = json.loads(report_path.read_text(encoding="utf-8"))
    tables = build_paper_tables(run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    (output_dir / "paper_tables.md").write_text(render_markdown_tables(tables), encoding="utf-8")
    summary = {
        "campaign_id": plan["campaign_id"],
        "scale": plan["scale"],
        "runs_total": len(plan["runs"]),
        "runs_found": len(run_dirs),
        "validations": validations,
        "validations_passed": sum(1 for report in validations.values() if report.get("passed")),
        "paper_tables": tables,
        "paper_tables_dir": str(output_dir),
    }
    (output_dir / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--scale", choices=["smoke", "formal_pilot", "full"], default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs with validation_report.json marked passed")
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True, help="Stop on first failed run")
    parser.add_argument("--runs-root", default="artifacts/runs")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    else:
        if not args.campaign_id:
            raise ValueError("--campaign-id is required when --plan is not provided")
        plan = build_campaign_plan(args.campaign_id, scale=args.scale, seed=args.seed)

    commands = [build_run_command(run) for run in plan["runs"]]
    if args.dry_run:
        for cmd in commands:
            print("+ " + " ".join(cmd))
    else:
        run_result = execute_campaign_runs(
            plan,
            runs_root=Path(args.runs_root),
            skip_existing=args.skip_existing,
            fail_fast=args.fail_fast,
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/campaigns") / plan["campaign_id"]
    summary = summarize_campaign(plan, Path(args.runs_root), output_dir)
    if not args.dry_run:
        summary["execution"] = run_result
        (output_dir / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"campaign_id": plan["campaign_id"], "runs": len(commands), "output_dir": str(output_dir)}, indent=2))
    if not args.dry_run and summary["validations_passed"] != summary["runs_total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
