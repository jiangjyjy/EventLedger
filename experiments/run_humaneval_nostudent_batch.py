from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_passed(run_dir: Path) -> bool:
    report_path = run_dir / "validation_report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return bool(report.get("passed")) and bool(manifest.get("skip_student"))


def build_command(offset: int, args: argparse.Namespace) -> list[str]:
    run_id = f"{args.run_prefix}{offset:03d}"
    cmd = [
        sys.executable,
        "experiments/run_pilot.py",
        "--dataset",
        "humaneval",
        "--offset",
        str(offset),
        "--limit",
        "1",
        "--run-id",
        run_id,
        "--planner-mode",
        "dynamic",
        "--max-retries",
        str(args.max_retries),
        "--max-cost",
        str(args.max_cost),
        "--early-stop-threshold",
        str(args.early_stop_threshold),
        "--prompt-version",
        args.prompt_version,
        "--k",
        str(args.k),
        "--replay-mode",
        args.replay_mode,
        "--all-compatible",
        "--top-m",
        str(args.top_m),
        "--operators-per-event",
        str(args.operators_per_event),
        "--event-selection",
        getattr(args, "event_selection", "top_m"),
        "--api",
        "--ppo-smoke",
        "--skip-student",
    ]
    if args.api_replay:
        cmd.append("--api-replay")
    return cmd


def iter_offsets(start: int, end: int, offsets_csv: str | None) -> list[int]:
    if offsets_csv:
        return [int(item.strip()) for item in offsets_csv.split(",") if item.strip()]
    return list(range(start, end))


def should_stop_after_failure(consecutive_failures: int, max_consecutive_failures: int) -> bool:
    return max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=164)
    parser.add_argument("--offsets", default=None, help="Comma-separated explicit offsets to run, overriding start/end")
    parser.add_argument("--run-prefix", default="humaneval_glm51_api_nostudent_task")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--replay-mode", choices=["structural", "behavior"], default="behavior")
    parser.add_argument("--api-replay", action="store_true", help="Use API model for counterfactual downstream replay")
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--operators-per-event", type=int, default=2)
    parser.add_argument("--event-selection", choices=["top_m", "random", "type_stratified"], default="top_m")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0)
    parser.add_argument("--prompt-version", default="default", choices=["default", "code_v2"])
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-consecutive-failures", type=int, default=0, help="Stop after N consecutive failed tasks; 0 disables")
    args = parser.parse_args()

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    summary = {"attempted": [], "skipped": [], "failed": {}, "passed": []}
    consecutive_failures = 0
    for offset in iter_offsets(args.start, args.end, args.offsets):
        run_id = f"{args.run_prefix}{offset:03d}"
        run_dir = Path("artifacts/runs") / run_id
        if args.skip_existing and run_passed(run_dir):
            print(f"+ skip {run_id} already validated", flush=True)
            summary["skipped"].append(run_id)
            consecutive_failures = 0
            continue
        cmd = build_command(offset, args)
        print("+ " + " ".join(cmd), flush=True)
        summary["attempted"].append(run_id)
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            summary["failed"][run_id] = str(exc)
            consecutive_failures += 1
            if args.fail_fast or should_stop_after_failure(consecutive_failures, args.max_consecutive_failures):
                break
            continue
        consecutive_failures = 0
        if run_passed(run_dir):
            summary["passed"].append(run_id)
    out_dir = Path("artifacts/campaigns")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.run_prefix.rstrip('_')}_batch_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
