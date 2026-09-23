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
    return bool(report.get("passed")) and report.get("controlled_traces") == 1


def build_command(offset: int, args: argparse.Namespace) -> list[str]:
    run_id = f"{args.run_prefix}{offset:04d}"
    return [
        sys.executable,
        "experiments/run_pilot.py",
        "--dataset", "gsm8k",
        "--offset", str(offset),
        "--limit", "1",
        "--run-id", run_id,
        "--planner-mode", "static",
        "--max-retries", str(args.max_retries),
        "--max-cost", str(args.max_cost),
        "--early-stop-threshold", str(args.early_stop_threshold),
        "--prompt-version", "gsm8k_stable_v1",
        "--operator-set", "gsm8k_v1",
        "--k", str(args.k),
        "--replay-mode", "behavior",
        "--all-compatible",
        "--top-m", str(args.top_m),
        "--operators-per-event", str(args.operators_per_event),
        "--event-selection", "type_stratified",
        "--api",
        "--api-replay",
        "--ppo-smoke",
        "--skip-student",
    ]


def iter_offsets(start: int, end: int, offsets_csv: str | None) -> list[int]:
    if offsets_csv:
        return [int(item.strip()) for item in offsets_csv.split(",") if item.strip()]
    return list(range(start, end))


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=1319)
    parser.add_argument("--offsets", default=None)
    parser.add_argument("--run-prefix", default="GSM8K/gsm8k_glm51_full_k3_top8_ops3_task")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--top-m", type=int, default=8)
    parser.add_argument("--operators-per-event", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--early-stop-threshold", type=float, default=0.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    args = parser.parse_args()

    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    summary = {"configuration": vars(args), "attempted": [], "skipped": [], "failed": {}, "passed": []}
    summary_path = Path("artifacts/campaigns/gsm8k_full_batch_summary.json")
    consecutive_failures = 0
    for offset in iter_offsets(args.start, args.end, args.offsets):
        run_id = f"{args.run_prefix}{offset:04d}"
        run_dir = Path("artifacts/runs") / run_id
        if args.skip_existing and run_passed(run_dir):
            summary["skipped"].append(run_id)
            write_summary(summary_path, summary)
            print(f"+ skip {run_id} already semantically validated", flush=True)
            continue
        command = build_command(offset, args)
        summary["attempted"].append(run_id)
        write_summary(summary_path, summary)
        print("+ " + " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            summary["failed"][run_id] = str(exc)
            consecutive_failures += 1
            write_summary(summary_path, summary)
            if args.fail_fast or consecutive_failures >= args.max_consecutive_failures:
                break
            continue
        consecutive_failures = 0
        if run_passed(run_dir):
            summary["passed"].append(run_id)
        else:
            summary["failed"][run_id] = "pipeline exited successfully but semantic validation did not pass"
        write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
