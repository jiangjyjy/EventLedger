from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple


PILOT_SEED_FILES = ("traces.jsonl", "credit_labels.jsonl", "prs_summary.json")


class Shard(NamedTuple):
    index: int
    offset: int
    limit: int
    run_id: str


def partition_shards(total: int, count: int, run_prefix: str) -> list[Shard]:
    if total <= 0 or count <= 0 or count > total:
        raise ValueError("total and count must define non-empty shards")
    base, remainder = divmod(total, count)
    shards = []
    offset = 0
    for index in range(count):
        limit = base + (1 if index < remainder else 0)
        end = offset + limit - 1
        run_id = f"{run_prefix}{index:02d}_{offset:03d}_{end:03d}"
        shards.append(Shard(index, offset, limit, run_id))
        offset += limit
    return shards


def seed_first_shard(pilot_dir: Path, shard_dir: Path) -> None:
    missing = [name for name in PILOT_SEED_FILES if not (pilot_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"pilot is missing resumable artifacts: {missing}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    for name in PILOT_SEED_FILES:
        destination = shard_dir / name
        if not destination.exists():
            shutil.copy2(pilot_dir / name, destination)


def trace_command(shard: Shard) -> list[str]:
    return [
        sys.executable,
        "experiments/run_trace_collection.py",
        "--dataset", "mbpp",
        "--limit", str(shard.limit),
        "--offset", str(shard.offset),
        "--run-id", shard.run_id,
        "--seed", "0",
        "--api",
        "--planner-mode", "dynamic",
        "--max-retries", "1",
        "--max-cost", "10.0",
        "--early-stop-threshold", "0.0",
        "--token-cost", "0.00001",
        "--prompt-version", "mbpp_v1",
    ]


def counterfactual_command(shard: Shard) -> list[str]:
    return [
        sys.executable,
        "experiments/run_counterfactuals.py",
        "--run-id", shard.run_id,
        "--operator-set", "mbpp_v1",
        "--all-compatible",
        "--top-m", "8",
        "--primary-events", "5",
        "--operators-per-event", "2",
        "--tail-operators-per-event", "1",
        "--event-selection", "type_stratified",
        "--k", "1",
        "--replay-mode", "behavior",
        "--api-replay",
        "--replay-timeout-seconds", "240",
        "--resume",
    ]


def jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def process_alive(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return False
    return Path(f"/proc/{pid}").exists()


def wait_for_pilot(pilot_dir: Path, expected_traces: int, poll_seconds: int) -> None:
    pid_path = pilot_dir / "pilot20_pipeline.pid"
    while process_alive(pid_path):
        progress = read_json(pilot_dir / "credit_progress.json")
        print(
            json.dumps(
                {
                    "stage": "waiting_for_pilot",
                    "traces": jsonl_count(pilot_dir / "traces.jsonl"),
                    "credit_labels": jsonl_count(pilot_dir / "credit_labels.jsonl"),
                    "credit_status": progress.get("status"),
                    "processed_traces": progress.get("processed_traces"),
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)
    if jsonl_count(pilot_dir / "traces.jsonl") != expected_traces:
        raise RuntimeError("pilot trace collection did not complete")
    if read_json(pilot_dir / "credit_progress.json").get("status") != "completed":
        raise RuntimeError("pilot counterfactual stage did not complete")
    if jsonl_count(pilot_dir / "controlled_traces.jsonl") != expected_traces:
        raise RuntimeError("pilot control stage did not complete")


def run_stage(command: list[str], env: dict[str, str], attempts: int, retry_delay: int) -> None:
    for attempt in range(1, attempts + 1):
        print("+ " + " ".join(command), flush=True)
        completed = subprocess.run(command, env=env)
        if completed.returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(completed.returncode, command)
        print(f"stage failed with exit {completed.returncode}; retry {attempt}/{attempts} after {retry_delay}s", flush=True)
        time.sleep(retry_delay)



def run_counterfactual_stage(command: list[str], env: dict[str, str], run_dir: Path, attempts: int, retry_delay: int) -> None:
    watchdog_script = [
        sys.executable,
        "experiments/checkpoint_watchdog.py",
        "--run-dir",
        str(run_dir),
        "--checkpoint-name",
        "credit_jobs.jsonl",
        "--log-path",
        str(run_dir / "checkpoint_watchdog.log"),
        "--idle-timeout-seconds",
        os.environ.get("CARVE_WATCHDOG_IDLE_TIMEOUT_SECONDS", "1800"),
    ]
    for attempt in range(1, attempts + 1):
        print("+ " + " ".join(command), flush=True)
        worker = subprocess.Popen(command, env=env)
        watchdog = subprocess.Popen(
            [*watchdog_script, "--pid", str(worker.pid)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        returncode = worker.wait()
        watchdog.wait()
        if returncode == 0:
            return
        if attempt == attempts:
            raise subprocess.CalledProcessError(returncode, command)
        print(f"stage failed with exit {returncode}; retry {attempt}/{attempts} after {retry_delay}s", flush=True)
        time.sleep(retry_delay)

def update_manifest(run_dir: Path, shard: Shard) -> None:
    path = run_dir / "manifest.json"
    manifest = read_json(path)
    manifest.update(
        {
            "run_id": shard.run_id,
            "dataset": "mbpp",
            "model": "api",
            "api": True,
            "api_replay": True,
            "offset": shard.offset,
            "limit": shard.limit,
            "seed": 0,
            "planner_mode": "dynamic",
            "prompt_version": "mbpp_v1",
            "skip_student": True,
            "operator_config": {
                "operator_set": "mbpp_v1",
                "k": 1,
                "all_compatible": True,
                "top_m": 8,
                "primary_events": 5,
                "operators_per_event": 2,
                "tail_operators_per_event": 1,
                "use_crn": True,
                "stop_counterfactual": True,
                "event_selection": "type_stratified",
                "reward_mode": "composed",
                "replay_mode": "behavior",
                "api_replay": True,
                "coverage_policy": "sampled",
            },
            "stages": [
                "trace_collection",
                "counterfactuals",
                "rewards",
                "control",
                "baselines",
                "judge_reranking",
                "analyze",
            ],
        }
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validation_passed(run_dir: Path, expected_traces: int) -> bool:
    report = read_json(run_dir / "validation_report.json")
    return (
        report.get("passed") is True
        and report.get("traces") == expected_traces
        and report.get("controlled_traces") == expected_traces
    )


def downstream_commands(shard: Shard) -> list[list[str]]:
    run = ["--run-id", shard.run_id]
    return [
        [sys.executable, "experiments/run_rewards.py", *run, "--reward-mode", "composed"],
        [sys.executable, "experiments/run_control.py", *run, "--reward-source", "teacher_rewards"],
        [sys.executable, "experiments/run_baselines.py", *run],
        [sys.executable, "experiments/run_judge_reranking.py", *run],
        [sys.executable, "experiments/analyze.py", *run],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-run-id", default="MBPP/mbpp_v1_api_pilot20_seed0")
    parser.add_argument("--run-prefix", default="MBPP/mbpp_v1_k1_top8_op2_tail1_shard")
    parser.add_argument("--total", type=int, default=427)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--stage-attempts", type=int, default=6)
    parser.add_argument("--retry-delay", type=int, default=1800)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    runs_root = root / "artifacts" / "runs"
    pilot_dir = runs_root / args.pilot_run_id
    shards = partition_shards(args.total, args.shards, args.run_prefix)

    wait_for_pilot(pilot_dir, args.pilot_size, args.poll_seconds)
    pilot = Shard(-1, 0, args.pilot_size, args.pilot_run_id)
    update_manifest(pilot_dir, pilot)
    run_stage([sys.executable, "experiments/validate_run.py", "--run-id", args.pilot_run_id, "--write"], env, 1, 0)

    for shard in shards:
        run_dir = runs_root / shard.run_id
        if validation_passed(run_dir, shard.limit):
            print(f"+ skip {shard.run_id}: already validated", flush=True)
            continue
        if shard.index == 0:
            seed_first_shard(pilot_dir, run_dir)
        run_stage(trace_command(shard), env, args.stage_attempts, args.retry_delay)
        actual_traces = jsonl_count(run_dir / "traces.jsonl")
        if actual_traces != shard.limit:
            raise RuntimeError(f"{shard.run_id}: expected {shard.limit} traces, found {actual_traces}")
        run_counterfactual_stage(counterfactual_command(shard), env, run_dir, args.stage_attempts, args.retry_delay)
        for command in downstream_commands(shard):
            run_stage(command, env, 1, 0)
        update_manifest(run_dir, shard)
        run_stage([sys.executable, "experiments/validate_run.py", "--run-id", shard.run_id, "--write"], env, 1, 0)
        if not validation_passed(run_dir, shard.limit):
            raise RuntimeError(f"{shard.run_id}: semantic validation did not pass")
    print(json.dumps({"status": "completed", "shards": [shard._asdict() for shard in shards]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
