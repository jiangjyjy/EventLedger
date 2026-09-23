from __future__ import annotations

import argparse
import os
import signal
import time
from pathlib import Path


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def log_message(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}\n")


def terminate_process(pid: int, grace_seconds: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and process_alive(pid):
        time.sleep(1)
    if process_alive(pid):
        os.kill(pid, signal.SIGKILL)


def watch(
    run_dir: Path,
    pid: int,
    idle_timeout_seconds: int,
    poll_seconds: int,
    checkpoint_name: str,
    log_path: Path,
) -> int:
    checkpoint = run_dir / checkpoint_name
    last_count = line_count(checkpoint)
    idle_seconds = 0
    log_message(log_path, f"START pid={pid} jobs={last_count} idle_timeout={idle_timeout_seconds}s")
    while process_alive(pid):
        time.sleep(poll_seconds)
        current_count = line_count(checkpoint)
        if current_count > last_count:
            last_count = current_count
            idle_seconds = 0
        else:
            idle_seconds += poll_seconds
        log_message(log_path, f"CHECK pid={pid} jobs={current_count} idle={idle_seconds}s")
        if idle_seconds >= idle_timeout_seconds:
            log_message(log_path, f"STOP no_new_job_labels_for_{idle_timeout_seconds}s pid={pid} jobs={current_count}")
            terminate_process(pid, grace_seconds=min(30, max(5, poll_seconds)))
            return 2
    log_message(log_path, f"EXIT pid={pid} jobs={line_count(checkpoint)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--checkpoint-name", default="credit_jobs.jsonl")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--idle-timeout-seconds",
        type=int,
        default=int(os.environ.get("CARVE_WATCHDOG_IDLE_TIMEOUT_SECONDS", "1800")),
    )
    args = parser.parse_args()
    log_path = args.log_path or (args.run_dir / "checkpoint_watchdog.log")
    return watch(
        args.run_dir,
        args.pid,
        args.idle_timeout_seconds,
        args.poll_seconds,
        args.checkpoint_name,
        log_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
