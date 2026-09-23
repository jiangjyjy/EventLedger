from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMES = {"paired_structural_shapley", "selector_direct_delta"}
SPIDER_DECISION_ROLES = {"planner", "sql_writer_a", "sql_writer_b", "selector"}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_split(
    task_ids: list[str],
    *,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    success_by_task: dict[str, bool] | None = None,
) -> dict[str, list[str]]:
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Spider bundle must contain exactly one trace per task")
    if train_count + validation_count + test_count != len(task_ids):
        raise ValueError("split counts must partition every source task")
    names = ("train", "validation", "test")
    capacities = (train_count, validation_count, test_count)
    rng = random.Random(seed)
    if success_by_task is None:
        ordered = sorted(task_ids)
        rng.shuffle(ordered)
        train_end = train_count
        validation_end = train_end + validation_count
        return dict(zip(names, (ordered[:train_end], ordered[train_end:validation_end], ordered[validation_end:])))

    failures = sorted(task_id for task_id in task_ids if not success_by_task[task_id])
    successes = sorted(task_id for task_id in task_ids if success_by_task[task_id])
    rng.shuffle(failures)
    rng.shuffle(successes)
    raw = [len(failures) * capacity / len(task_ids) for capacity in capacities]
    failure_counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda value: (raw[value] - failure_counts[value], -value), reverse=True)[: len(failures) - sum(failure_counts)]:
        failure_counts[index] += 1
    result: dict[str, list[str]] = {}
    failure_start = 0
    success_start = 0
    for name, capacity, failure_count in zip(names, capacities, failure_counts):
        success_count = capacity - failure_count
        members = failures[failure_start : failure_start + failure_count] + successes[success_start : success_start + success_count]
        rng.shuffle(members)
        result[name] = members
        failure_start += failure_count
        success_start += success_count
    return result


def prepare_distillation(
    source_bundle: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    train_count: int = 70,
    validation_count: int = 15,
    test_count: int = 15,
) -> dict[str, Any]:
    source_bundle = source_bundle.resolve()
    traces_path = source_bundle / "traces.jsonl"
    labels_path = source_bundle / "credit_labels.jsonl"
    traces = _rows(traces_path)
    labels = _rows(labels_path)
    trace_events = {trace["trace_id"]: {event["event_id"] for event in trace["events"]} for trace in traces}
    if len(trace_events) != len(traces):
        raise ValueError("duplicate trace_id in source bundle")
    task_ids = [trace["task_id"] for trace in traces]
    success_by_task = {trace["task_id"]: bool(trace.get("success")) for trace in traces}
    trainable_factual_events = sum(
        1
        for trace in traces
        for event in trace["events"]
        if trace.get("dataset") != "spider" or event.get("agent_role") in SPIDER_DECISION_ROLES
    )
    schemes: Counter[str] = Counter()
    event_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for label in labels:
        trace_id = label.get("trace_id")
        event_id = label.get("event_id")
        if trace_id not in trace_events or event_id not in trace_events[trace_id]:
            raise ValueError(f"label references missing trace/event: {trace_id}/{event_id}")
        if label.get("abstained"):
            raise ValueError("source training bundle must exclude abstained labels")
        scheme = (label.get("metadata") or {}).get("credit_scheme")
        if scheme not in SUPPORTED_SCHEMES:
            raise ValueError(f"unsupported credit scheme: {scheme}")
        score = label.get("delta_mean")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"invalid delta_mean for {trace_id}/{event_id}")
        schemes[scheme] += 1
        event_scores[(trace_id, event_id)].append(float(score))

    split = _task_split(
        task_ids,
        seed=seed,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        success_by_task=success_by_task,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_bundle": str(source_bundle),
        "source_hashes": {"traces.jsonl": _sha256(traces_path), "credit_labels.jsonl": _sha256(labels_path)},
        "seed": seed,
        "traces": len(traces),
        "tasks": len(task_ids),
        "credit_labels": len(labels),
        "event_credit_targets": len(event_scores),
        "trainable_factual_events": trainable_factual_events,
        "credit_schemes": dict(sorted(schemes.items())),
        "split_counts": {name: len(ids) for name, ids in split.items()},
        "factual_success": sum(bool(trace.get("success")) for trace in traces),
        "factual_failure": sum(not bool(trace.get("success")) for trace in traces),
    }
    (output_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    command = (
        "python experiments/run_student_lora.py "
        f"--source-run {source_bundle} --split-file {output_dir / 'split.json'} "
        "--model-path <LOCAL_QWEN_MODEL_PATH> "
        "--output-dir <NEW_STUDENT_OUTPUT_DIR> --device cuda:<ID> "
        "--epochs 1 --max-event-tokens 512 --event-micro-batch-size 2 "
        "--hidden-dim 256 --lora-r 16 --lora-alpha 32 --ranking-beta 0.2 --huber-delta 1.0\n"
    )
    (output_dir / "train_command.txt").write_text(command, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and prepare Spider DAG data for LoRA distillation without starting training")
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=15)
    args = parser.parse_args()
    print(json.dumps(prepare_distillation(args.source_bundle, args.output_dir, seed=args.seed, train_count=args.train_count, validation_count=args.validation_count, test_count=args.test_count), sort_keys=True))


if __name__ == "__main__":
    main()
