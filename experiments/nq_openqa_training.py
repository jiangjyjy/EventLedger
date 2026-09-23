from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path

from carve.student_lora.data import TaskSplit, make_task_split


OPENQA_TRAIN_TASKS = 70
OPENQA_VALIDATION_TASKS = 15
OPENQA_TEST_TASKS = 15


def pre_generation_action_prompt(question: str, retrieved_evidence: str, router_assignment: str) -> str:
    """Build the action-policy input from state available before either Reader runs."""
    return (
        "Choose an OpenQA execution route. Return exactly A, B, or F.\n"
        "A: run Reader A only. B: run Reader B only. F: run both Readers and factual selector.\n\n"
        f"Question:\n{question}\n\nRetrieved evidence:\n{retrieved_evidence}\n\n"
        f"Router assignment:\n{router_assignment}\n\nAction:"
    )


def serializable_config(values: dict[object, object]) -> dict[object, object]:
    """Convert command-line Path values before persisting an experiment config."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def make_openqa_split(task_ids: Iterable[str], seed: int) -> TaskSplit:
    """Create the fixed, reproducible 70/15/15 NQ-open partition."""
    unique_task_ids = sorted(set(task_ids))
    expected = OPENQA_TRAIN_TASKS + OPENQA_VALIDATION_TASKS + OPENQA_TEST_TASKS
    if len(unique_task_ids) != expected:
        raise ValueError(f"OpenQA training requires exactly {expected} unique tasks, got {len(unique_task_ids)}")
    return make_task_split(
        unique_task_ids,
        seed=seed,
        train_count=OPENQA_TRAIN_TASKS,
        val_count=OPENQA_VALIDATION_TASKS,
        test_count=OPENQA_TEST_TASKS,
    )


def select_split_traces(traces: Sequence[object], split: TaskSplit, partition: str) -> list[object]:
    task_ids = getattr(split, partition, None)
    if task_ids is None:
        raise ValueError(f"unsupported OpenQA partition: {partition}")
    trace_by_task = {trace.task_id: trace for trace in traces}
    missing = [task_id for task_id in task_ids if task_id not in trace_by_task]
    if missing:
        raise ValueError(f"OpenQA traces are missing {partition} tasks: {missing[:3]}")
    return [trace_by_task[task_id] for task_id in task_ids]


def write_openqa_split(source_run: Path, output_dir: Path, seed: int) -> Path:
    trace_path = source_run / "traces.jsonl"
    task_ids = [json.loads(line)["task_id"] for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    split = make_openqa_split(task_ids, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "split.json"
    path.write_text(
        json.dumps({"train": list(split.train), "validation": list(split.validation), "test": list(split.test)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
