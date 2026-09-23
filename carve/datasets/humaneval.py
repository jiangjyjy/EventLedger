from __future__ import annotations

import os
from pathlib import Path

from carve.schemas import Task
from .io import existing_path, iter_jsonl


def load_humaneval_jsonl(path: str | Path, limit: int | None = None, offset: int = 0) -> list[Task]:
    tasks = []
    for row_idx, row in enumerate(iter_jsonl(path)):
        if row_idx < offset:
            continue
        task_id = str(row.get("task_id") or row.get("id") or f"humaneval_{len(tasks)}")
        prompt = row.get("prompt") or row.get("question") or row.get("text") or ""
        reference = row.get("canonical_solution") or row.get("reference")
        tests = row.get("test") or row.get("tests")
        tasks.append(Task(task_id, "humaneval", prompt, reference=reference, tests=tests, metadata=row))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_humaneval_sample(limit: int = 2, offset: int = 0) -> list[Task]:
    path = existing_path(os.environ.get("CARVE_HUMANEVAL_PATH"), "data/raw/humaneval.jsonl", "data/raw/HumanEval.jsonl")
    if path:
        return load_humaneval_jsonl(path, limit, offset=offset)
    tasks = [
        Task("humaneval_0", "humaneval", "Write a function add(a, b) that returns a + b.", tests="assert add(2, 3) == 5"),
        Task("humaneval_1", "humaneval", "Write a function is_even(n) that returns whether n is even.", tests="assert is_even(4)"),
    ]
    return tasks[offset : offset + limit]
