from __future__ import annotations

import os
from pathlib import Path

from carve.schemas import Task
from .io import existing_path, iter_jsonl


def load_mbpp_jsonl(path: str | Path, limit: int | None = None, offset: int = 0) -> list[Task]:
    tasks = []
    for row_idx, row in enumerate(iter_jsonl(path)):
        if row_idx < offset:
            continue
        task_id = str(row.get("task_id") or row.get("id") or row.get("problem_id") or f"mbpp_{row_idx}")
        prompt = row.get("text") or row.get("prompt") or row.get("question") or ""
        reference = row.get("code") or row.get("canonical_solution") or row.get("reference")
        tests_raw = row.get("test_list") or row.get("tests") or row.get("test")
        tests = "\n".join(tests_raw) if isinstance(tests_raw, list) else tests_raw
        tasks.append(Task(task_id, "mbpp", prompt, reference=reference, tests=tests, metadata=row))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_mbpp_sample(limit: int = 2, offset: int = 0) -> list[Task]:
    path = existing_path(os.environ.get("CARVE_MBPP_PATH"), "data/raw/mbpp.jsonl", "data/raw/mbpp_test.jsonl")
    if path:
        return load_mbpp_jsonl(path, limit, offset=offset)
    tasks = [
        Task("mbpp_0", "mbpp", "Write a Python function to square a number.", tests="assert square(5) == 25"),
        Task("mbpp_1", "mbpp", "Write a Python function to reverse a string.", tests="assert reverse('ab') == 'ba'"),
    ]
    return tasks[offset : offset + limit]
