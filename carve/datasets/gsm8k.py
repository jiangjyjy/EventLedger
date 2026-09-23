from __future__ import annotations

import os
import re
from pathlib import Path

from carve.schemas import Task
from .io import existing_path, iter_jsonl


def extract_gsm8k_reference(answer: str | None) -> str | None:
    if not answer:
        return None
    if "####" in answer:
        answer = answer.split("####")[-1]
    matches = re.findall(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
    return matches[-1] if matches else answer.strip()


def load_gsm8k_jsonl(path: str | Path, limit: int | None = None, offset: int = 0) -> list[Task]:
    tasks = []
    for row_idx, row in enumerate(iter_jsonl(path)):
        if row_idx < offset:
            continue
        task_id = str(row.get("task_id") or row.get("id") or f"gsm8k_{row_idx}")
        prompt = row.get("question") or row.get("prompt") or row.get("text") or ""
        reference = extract_gsm8k_reference(row.get("answer") or row.get("reference"))
        tasks.append(Task(task_id, "gsm8k", prompt, reference=reference, tests=None, metadata=row))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_gsm8k_sample(limit: int = 2, offset: int = 0) -> list[Task]:
    path = existing_path(os.environ.get("CARVE_GSM8K_PATH"), "data/raw/gsm8k.jsonl", "data/raw/gsm8k_test.jsonl")
    if path:
        return load_gsm8k_jsonl(path, limit, offset=offset)
    tasks = [
        Task("gsm8k_0", "gsm8k", "If Ana has 40 apples and buys 2 more, how many apples?", reference="42"),
        Task("gsm8k_1", "gsm8k", "Tom has 10 books and gets 5. How many books?", reference="15"),
    ]
    return tasks[offset : offset + limit]
