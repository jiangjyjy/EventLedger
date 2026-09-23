from __future__ import annotations

import os
import json
import re
import shlex
from pathlib import Path

from carve.schemas import Task

from .io import existing_path, iter_jsonl


REQUIRED_PREPARED_FIELDS = ("repo_path", "base_commit", "test_command", "patch")


def resolve_swebench_test_command(row: dict) -> str:
    """Return an explicit test command, deriving one from SWE-bench FAIL_TO_PASS."""
    for key in ("test_command", "tests", "test_cmd"):
        value = row.get(key)
        if value:
            return str(value).strip()
    raw = row.get("FAIL_TO_PASS")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = [raw]
    if isinstance(raw, (list, tuple)) and raw:
        tests = [str(item).strip() for item in raw if str(item).strip()]
        if tests:
            if row.get("repo") == "sympy/sympy":
                patch = str(row.get("test_patch") or "")
                file_match = re.search(r"diff --git a/([^\s]+)", patch)
                if file_match:
                    labels = " ".join(shlex.quote(test) for test in tests)
                    return f"python bin/test {shlex.quote(file_match.group(1))} -k {labels}"
            if row.get("repo") == "django/django":
                labels = []
                for test in tests:
                    match = re.match(r"^(.+?) \(([^()]+)\)$", test)
                    if match:
                        label = match.group(2)
                        if not label.endswith(f".{match.group(1)}"):
                            label = f"{label}.{match.group(1)}"
                        labels.append(label)
                        continue
                    patch = str(row.get("test_patch") or "")
                    file_match = re.search(r"diff --git a/tests/([^/\n]+)", patch)
                    if file_match:
                        labels.append(file_match.group(1))
                    else:
                        labels.append(test)
                return "python tests/runtests.py --parallel=1 " + " ".join(shlex.quote(label) for label in labels)
            return "python -m pytest -q " + " ".join(shlex.quote(test) for test in tests)
    raise ValueError("SWE-bench row is missing test_command and FAIL_TO_PASS")


def validate_prepared_swebench_row(row: dict, row_number: int | None = None) -> None:
    missing = [field for field in REQUIRED_PREPARED_FIELDS if not row.get(field)]
    if missing:
        where = f" at row {row_number}" if row_number is not None else ""
        raise ValueError(f"SWE-bench prepared row{where} missing required fields: {', '.join(missing)}")


def load_swebench_jsonl(path: str | Path, limit: int | None = None) -> list[Task]:
    tasks = []
    for row in iter_jsonl(path):
        task_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or f"swebench_lite_{len(tasks)}")
        prompt = row.get("problem_statement") or row.get("prompt") or row.get("issue") or ""
        tests = resolve_swebench_test_command(row)
        metadata = dict(row)
        metadata.setdefault("instance_id", task_id)
        metadata.setdefault("repo", row.get("repo"))
        metadata.setdefault("base_commit", row.get("base_commit"))
        if row.get("repo_path"):
            metadata["repo_path"] = row["repo_path"]
        if row.get("patch"):
            metadata["reference_patch"] = row["patch"]
        tasks.append(Task(task_id, "swebench_lite", prompt, tests=tests, metadata=metadata))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_swebench_sample(limit: int = 1) -> list[Task]:
    path = existing_path(os.environ.get("CARVE_SWEBENCH_PATH"), "data/raw/swebench_lite.jsonl", "data/raw/swebench.jsonl")
    if path:
        return load_swebench_jsonl(path, limit)
    return [
        Task(
            "swebench_lite_0",
            "swebench_lite",
            "Fix the failing test in a small repository issue.",
            tests="python -m unittest",
            metadata={"repo": "synthetic/repo", "issue": "failing edge case"},
        )
    ][:limit]
