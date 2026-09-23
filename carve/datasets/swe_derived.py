from __future__ import annotations

from pathlib import Path

from carve.schemas import Task
from carve.swe_derived.contracts import DerivedCase

from .io import iter_jsonl


def _task_from_case(case: DerivedCase) -> Task:
    metadata = {
        "case_id": case.case_id,
        "source_instance_id": case.source_instance_id,
        "bug_family": case.bug_family,
        "repo_path": str(case.repo_path),
        "base_commit": case.base_commit,
        "public_test_command": case.public_test_command,
        "timeout_seconds": case.timeout_seconds,
        "environment_id": case.environment_id,
    }
    return Task(
        task_id=case.case_id,
        dataset="swe_derived",
        prompt=case.problem_statement,
        tests=case.public_test_command,
        metadata=metadata,
    )


def load_swe_derived_jsonl(path: str | Path, limit: int | None = None, offset: int = 0) -> list[Task]:
    if limit is not None and limit <= 0:
        return []
    manifest_path = Path(path).resolve()
    tasks: list[Task] = []
    for row_index, row in enumerate(iter_jsonl(manifest_path)):
        if row_index < offset:
            continue
        case = DerivedCase.from_manifest_row(row, manifest_path.parent)
        tasks.append(_task_from_case(case))
        if limit is not None and len(tasks) >= limit:
            break
    return tasks
