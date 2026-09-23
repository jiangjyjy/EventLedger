from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .candidates import touched_paths


def _patch(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        return ""
    return value


def _test_paths(patch: str) -> list[str]:
    return sorted({path for path in touched_paths(patch) if path.endswith(".py") and "/test" in f"/{path}"})


def _source_paths(patch: str, test_paths: list[str]) -> list[str]:
    return sorted(
        {
            path.split("/", 1)[0] if "/" in path else path
            for path in touched_paths(patch)
            if path not in test_paths
        }
    )


def derive_strict_recipe(
    row: Mapping[str, Any], *, source_repo_path: Path
) -> tuple[dict[str, Any] | None, str | None]:
    instance_id = row.get("instance_id")
    base_commit = row.get("base_commit")
    if not isinstance(instance_id, str) or not instance_id:
        return None, "missing_instance_id"
    if not isinstance(base_commit, str) or not base_commit:
        return None, "missing_base_commit"

    test_paths = _test_paths(_patch(row, "test_patch"))
    if len(test_paths) < 2:
        return None, "insufficient_distinct_python_test_files"
    source_paths = _source_paths(_patch(row, "patch"), test_paths)
    if not source_paths:
        return None, "gold_patch_has_no_non_test_source_files"

    public_test = test_paths[:1]
    verifier_tests = test_paths[1:]
    return {
        "case_id": instance_id,
        "source_instance_id": instance_id,
        "source_repo_path": str(source_repo_path.resolve()),
        "source_base_commit": base_commit,
        "include_paths": source_paths,
        "public_test_paths": public_test,
        "verifier_test_paths": verifier_tests,
        "public_test_command": "python -m pytest -q " + " ".join(public_test),
        "hidden_test_command": "python -m pytest -q " + " ".join(
            f"verifier_tests/{path}" for path in verifier_tests
        ),
        "bug_family": "swe_derived",
        "environment_id": "python-3.12",
        "adaptations": [
            {
                "kind": "dependency_closure",
                "reason": "preserve files directly touched by the source patch",
            }
        ],
    }, None
