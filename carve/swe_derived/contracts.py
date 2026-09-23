from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FORBIDDEN_ADAPTATIONS = {"algorithm_change", "assertion_weakening", "invented_behavior"}


def _required_string(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def validate_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} contains path traversal")
    if path == Path("."):
        raise ValueError(f"{field} must name a file or directory")
    return path.as_posix()


def _relative_paths(data: Mapping[str, Any], field: str) -> tuple[str, ...]:
    values = data.get(field)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} must be a non-empty list of relative paths")
    return tuple(validate_relative_path(value, field) for value in values)


def _resolve_case_path(case_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path.resolve()
    relative_path = validate_relative_path(value, field)
    return (case_root / relative_path).resolve()


def _fail_to_pass(data: Mapping[str, Any]) -> tuple[str, ...]:
    value = data.get("FAIL_TO_PASS")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("FAIL_TO_PASS must be a JSON list or list of test identifiers") from error
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("FAIL_TO_PASS must be a non-empty list of test identifiers")
    tests = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    if len(tests) != len(value):
        raise ValueError("FAIL_TO_PASS must contain only non-empty test identifiers")
    return tests


@dataclass(frozen=True)
class ExtractionRecipe:
    case_id: str
    source_instance_id: str
    source_repo_path: Path
    source_base_commit: str
    include_paths: tuple[str, ...]
    public_test_paths: tuple[str, ...]
    verifier_test_paths: tuple[str, ...]
    adaptations: tuple[dict[str, str], ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtractionRecipe":
        source_repo = _required_string(data, "source_repo_path")
        source_path = Path(source_repo)
        if not source_path.is_absolute():
            raise ValueError("source_repo_path must be an absolute path")

        adaptations = data.get("adaptations")
        if not isinstance(adaptations, (list, tuple)):
            raise ValueError("adaptations must be a list")
        validated_adaptations: list[dict[str, str]] = []
        for adaptation in adaptations:
            if not isinstance(adaptation, Mapping):
                raise ValueError("adaptations must contain dictionaries")
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in adaptation.items()):
                raise ValueError("adaptations must contain only string keys and values")
            kind = adaptation.get("kind")
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError("adaptations require a kind")
            kind = kind.strip().lower()
            if kind in FORBIDDEN_ADAPTATIONS:
                raise ValueError(f"forbidden adaptation kind: {kind}")
            if not isinstance(adaptation.get("reason"), str) or not adaptation["reason"].strip():
                raise ValueError("adaptations require a reason")
            normalized_adaptation = dict(adaptation)
            normalized_adaptation["kind"] = kind
            validated_adaptations.append(normalized_adaptation)

        return cls(
            case_id=_required_string(data, "case_id"),
            source_instance_id=_required_string(data, "source_instance_id"),
            source_repo_path=source_path.resolve(),
            source_base_commit=_required_string(data, "source_base_commit"),
            include_paths=_relative_paths(data, "include_paths"),
            public_test_paths=_relative_paths(data, "public_test_paths"),
            verifier_test_paths=_relative_paths(data, "verifier_test_paths"),
            adaptations=tuple(validated_adaptations),
        )


@dataclass(frozen=True)
class DerivedCase:
    case_id: str
    source_instance_id: str
    bug_family: str
    problem_statement: str
    repo_path: Path
    base_commit: str
    public_test_command: str
    hidden_test_command: str
    fail_to_pass: tuple[str, ...]
    gold_patch_path: Path
    timeout_seconds: int
    environment_id: str

    @classmethod
    def from_path(cls, path: str | Path) -> "DerivedCase":
        task_path = Path(path).resolve()
        if task_path.is_dir():
            task_path = task_path / "task.json"
        with task_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError("task.json must contain a JSON object")
        return cls._from_dict(data, task_path.parent)

    @classmethod
    def from_manifest_row(cls, data: Mapping[str, Any], manifest_dir: str | Path) -> "DerivedCase":
        case_path = data.get("case_path") or data.get("task_path")
        if case_path is not None:
            return cls.from_path(_resolve_case_path(Path(manifest_dir).resolve(), case_path, "case_path"))
        return cls._from_dict(data, Path(manifest_dir).resolve())

    @classmethod
    def _from_dict(cls, data: Mapping[str, Any], case_root: Path) -> "DerivedCase":
        timeout = data.get("timeout_seconds")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        return cls(
            case_id=_required_string(data, "case_id"),
            source_instance_id=_required_string(data, "source_instance_id"),
            bug_family=_required_string(data, "bug_family"),
            problem_statement=_required_string(data, "problem_statement"),
            repo_path=_resolve_case_path(case_root, data.get("repo_path"), "repo_path"),
            base_commit=_required_string(data, "base_commit"),
            public_test_command=_required_string(data, "public_test_command"),
            hidden_test_command=_required_string(data, "hidden_test_command"),
            fail_to_pass=_fail_to_pass(data),
            gold_patch_path=_resolve_case_path(case_root, data.get("gold_patch"), "gold_patch"),
            timeout_seconds=timeout,
            environment_id=_required_string(data, "environment_id"),
        )
