from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


DEPENDENCY_RISK_PATH_PARTS = {
    "environment.yaml",
    "environment.yml",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
    "tox.ini",
}


@dataclass(frozen=True)
class CandidateScore:
    instance_id: str
    repo: str
    touched_files: int
    changed_lines: int
    python_only: bool
    dependency_risk: int


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _patch_text(row: Mapping[str, Any]) -> str:
    value = row.get("patch", "")
    if not isinstance(value, str):
        raise ValueError("patch must be a string")
    return value


def _path_from_diff_header(line: str) -> str | None:
    parts = line.split()
    if len(parts) < 4 or parts[0] != "diff" or parts[1] != "--git":
        return None
    target = parts[3]
    if target.startswith("b/"):
        target = target[2:]
    return target.strip()


def touched_paths(patch: str) -> tuple[str, ...]:
    paths: list[str] = []
    seen: set[str] = set()
    for line in patch.splitlines():
        path = _path_from_diff_header(line)
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return tuple(paths)


def changed_line_count(patch: str) -> int:
    count = 0
    for line in patch.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith(("+", "-")):
            count += 1
    return count


def _is_python_path(path: str) -> bool:
    return path.endswith(".py")


def _path_dependency_risk(path: str) -> int:
    normalized = path.lower()
    name = normalized.rsplit("/", 1)[-1]
    if name in DEPENDENCY_RISK_PATH_PARTS:
        return 2
    if normalized.startswith("requirements/") or "/requirements/" in normalized:
        return 2
    if f"/{normalized}".find("/migrations/") >= 0:
        return 1
    return 0


def dependency_risk(paths: Iterable[str]) -> int:
    return max((_path_dependency_risk(path) for path in paths), default=0)


def score_candidate(row: Mapping[str, Any]) -> CandidateScore:
    patch = _patch_text(row)
    paths = touched_paths(patch)
    return CandidateScore(
        instance_id=_required_string(row, "instance_id"),
        repo=_required_string(row, "repo"),
        touched_files=len(paths),
        changed_lines=changed_line_count(patch),
        python_only=bool(paths) and all(_is_python_path(path) for path in paths),
        dependency_risk=dependency_risk(paths),
    )


def rank_candidates(rows: Iterable[Mapping[str, Any]]) -> list[CandidateScore]:
    scored = [score_candidate(row) for row in rows]
    seen_instance_ids: set[str] = set()
    for candidate in scored:
        if candidate.instance_id in seen_instance_ids:
            raise ValueError(f"duplicate instance_id: {candidate.instance_id}")
        seen_instance_ids.add(candidate.instance_id)
    return sorted(
        scored,
        key=lambda candidate: (
            not candidate.python_only,
            candidate.dependency_risk,
            candidate.touched_files,
            candidate.changed_lines,
            candidate.instance_id,
        ),
    )


def candidate_to_dict(candidate: CandidateScore) -> dict[str, Any]:
    return asdict(candidate)
