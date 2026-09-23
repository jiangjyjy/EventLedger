from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import shlex
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .contracts import DerivedCase, ExtractionRecipe, validate_relative_path


_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_REGULAR_MODES = {"100644": 0o644, "100755": 0o755}
_SYMLINK_MODE = "120000"


@dataclass(frozen=True)
class _GitFile:
    source_path: str
    git_mode: str
    content: bytes


@dataclass(frozen=True)
class _PlannedFile(_GitFile):
    output_path: str
    visibility: str


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, input=input_bytes, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(message or f"git command failed: {' '.join(args)}") from error


def _resolve_commit(repo: Path, commit: str) -> str:
    if not _FULL_COMMIT.fullmatch(commit):
        raise ValueError("source_base_commit must be a full commit id")
    resolved = _git(repo, "rev-parse", f"{commit}^{{commit}}").decode().strip()
    if resolved.lower() != commit.lower():
        raise ValueError("source_base_commit must resolve to the exact full commit id")
    return resolved


def _read_git_file(repo: Path, commit: str, relative: str) -> _GitFile:
    safe = validate_relative_path(relative, "source path")
    raw_tree = _git(
        repo,
        "--literal-pathspecs",
        "ls-tree",
        "-z",
        "--full-tree",
        commit,
        "--",
        safe,
    )
    entries = [entry for entry in raw_tree.split(b"\0") if entry]
    if len(entries) != 1:
        raise ValueError(f"source path must name one exact git entry: {safe}")
    metadata, separator, raw_path = entries[0].partition(b"\t")
    if not separator or os.fsdecode(raw_path) != safe:
        raise ValueError(f"source path must name one exact git entry: {safe}")
    try:
        mode, object_type, object_id = metadata.decode("ascii").split()
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid git tree metadata for source path: {safe}") from error
    if object_type != "blob" or mode not in {*_REGULAR_MODES, _SYMLINK_MODE}:
        raise ValueError(f"unsupported git mode {mode} for source path: {safe}")
    content = _git(repo, "cat-file", "blob", object_id)
    if mode == _SYMLINK_MODE and (not content or b"\0" in content):
        raise ValueError(f"unsupported symlink target for source path: {safe}")
    return _GitFile(safe, mode, content)


def _expand_git_path(repo: Path, commit: str, relative: str) -> tuple[str, ...]:
    safe = validate_relative_path(relative, "source path")
    raw = _git(repo, "--literal-pathspecs", "ls-tree", "-r", "-z", "--name-only", commit, "--", safe)
    paths = tuple(os.fsdecode(path) for path in raw.split(b"\0") if path)
    if not paths:
        raise ValueError(f"source path must name one exact git entry: {safe}")
    return paths


def _read_worktree_file(repo: Path, relative: str) -> _GitFile:
    safe = validate_relative_path(relative, "source path")
    path = repo / safe
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"source path must name one exact worktree entry: {safe}") from error
    if stat.S_ISREG(metadata.st_mode):
        git_mode = "100755" if metadata.st_mode & stat.S_IXUSR else "100644"
        return _GitFile(safe, git_mode, path.read_bytes())
    if stat.S_ISLNK(metadata.st_mode):
        content = os.fsencode(os.readlink(path))
        if not content or b"\0" in content:
            raise ValueError(f"unsupported symlink target for source path: {safe}")
        return _GitFile(safe, _SYMLINK_MODE, content)
    raise ValueError(f"unsupported worktree entry for source path: {safe}")


def read_at_commit(repo: Path, commit: str, relative: str) -> bytes:
    resolved = _resolve_commit(repo, commit)
    return _read_git_file(repo, resolved, relative).content


def _write(path: Path, planned: _PlannedFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if planned.git_mode == _SYMLINK_MODE:
        path.symlink_to(os.fsdecode(planned.content))
        return
    path.write_bytes(planned.content)
    path.chmod(_REGULAR_MODES[planned.git_mode])


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _initialize_repo(repo: Path, instance_id: str) -> str:
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo, "-c", "user.email=anonymous@example.com",
        "-c", "user.name=CARVE SWE Derived", "commit", "-q", "-m",
        f"buggy base from {instance_id}",
    )
    return _git(repo, "rev-parse", "HEAD").decode().strip()


def _required(row: Mapping[str, Any], field: str) -> Any:
    value = row.get(field)
    if value is None or value == "":
        raise ValueError(f"source row missing {field}")
    return value


def _paths_overlap(left: str, right: str) -> bool:
    left_parts = Path(left).parts
    right_parts = Path(right).parts
    common = min(len(left_parts), len(right_parts))
    return left_parts[:common] == right_parts[:common]


def _reject_output_collisions(paths: tuple[str, ...], root: str) -> None:
    for index, path in enumerate(paths):
        for previous in paths[:index]:
            if _paths_overlap(path, previous):
                raise ValueError(f"output collision under {root}: {path} overlaps {previous}")


def _plan_files(
    recipe: ExtractionRecipe,
    commit: str,
    *,
    source_repo: Path | None = None,
    include_paths: tuple[str, ...] | None = None,
    public_test_paths: tuple[str, ...] | None = None,
    verifier_test_paths: tuple[str, ...] | None = None,
    read_file: Callable[[str], _GitFile] | None = None,
) -> list[_PlannedFile]:
    source_repo = source_repo or recipe.source_repo_path
    include_paths = recipe.include_paths if include_paths is None else include_paths
    public_test_paths = recipe.public_test_paths if public_test_paths is None else public_test_paths
    verifier_test_paths = recipe.verifier_test_paths if verifier_test_paths is None else verifier_test_paths
    read_file = read_file or (lambda relative: _read_git_file(source_repo, commit, relative))
    agent_paths = (*include_paths, *public_test_paths)
    verifier_paths = verifier_test_paths
    _reject_output_collisions(agent_paths, "repo")
    _reject_output_collisions(verifier_paths, "verifier_tests")
    for agent_path in agent_paths:
        for verifier_path in verifier_paths:
            if _paths_overlap(agent_path, verifier_path):
                raise ValueError(
                    "agent-visible and verifier paths overlap: "
                    f"{agent_path} overlaps {verifier_path}"
                )

    expanded_include_paths = tuple(
        path
        for relative in include_paths
        for path in _expand_git_path(source_repo, commit, relative)
    )
    planned: list[_PlannedFile] = []
    for relative in (*expanded_include_paths, *public_test_paths):
        source = read_file(relative)
        planned.append(
            _PlannedFile(
                source_path=source.source_path,
                git_mode=source.git_mode,
                content=source.content,
                output_path=f"repo/{relative}",
                visibility="agent",
            )
        )
    for relative in verifier_paths:
        source = read_file(relative)
        planned.append(
            _PlannedFile(
                source_path=source.source_path,
                git_mode=source.git_mode,
                content=source.content,
                output_path=f"verifier_tests/{relative}",
                visibility="verifier",
            )
        )
    for item in planned:
        if item.git_mode != _SYMLINK_MODE:
            continue
        target = os.fsdecode(item.content)
        root = item.output_path.split("/", 1)[0]
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(item.output_path), target))
        if target.startswith("/") or not resolved.startswith(root + "/"):
            raise ValueError(f"symlink target escapes {root}: {item.source_path} -> {target}")
    return planned


def _plan_test_patch_files(
    recipe: ExtractionRecipe, commit: str, test_patch: str
) -> list[_PlannedFile]:
    with tempfile.TemporaryDirectory(prefix="carve-swe-derived-test-patch-") as raw:
        tree = Path(raw) / "tree"
        _git(recipe.source_repo_path, "worktree", "add", "--detach", str(tree), commit)
        try:
            applied = subprocess.run(
                ["git", "apply", "--check", "--recount", "-"],
                cwd=tree,
                input=test_patch,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if applied.returncode:
                raise ValueError(applied.stderr.strip() or "test_patch cannot apply")
            subprocess.run(
                ["git", "apply", "--recount", "-"], cwd=tree, input=test_patch,
                text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            planned = _plan_files(
                recipe,
                commit,
                include_paths=(),
                read_file=lambda relative: _read_worktree_file(tree, relative),
            )
        finally:
            _git(recipe.source_repo_path, "worktree", "remove", "--force", str(tree))
    return planned


def _validate_hidden_test_command(command: str, planned_files: list[_PlannedFile]) -> None:
    available = {item.output_path for item in planned_files if item.visibility == "verifier"}
    referenced = {
        token.split("::", 1)[0]
        for token in shlex.split(command)
        if token.split("::", 1)[0].startswith("verifier_tests/")
    }
    if not referenced or not referenced <= available:
        missing = sorted(referenced - available)
        raise ValueError(
            "hidden_test_command must reference preserved verifier paths"
            + (f": {missing}" if missing else "")
        )


def build_case(
    recipe: ExtractionRecipe,
    source_row: Mapping[str, Any],
    output_root: str | Path,
) -> DerivedCase:
    if source_row.get("instance_id") != recipe.source_instance_id:
        raise ValueError("source row instance_id does not match recipe")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", recipe.case_id):
        raise ValueError("case_id must be a simple directory name")

    root = Path(output_root).resolve()
    case_root = root / recipe.case_id
    if case_root.exists():
        raise ValueError(f"case already exists: {case_root}")
    commit = _resolve_commit(recipe.source_repo_path, recipe.source_base_commit)
    if source_row.get("test_patch"):
        planned_files = _plan_files(
            recipe, commit, public_test_paths=(), verifier_test_paths=()
        )
        planned_files.extend(
            _plan_test_patch_files(recipe, commit, str(source_row["test_patch"]))
        )
    else:
        planned_files = _plan_files(recipe, commit)

    gold_patch = str(_required(source_row, "patch"))
    gold_patch_bytes = gold_patch.encode("utf-8")
    fail_to_pass = source_row.get("FAIL_TO_PASS")
    if isinstance(fail_to_pass, str):
        fail_to_pass = json.loads(fail_to_pass)
    task_values = {
        "bug_family": str(_required(source_row, "bug_family")),
        "problem_statement": str(_required(source_row, "problem_statement")),
        "public_test_command": str(_required(source_row, "public_test_command")),
        "hidden_test_command": str(_required(source_row, "hidden_test_command")),
        "FAIL_TO_PASS": fail_to_pass,
        "timeout_seconds": int(source_row.get("timeout_seconds", 30)),
        "environment_id": str(source_row.get("environment_id", "python-3.10")),
    }
    _validate_hidden_test_command(task_values["hidden_test_command"], planned_files)

    case_root.mkdir(parents=True)
    repo = case_root / "repo"
    repo.mkdir()
    (case_root / "verifier_tests").mkdir()

    files: list[dict[str, str]] = []
    for planned in planned_files:
        output = case_root / planned.output_path
        _write(output, planned)
        digest = _sha256(planned.content)
        files.append({
            "source_path": planned.source_path,
            "output_path": planned.output_path,
            "source_sha256": digest,
            "output_sha256": digest,
            "visibility": planned.visibility,
            "git_mode": planned.git_mode,
        })

    base_commit = _initialize_repo(repo, recipe.source_instance_id)
    (case_root / "gold.patch").write_bytes(gold_patch_bytes)

    task = {
        "case_id": recipe.case_id,
        "source_instance_id": recipe.source_instance_id,
        "repo_path": str(repo.resolve()),
        "base_commit": base_commit,
        "gold_patch": "gold.patch",
        **task_values,
    }
    (case_root / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    audit = {
        "case_id": recipe.case_id,
        "source": {
            "instance_id": recipe.source_instance_id,
            "repo_path": str(recipe.source_repo_path),
            "base_commit": recipe.source_base_commit,
        },
        "adaptations": list(recipe.adaptations),
        "files": files,
        "gold_patch": {
            "output_path": "gold.patch",
            "sha256": _sha256(gold_patch_bytes),
            "source_field": "patch",
        },
    }
    if source_row.get("test_patch"):
        audit["test_patch"] = {
            "sha256": _sha256(str(source_row["test_patch"]).encode("utf-8")),
            "source_field": "test_patch",
        }
    (case_root / "adaptation_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return DerivedCase.from_path(case_root)
