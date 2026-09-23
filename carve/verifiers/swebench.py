from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
import hashlib
import os
from pathlib import Path

from carve.schemas import Score


def _strip_markdown_fence(patch_text: str) -> str:
    lines = patch_text.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith(chr(96) * 3) and lines[-1].strip() == chr(96) * 3:
        return "\n".join(lines[1:-1]) + "\n"
    return patch_text


def _normalize_unified_diff(patch_text: str) -> str:
    lines = patch_text.splitlines()
    output = []
    index = 0
    hunk_re = re.compile(r"^(@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@.*)$")
    while index < len(lines):
        match = hunk_re.match(lines[index])
        if not match:
            output.append(lines[index])
            index += 1
            continue
        hunk_start = lines[index]
        body = []
        index += 1
        while index < len(lines) and not lines[index].startswith("@@ ") and not lines[index].startswith("--- "):
            body.append(" " if lines[index] == "" else lines[index])
            index += 1
        old_count = sum(line.startswith((" ", "-")) for line in body)
        new_count = sum(line.startswith((" ", "+")) for line in body)
        normalized = re.sub(r"^(@@ -\d+)(?:,\d+)?( \+\d+)(?:,\d+)?( @@.*)$", rf"\1,{old_count}\2,{new_count}\3", hunk_start)
        output.append(normalized)
        output.extend(body)
    return "\n".join(output) + ("\n" if patch_text.endswith("\n") else "")


def _strip_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if not (line.startswith("--- ") or line.startswith("+++ ")):
            continue
        raw = line[4:].strip().split("\t", 1)[0].split(" ", 1)[0]
        if raw == "/dev/null":
            continue
        path = _strip_prefix(raw)
        if path not in paths:
            paths.append(path)
    return paths


def _patch_source_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        if not line.startswith("--- "):
            continue
        raw = line[4:].strip().split("\t", 1)[0].split(" ", 1)[0]
        if raw != "/dev/null":
            paths.append(_strip_prefix(raw))
    return paths


def apply_patch_to_repo(repo_path: str | Path, patch_text: str) -> list[str]:
    repo = Path(repo_path)
    patch_text = _strip_markdown_fence(patch_text)
    patch_text = _normalize_unified_diff(patch_text)
    if not patch_text.strip():
        raise ValueError("patch did not contain any file changes")
    changed = _patch_paths(patch_text)
    if not changed:
        raise ValueError("patch did not contain any file changes")
    for relative in _patch_source_paths(patch_text):
        if not (repo / relative).exists():
            raise FileNotFoundError(relative)
    check = subprocess.run(
        ["git", "apply", "--check", "--recount", "--unsafe-paths", "-"],
        cwd=repo,
        input=patch_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check.returncode != 0:
        raise ValueError(check.stderr.strip() or "git apply --check failed")
    applied = subprocess.run(
        ["git", "apply", "--recount", "--unsafe-paths", "-"],
        cwd=repo,
        input=patch_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if applied.returncode != 0:
        raise ValueError(applied.stderr.strip() or "git apply failed")
    if not changed:
        raise ValueError("patch did not contain any file changes")
    return changed


class SWEBenchVerifier:
    def __init__(
        self,
        work_root: str | Path = "artifacts/swebench_work",
        timeout_s: float = 120.0,
        cleanup_success: bool = False,
        cleanup_failure: bool = False,
        reuse_worktrees: bool = False,
    ):
        self.work_root = Path(work_root)
        self.timeout_s = timeout_s
        self.cleanup_success = cleanup_success
        self.cleanup_failure = cleanup_failure
        self.reuse_worktrees = reuse_worktrees
        self._reusable_worktrees: dict[Path, Path] = {}

    @staticmethod
    def _create_work_dir(source: Path, work_dir: Path, base_commit: str | None) -> bool:
        source = source.resolve()
        work_dir = work_dir.resolve()
        if (source / ".git").exists():
            target = base_commit or "HEAD"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(work_dir), target],
                cwd=source,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return True
        shutil.copytree(source, work_dir)
        return False

    @staticmethod
    def _remove_work_dir(source: Path, work_dir: Path, is_worktree: bool) -> None:
        source = source.resolve()
        work_dir = work_dir.resolve()
        if not work_dir.exists():
            return
        if is_worktree:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(work_dir)],
                cwd=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        else:
            shutil.rmtree(work_dir)

    @staticmethod
    def _reset_worktree(work_dir: Path, base_commit: str | None) -> None:
        target = base_commit or "HEAD"
        subprocess.run(["git", "reset", "--hard", target], cwd=work_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "clean", "-fdx"], cwd=work_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _prepare_work_dir(self, source: Path, base_commit: str | None, work_dir: Path) -> tuple[Path, bool, bool]:
        source = source.resolve()
        if self.reuse_worktrees and (source / ".git").exists():
            reusable = self._reusable_worktrees.get(source)
            if reusable is None:
                reusable = (self.work_root / f"repo_{hashlib.sha1(str(source).encode()).hexdigest()[:12]}" ).resolve()
                if not reusable.exists():
                    self._create_work_dir(source, reusable, base_commit)
                self._reusable_worktrees[source] = reusable
            self._reset_worktree(reusable, base_commit)
            return reusable, True, True
        return work_dir, self._create_work_dir(source, work_dir, base_commit), False

    def close(self) -> None:
        for source, work_dir in list(self._reusable_worktrees.items()):
            self._remove_work_dir(source, work_dir, True)
        self._reusable_worktrees.clear()

    def verify(
        self,
        patch_text: str,
        tests: str | None = None,
        repo_path: str | None = None,
        base_commit: str | None = None,
        setup_patch: str | None = None,
    ) -> Score:
        if not repo_path:
            return Score(0.0, False, {"tests": tests, "mode": "repo_missing"}, stderr="repo_path is required")
        source = Path(repo_path)
        if not source.exists():
            return Score(0.0, False, {"tests": tests, "mode": "repo_missing", "repo_path": repo_path}, stderr="repo_path does not exist")
        if not tests or not str(tests).strip():
            return Score(0.0, False, {"tests": tests, "mode": "test_command_missing"}, stderr="test_command is required")
        self.work_root.mkdir(parents=True, exist_ok=True)
        work_dir = self.work_root / f"swebench_{int(time.time() * 1000)}"
        is_worktree = False
        persistent_worktree = False
        start = time.time()
        try:
            work_dir, is_worktree, persistent_worktree = self._prepare_work_dir(source, base_commit, work_dir)
            work_dir = work_dir.resolve()
            if setup_patch:
                apply_patch_to_repo(work_dir, setup_patch)
            changed = apply_patch_to_repo(work_dir, patch_text)
            command = self._normalize_test_command(str(tests))
            test_env = dict(os.environ)
            existing_pythonpath = test_env.get("PYTHONPATH")
            python_paths = [str(work_dir / "src")] if (work_dir / "src").is_dir() else []
            python_paths.append(str(work_dir))
            if existing_pythonpath:
                python_paths.append(existing_pythonpath)
            test_env["PYTHONPATH"] = os.pathsep.join(python_paths)
            completed = subprocess.run(
                command,
                cwd=work_dir,
                env=test_env,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
            )
            runtime_ms = (time.time() - start) * 1000.0
            success = completed.returncode == 0
            details = {
                "tests": command,
                "mode": "repo_test",
                "base_commit": base_commit,
                "returncode": completed.returncode,
                "changed_files": changed,
                "work_dir": str(work_dir) if (not success or not self.cleanup_success) else None,
                "cleaned_work_dir": bool((success or (not success and self.cleanup_failure)) and (self.cleanup_success or self.cleanup_failure)),
                "reused_worktree": persistent_worktree,
                "stdout": completed.stdout[-4000:],
            }
            if persistent_worktree:
                self._reset_worktree(work_dir, base_commit)
            elif success and self.cleanup_success:
                self._remove_work_dir(source, work_dir, is_worktree)
            elif not success and self.cleanup_failure:
                self._remove_work_dir(source, work_dir, is_worktree)
            return Score(
                score=1.0 if success else 0.0,
                success=success,
                details=details,
                stderr=completed.stderr[-4000:] if completed.stderr else None,
                runtime_ms=runtime_ms,
            )
        except Exception as exc:
            runtime_ms = (time.time() - start) * 1000.0
            if persistent_worktree:
                self._reset_worktree(work_dir, base_commit)
            elif self.cleanup_failure and work_dir.exists():
                self._remove_work_dir(source, work_dir, is_worktree)
            return Score(
                score=0.0,
                success=False,
                details={"tests": tests, "mode": "repo_test_error", "work_dir": str(work_dir), "error": type(exc).__name__},
                stderr=str(exc),
                runtime_ms=runtime_ms,
            )

    @staticmethod
    def _normalize_test_command(command: str) -> str:
        if command == "python" or command.startswith("python "):
            return sys.executable + command[len("python") :]
        return command
