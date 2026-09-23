from __future__ import annotations

import io
import os
import shlex
import shutil
import site
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath

from carve.schemas import Score
from carve.swe_derived.contracts import DerivedCase

from .swebench import _normalize_unified_diff, _strip_markdown_fence


_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "XDG_RUNTIME_DIR",
    "DBUS_SESSION_BUS_ADDRESS",
)
_PRIVATE_MARKERS = ("verifier_tests",)


def _redact_public_output(text: str, case: DerivedCase) -> str:
    redacted = text
    private_values = [
        *_PRIVATE_MARKERS,
        str(case.gold_patch_path),
        *case.fail_to_pass,
    ]
    for value in private_values:
        redacted = redacted.replace(value, "[private]")
    return redacted


def _header_path(raw: str) -> str | None:
    values = shlex.split(raw)
    if not values or values[0] == "/dev/null":
        return None
    path = values[0]
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _validated_patch_paths(repo: Path, patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        candidates: list[str | None] = []
        if line.startswith("diff --git "):
            values = shlex.split(line)
            if len(values) != 4:
                raise ValueError("invalid diff header")
            candidates.extend((_header_path(values[2]), _header_path(values[3])))
        elif line.startswith(("--- ", "+++ ")):
            candidates.append(_header_path(line[4:]))
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            candidates.append(_header_path(line.split(" ", 2)[2]))
        for raw in candidates:
            if raw is None:
                continue
            relative = PurePosixPath(raw)
            if not raw or relative.is_absolute() or ".." in relative.parts:
                raise ValueError("patch path escapes sandbox")
            if relative.parts[0] == "verifier_tests":
                raise ValueError("patch targets reserved private path")
            resolved = (repo / Path(*relative.parts)).resolve(strict=False)
            if not resolved.is_relative_to(repo):
                raise ValueError("patch path escapes sandbox")
            normalized = relative.as_posix()
            if normalized not in paths:
                paths.append(normalized)
    if not paths:
        raise ValueError("patch did not contain any file changes")
    return paths


def _apply_contained_patch(repo: Path, patch_text: str) -> list[str]:
    normalized = _normalize_unified_diff(_strip_markdown_fence(patch_text))
    if not normalized.strip():
        raise ValueError("patch did not contain any file changes")
    paths = _validated_patch_paths(repo, normalized)
    for arguments in (
        ["git", "apply", "--check", "--recount", "-"],
        ["git", "apply", "--recount", "-"],
    ):
        completed = subprocess.run(
            arguments,
            cwd=repo,
            input=normalized,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or "git apply failed")
    return paths


def _export_commit(repo: Path, commit: str, destination: Path) -> None:
    archived = subprocess.run(
        ["git", "archive", "--format=tar", commit],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
        _extract_archive(archive, destination)


def _archive_target(root: Path, raw: str) -> Path:
    relative = PurePosixPath(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("archive path escapes sandbox")
    target = root.joinpath(*relative.parts)
    if not target.parent.resolve(strict=False).is_relative_to(root):
        raise ValueError("archive path escapes sandbox")
    return target


def _extract_archive(archive: tarfile.TarFile, destination: Path) -> None:
    directory_modes: list[tuple[Path, int]] = []
    for member in archive.getmembers():
        target = _archive_target(destination, member.name)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            directory_modes.append((target, member.mode & 0o777))
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ValueError("archive contains duplicate path")
        if member.isreg():
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("archive regular file has no content")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
        elif member.issym():
            link = PurePosixPath(member.linkname)
            if link.is_absolute():
                raise ValueError("archive symlink escapes sandbox")
            resolved = target.parent.joinpath(*link.parts).resolve(strict=False)
            if not resolved.is_relative_to(destination):
                raise ValueError("archive symlink escapes sandbox")
            target.symlink_to(member.linkname)
        elif member.islnk():
            source = _archive_target(destination, member.linkname).resolve(strict=True)
            if not source.is_relative_to(destination):
                raise ValueError("archive hardlink escapes sandbox")
            target.hardlink_to(source)
        else:
            raise ValueError("archive contains unsupported file type")
    for directory, mode in reversed(directory_modes):
        directory.chmod(mode)


def _runtime_read_only_paths() -> list[Path]:
    candidates = [
        Path(path)
        for path in ("/usr", "/bin", "/lib", "/lib64", "/etc", sys.base_prefix)
    ]
    candidates.extend(Path(path) for path in site.getsitepackages())
    candidates.append(Path(site.getusersitepackages()))
    result: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved not in result:
            result.append(resolved)
    return result


class SWEDerivedVerifier:
    def __init__(self, work_root: str | Path = "/tmp/carve-swe-derived-work"):
        self.work_root = Path(work_root).resolve()

    def verify_public(self, patch_text: str, case: DerivedCase) -> Score:
        return self._verify(patch_text, case, hidden=False)

    def verify_hidden(self, patch_text: str, case: DerivedCase) -> Score:
        raw = self._verify(patch_text, case, hidden=True)
        return Score(
            score=raw.score,
            success=raw.success,
            details={
                "mode": "hidden_verifier",
                "returncode": raw.details.get("returncode"),
                "changed_files": raw.details.get("changed_files", []),
                "tests_passed": raw.success,
            },
            stderr=None,
            runtime_ms=raw.runtime_ms,
        )

    def _verify(self, patch_text: str, case: DerivedCase, *, hidden: bool) -> Score:
        mode = "hidden_verifier" if hidden else "public_verifier"
        error_mode = "hidden_verifier_error" if hidden else "public_verifier_error"
        started = time.perf_counter()
        self.work_root.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        work_dir = self.work_root / f"{case.case_id}-{token}"
        result: Score
        try:
            _export_commit(case.repo_path, case.base_commit, work_dir)
            changed = _apply_contained_patch(work_dir, patch_text)
            if hidden:
                source = case.gold_patch_path.parent / "verifier_tests"
                shutil.copytree(source, work_dir / "verifier_tests", symlinks=True)

            command = self._command(
                case.hidden_test_command if hidden else case.public_test_command
            )
            unit = f"carve-swe-derived-{token}.service"
            sandbox = Path(__file__).with_name("_sandbox_exec.py").resolve()
            sandbox_command = [
                "systemd-run",
                "--user",
                "--wait",
                "--pipe",
                "--collect",
                f"--unit={unit}",
                f"--property=RuntimeMaxSec={case.timeout_seconds}s",
                "--property=KillMode=control-group",
                sys.executable,
                str(sandbox),
                "--work-dir",
                str(work_dir),
            ]
            for path in _runtime_read_only_paths():
                sandbox_command.extend(("--read-only", str(path)))
            sandbox_command.extend(("--", *command))
            environment = {
                key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ
            }
            completed = subprocess.run(
                sandbox_command,
                cwd=work_dir,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=case.timeout_seconds + 15,
            )
            combined = completed.stdout + completed.stderr
            timed_out = "Finished with result: timeout" in combined
            success = completed.returncode == 0 and not timed_out
            details = {
                "mode": error_mode if timed_out else mode,
                "returncode": completed.returncode,
                "changed_files": changed,
                "tests_passed": success,
            }
            if not hidden:
                details["stdout"] = _redact_public_output(completed.stdout[-4000:], case)
            result = Score(
                1.0 if success else 0.0,
                success,
                details,
                stderr=(
                    None
                    if hidden
                    else (
                        f"verification timeout after {case.timeout_seconds} seconds"
                        if timed_out
                        else (_redact_public_output(completed.stderr[-4000:], case) or None)
                    )
                ),
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(
                ["systemctl", "--user", "stop", f"carve-swe-derived-{token}.service"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            result = Score(
                0.0,
                False,
                {"mode": error_mode, "error": "TimeoutExpired"},
                stderr=f"verification timeout after {case.timeout_seconds} seconds",
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as error:
            result = Score(
                0.0,
                False,
                {"mode": error_mode, "error": type(error).__name__},
                stderr=None if hidden else _redact_public_output(str(error), case),
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )

        try:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            if work_dir.exists():
                raise OSError("sandbox cleanup incomplete")
        except Exception as error:
            result = Score(
                0.0,
                False,
                {"mode": error_mode, "error": "CleanupError"},
                stderr=None if hidden else _redact_public_output(str(error), case),
                runtime_ms=(time.perf_counter() - started) * 1000.0,
            )
        return result

    @staticmethod
    def _command(raw: str) -> list[str]:
        command = shlex.split(raw)
        if not command:
            raise ValueError("test command is empty")
        if command[0] == "python":
            command[0] = sys.executable
        if Path(command[0]).resolve() != Path(sys.executable).resolve():
            raise ValueError("test command must use the verifier Python interpreter")
        return command
