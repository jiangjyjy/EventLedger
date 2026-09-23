from __future__ import annotations

import argparse
import ctypes
import errno
import os
import platform
import sys
from pathlib import Path


_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_BASE = (1 << 13) - 1
_FS_READ_ONLY = _FS_EXECUTE | _FS_READ_FILE | _FS_READ_DIR

_NET_BIND_TCP = 1 << 0
_NET_CONNECT_TCP = 1 << 1

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


def _syscall(libc: ctypes.CDLL, number: int, *args: object) -> int:
    result = libc.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _landlock_access(abi: int) -> int:
    access = _FS_BASE
    if abi >= 2:
        access |= _FS_REFER
    if abi >= 3:
        access |= _FS_TRUNCATE
    return access


def _add_path_rule(
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: Path,
    allowed_access: int,
) -> None:
    path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathBeneathAttr(allowed_access=allowed_access, parent_fd=path_fd)
        _syscall(
            libc,
            _LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            0,
        )
    finally:
        os.close(path_fd)


def _restrict_filesystem(work_dir: Path, read_only_paths: list[Path]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    abi = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET,
        0,
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    handled_fs = _landlock_access(abi)
    handled_net = _NET_BIND_TCP | _NET_CONNECT_TCP if abi >= 4 else 0
    ruleset = _RulesetAttr(
        handled_access_fs=handled_fs,
        handled_access_net=handled_net,
    )
    ruleset_fd = _syscall(
        libc,
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset),
        ctypes.sizeof(ruleset),
        0,
    )
    try:
        _add_path_rule(libc, ruleset_fd, work_dir, handled_fs)
        _add_path_rule(
            libc,
            ruleset_fd,
            Path("/dev/null"),
            (_FS_READ_FILE | _FS_WRITE_FILE) & handled_fs,
        )
        for path in read_only_paths:
            _add_path_rule(libc, ruleset_fd, path, _FS_READ_ONLY & handled_fs)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(libc, _LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def _deny_network_primitives() -> None:
    machine = platform.machine()
    architecture = {
        "x86_64": (0xC000003E, 41, 53, 425),
        "aarch64": (0xC00000B7, 198, 199, 425),
    }.get(machine)
    if architecture is None:
        raise RuntimeError(f"unsupported seccomp architecture: {machine}")
    audit_arch, socket_syscall, socketpair_syscall, io_uring_setup_syscall = architecture
    statements = (_SockFilter * 11)(
        _SockFilter(0x20, 0, 0, 4),
        _SockFilter(0x15, 1, 0, audit_arch),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(0x20, 0, 0, 0),
        _SockFilter(0x15, 0, 1, socket_syscall),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(0x15, 0, 1, socketpair_syscall),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(0x15, 0, 1, io_uring_setup_syscall),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_ERRNO | errno.EPERM),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_ALLOW),
    )
    program = _SockFprog(length=len(statements), filter=statements)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(program)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--read-only", action="append", default=[], type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("sandbox command is required")

    work_dir = args.work_dir.resolve(strict=True)
    read_only = [path.resolve(strict=True) for path in args.read_only]
    python_paths = [work_dir / "src", work_dir]
    python_paths.extend(path for path in read_only if path.name in {"site-packages", "dist-packages"})
    environment = {
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(str(path) for path in python_paths if path.is_dir()),
        "TMPDIR": str(work_dir / ".tmp"),
    }
    (work_dir / ".tmp").mkdir(mode=0o700)
    _restrict_filesystem(work_dir, read_only)
    _deny_network_primitives()
    os.chdir(work_dir)
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
