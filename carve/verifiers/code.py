from __future__ import annotations

import contextlib
import io
import signal
import time

from carve.schemas import Score


def extract_python_code(text: str) -> str:
    if "```" not in text:
        return text
    parts = text.split("```")
    for i in range(1, len(parts), 2):
        block = parts[i]
        lines = block.splitlines()
        if lines and lines[0].strip().lower() in {"python", "py"}:
            return "\n".join(lines[1:]).strip()
        if "def " in block or "class " in block or "import " in block:
            return block.strip()
    return text


class _VerificationTimeout(Exception):
    pass


class CodeVerifier:
    def __init__(self, timeout_s: float = 5.0):
        self.timeout_s = timeout_s

    def verify(self, code: str, tests: str | None = None) -> Score:
        start = time.time()
        if not tests:
            return Score(0.0, False, {"reason": "missing tests"})
        code = extract_python_code(code)
        namespace: dict[str, object] = {}
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

        def _raise_timeout(_signum, _frame):
            raise _VerificationTimeout

        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, namespace)
                exec(tests, namespace)
            return Score(1.0, True, {"tests": tests}, runtime_ms=(time.time() - start) * 1000.0)
        except _VerificationTimeout:
            stderr = f"verification timed out after {self.timeout_s:.2f}s"
            return Score(0.0, False, {"reason": "timeout", "tests": tests}, stderr=stderr, runtime_ms=(time.time() - start) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            stderr = repr(exc)
            return Score(0.0, False, {"tests": tests}, stderr=stderr, runtime_ms=(time.time() - start) * 1000.0)
        finally:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            signal.signal(signal.SIGALRM, previous_handler)
