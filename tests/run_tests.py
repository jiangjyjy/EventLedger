from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_core  # noqa: E402
import test_student_lora  # noqa: E402


def main() -> int:
    failures = 0
    for module in (test_core, test_student_lora):
        for name, fn in sorted(vars(module).items()):
            if name.startswith("test_") and callable(fn):
                try:
                    fn()
                    print(f"PASS {module.__name__}.{name}")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    print(f"FAIL {module.__name__}.{name}: {exc}")
                    if inspect.trace():
                        import traceback

                        traceback.print_exc()
    print(f"{'FAILED' if failures else 'PASSED'} {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
