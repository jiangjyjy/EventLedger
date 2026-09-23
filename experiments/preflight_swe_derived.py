from __future__ import annotations

import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.swe_derived.contracts import DerivedCase
from carve.verifiers.swe_derived import SWEDerivedVerifier


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _score_signature(score: Any) -> tuple[Any, ...]:
    details = score.details if isinstance(score.details, dict) else {}
    changed_files = tuple(details.get("changed_files", []))
    return (
        bool(score.success),
        float(score.score),
        details.get("returncode"),
        changed_files,
        bool(details.get("tests_passed", False)),
    )


def _verify_case_once(
    verifier: SWEDerivedVerifier,
    case: DerivedCase,
    patch: str,
) -> dict[str, Any]:
    public = verifier.verify_public(patch, case)
    hidden = verifier.verify_hidden(patch, case)
    return {
        "public": public,
        "hidden": hidden,
        "signature": (_score_signature(public), _score_signature(hidden)),
    }


def _public_evidence(entries: Iterable[dict[str, Any]]) -> str:
    evidence: list[str] = []
    for entry in entries:
        public = entry["public"]
        evidence.append(json.dumps(public.details, ensure_ascii=False, default=str))
        evidence.append(public.stderr or "")
    return "\n".join(evidence)


def preflight_case(
    case: DerivedCase,
    *,
    repeats: int = 3,
    verifier: SWEDerivedVerifier | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    verifier = verifier or SWEDerivedVerifier()
    gold_patch = case.gold_patch_path.read_text(encoding="utf-8")
    buggy_patch = (
        "diff --git a/.carve_noop b/.carve_noop\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/.carve_noop\n"
        "@@ -0,0 +1 @@\n"
        "+CARVE_BUGGY_PREFLIGHT_PATCH = True\n"
    )
    runs = []
    for _ in range(repeats):
        runs.append(
            {
                "buggy": _verify_case_once(verifier, case, buggy_patch),
                "gold": _verify_case_once(verifier, case, gold_patch),
            }
        )

    with tempfile.TemporaryDirectory(prefix="carve-swe-derived-cwd-") as cwd:
        previous_cwd = Path.cwd()
        os.chdir(cwd)
        try:
            arbitrary_cwd = _verify_case_once(verifier, case, gold_patch)
        finally:
            os.chdir(previous_cwd)

    buggy_public = [run["buggy"]["public"] for run in runs]
    buggy_hidden = [run["buggy"]["hidden"] for run in runs]
    gold = [run["gold"] for run in runs]
    gold_public = [entry["public"] for entry in gold]
    gold_hidden = [entry["hidden"] for entry in gold]
    all_scores = buggy_public + buggy_hidden + gold_public + gold_hidden
    all_scores.extend((arbitrary_cwd["public"], arbitrary_cwd["hidden"]))
    signatures = [run["buggy"]["signature"] + run["gold"]["signature"] for run in runs]
    repeatable = all(signature == signatures[0] for signature in signatures[1:])
    timeout_ms = (case.timeout_seconds + 5) * 1000
    within_timeout = all(score.runtime_ms <= timeout_ms for score in all_scores)
    forbidden = ["verifier_tests", str(case.gold_patch_path), gold_patch, *case.fail_to_pass]
    visible_entries = [
        entry
        for run in runs
        for entry in (run["buggy"], run["gold"])
    ]
    visible_entries.append(arbitrary_cwd)
    visible_text = _public_evidence(visible_entries)
    leakage_free = not any(marker and marker in visible_text for marker in forbidden)
    checks = {
        "buggy_public_failed": all(not score.success for score in buggy_public),
        "buggy_hidden_failed": all(not score.success for score in buggy_hidden),
        "gold_patch_applied": all(
            bool(entry["public"].details.get("changed_files"))
            and bool(entry["hidden"].details.get("changed_files"))
            for entry in gold
        ),
        "gold_public_passed": all(score.success for score in gold_public),
        "gold_hidden_passed": all(score.success for score in gold_hidden),
        "repeatable": repeatable,
        "within_timeout": within_timeout,
        "arbitrary_cwd_passed": arbitrary_cwd["public"].success and arbitrary_cwd["hidden"].success,
        "leakage_free": leakage_free,
    }
    result = {
        "case_id": case.case_id,
        "source_instance_id": case.source_instance_id,
        **checks,
        "qualified": all(checks.values()),
        "api_calls": 0,
        "repeats": repeats,
        "runs": runs,
        "arbitrary_cwd": arbitrary_cwd,
    }
    return result


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("manifest rows must be JSON objects")
                yield row


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_preflight(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    repeats: int = 3,
    verifier: SWEDerivedVerifier | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    qualified_path = output_dir / "qualified.jsonl"
    summary_path = output_dir / "summary.json"
    for path in (results_path, qualified_path, summary_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"output already exists: {path.name}")

    verifier = verifier or SWEDerivedVerifier(work_root=output_dir / "work")
    input_rows = 0
    qualified_rows = 0
    for row in _rows(input_path):
        input_rows += 1
        try:
            case = DerivedCase.from_manifest_row(row, input_path.parent)
            result = preflight_case(case, repeats=repeats, verifier=verifier)
            output_row = {
                key: value
                for key, value in result.items()
                if key not in {"runs", "arbitrary_cwd"}
            }
            append_jsonl_fsync(results_path, output_row)
            if result["qualified"]:
                qualified_rows += 1
                append_jsonl_fsync(qualified_path, {"case_path": str(case.gold_patch_path.parent)})
        except Exception as error:
            output_row = {
                "case_path": row.get("case_path"),
                "qualified": False,
                "error_type": type(error).__name__,
                "api_calls": 0,
            }
            append_jsonl_fsync(results_path, output_row)

    if not qualified_path.exists():
        with qualified_path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    summary = {
        "input_rows": input_rows,
        "qualified_rows": qualified_rows,
        "api_calls": 0,
        "results": str(results_path),
        "qualified_manifest": str(qualified_path),
    }
    _write_new_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    print(json.dumps(run_preflight(args.input, args.output_dir, repeats=args.repeats), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
