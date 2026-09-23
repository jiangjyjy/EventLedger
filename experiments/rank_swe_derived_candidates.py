from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from carve.datasets.io import iter_jsonl
from carve.swe_derived.candidates import CandidateScore, candidate_to_dict, rank_candidates


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(input_path: Path, input_rows: int, ranked: list[CandidateScore], selected: list[CandidateScore]) -> dict[str, Any]:
    risk_counts = Counter(str(candidate.dependency_risk) for candidate in selected)
    return {
        "input": str(input_path),
        "input_rows": input_rows,
        "ranked_rows": len(ranked),
        "selected_rows": len(selected),
        "api_calls": 0,
        "python_only_rows": sum(1 for candidate in selected if candidate.python_only),
        "dependency_risk_counts": dict(sorted(risk_counts.items())),
        "top_instance_ids": [candidate.instance_id for candidate in selected[:10]],
    }


def run(input_path: Path, limit: int, output_path: Path, summary_path: Path) -> dict[str, Any]:
    rows = list(iter_jsonl(input_path))
    ranked = rank_candidates(rows)
    selected = ranked[:limit]
    _write_jsonl(output_path, [candidate_to_dict(candidate) for candidate in selected])
    summary = _summary(input_path.resolve(), len(rows), ranked, selected)
    _write_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank SWE-derived candidate cases without model calls.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--limit", default=30, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    summary = run(args.input, args.limit, args.output, args.summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
