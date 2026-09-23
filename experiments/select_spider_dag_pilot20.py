from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _event(row: dict, role: str) -> dict:
    return next(event for event in row["events"] if event["agent_role"] == role)


def select_rows(rows: list[dict], limit: int, seed: int = 17) -> tuple[list[dict], dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        a, b = _event(row, "sql_writer_a"), _event(row, "sql_writer_b")
        pa = bool(_event(row, "public_sql_verifier_a")["metadata"].get("verifier_success"))
        pb = bool(_event(row, "public_sql_verifier_b")["metadata"].get("verifier_success"))
        choice = _event(row, "selector")["content"]
        relation = "sql_same" if a["content"].strip() == b["content"].strip() else "sql_diff"
        public = "public_same" if pa == pb else "public_diff"
        outcome = "success" if row.get("success") else "failure"
        buckets[(outcome, f"{relation}_{public}")].append(row)
    ordered = []
    # Reserve both factual outcomes first, then maximize branch/status diversity.
    for outcome, count in (("success", 12), ("failure", 8)):
        candidates = [row for key, values in buckets.items() if key[0] == outcome for row in values]
        candidates.sort(key=lambda row: (row["task_id"], row["trace_id"]))
        ordered.extend(candidates[:count])
    if len(ordered) < limit:
        chosen = {row["trace_id"] for row in ordered}
        remaining = [row for row in rows if row["trace_id"] not in chosen]
        remaining.sort(key=lambda row: (not bool(row.get("success")), row["task_id"], row["trace_id"]))
        ordered.extend(remaining[: limit - len(ordered)])
    selected = ordered[:limit]
    manifest = {
        "selection": "spider_dag_stratified_pilot20",
        "seed": seed,
        "requested": limit,
        "selected": len(selected),
        "success": sum(bool(row.get("success")) for row in selected),
        "failure": sum(not bool(row.get("success")) for row in selected),
        "trace_ids": [row["trace_id"] for row in selected],
        "task_ids": [row["task_id"] for row in selected],
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--exclude-traces", type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if bool(args.input) == bool(args.input_dir):
        raise ValueError("provide exactly one of --input or --input-dir")
    if args.input_dir:
        paths = sorted(args.input_dir.glob("factual_slice*.jsonl")) + [args.input_dir / "factual_remaining30.jsonl"]
        rows = [row for path in paths for row in _read(path)]
    else:
        rows = _read(args.input)
    excluded = set()
    if args.exclude_manifest:
        excluded = set(json.loads(args.exclude_manifest.read_text(encoding="utf-8")).get("task_ids", []))
    for path in args.exclude_traces or []:
        excluded.update(row["task_id"] for row in _read(path))
    if excluded:
        rows = [row for row in rows if row["task_id"] not in excluded]
    selected, manifest = select_rows(rows, args.limit, args.seed)
    manifest["excluded_task_ids"] = sorted(excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "success": manifest["success"], "failure": manifest["failure"]}, sort_keys=True))


if __name__ == "__main__":
    main()
