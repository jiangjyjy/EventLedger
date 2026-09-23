from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_training_bundles(bundles: list[Path], output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[str, dict] = {}
    labels: dict[tuple[str, str, str], dict] = {}
    duplicate_labels = 0
    for bundle in bundles:
        for row in _rows(bundle / "traces.jsonl"):
            traces.setdefault(row["trace_id"], row)
        for row in _rows(bundle / "credit_labels.jsonl"):
            key = (row["trace_id"], row["event_id"], row["operator_name"])
            if key in labels:
                duplicate_labels += 1
                continue
            labels[key] = row
    trace_rows = sorted(traces.values(), key=lambda row: (row["task_id"], row["trace_id"]))
    label_rows = sorted(labels.values(), key=lambda row: (row["trace_id"], row["event_id"], row["operator_name"]))
    (output_dir / "traces.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trace_rows), encoding="utf-8")
    (output_dir / "credit_labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in label_rows), encoding="utf-8")
    summary = {"source_bundles": len(bundles), "traces": len(trace_rows), "credit_labels": len(label_rows), "duplicate_labels_dropped": duplicate_labels}
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Spider DAG training bundles without duplicate labels")
    parser.add_argument("--bundle", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_training_bundles(args.bundle, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
