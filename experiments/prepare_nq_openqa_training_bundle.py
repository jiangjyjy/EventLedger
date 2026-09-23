from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_bundle(source_dirs: list[Path], output_dir: Path) -> dict[str, int]:
    traces: dict[str, dict] = {}
    labels: dict[tuple[str, str, str], dict] = {}
    for source in source_dirs:
        trace_path = source / "traces.jsonl"
        if not trace_path.exists():
            trace_path = source / "source_traces.jsonl"
        if trace_path.exists():
            for trace in _rows(trace_path):
                trace_id = trace["trace_id"]
                if trace_id in traces and traces[trace_id] != trace:
                    raise ValueError(f"conflicting trace: {trace_id}")
                traces[trace_id] = trace
        labels_path = source / "credit_labels.jsonl"
        if labels_path.exists():
            for label in _rows(labels_path):
                key = (label["trace_id"], label["event_id"], label["operator_name"])
                if key in labels and labels[key] != label:
                    raise ValueError(f"conflicting label: {key}")
                labels[key] = label
    missing = sorted({label["trace_id"] for label in labels.values()} - set(traces))
    if missing:
        raise ValueError(f"labels reference missing traces: {missing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "traces.jsonl", [traces[key] for key in sorted(traces)])
    _write(output_dir / "credit_labels.jsonl", [labels[key] for key in sorted(labels)])
    summary = {"traces": len(traces), "labels": len(labels)}
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deduplicated NQ-open training bundle")
    parser.add_argument("--source-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.source_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
