"""Create an isolated, reproducible training input for one formal RQ2 regime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from carve.student_lora.data import make_task_split


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_subset(
    *,
    task_ids: list[str],
    traces_path: Path,
    labels_path: Path,
    output_dir: Path,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, Any]:
    """Copy exactly the requested traces and their trace-scoped labels."""
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_ids must be unique")
    source_traces = _read_jsonl(traces_path)
    trace_by_task = {str(row["task_id"]): row for row in source_traces}
    if len(trace_by_task) != len(source_traces):
        raise ValueError("source traces must have one row per task_id")
    missing = [task_id for task_id in task_ids if task_id not in trace_by_task]
    if missing:
        raise ValueError(f"source traces are missing requested task IDs: {missing[:3]}")

    selected_traces = [trace_by_task[task_id] for task_id in task_ids]
    selected_trace_ids = {str(row["trace_id"]) for row in selected_traces}
    selected_labels = [row for row in _read_jsonl(labels_path) if str(row.get("trace_id", "")) in selected_trace_ids]
    split = make_task_split(
        task_ids,
        seed=seed,
        train_count=train_count,
        val_count=validation_count,
        test_count=test_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "traces.jsonl", selected_traces)
    _write_jsonl(output_dir / "credit_labels.jsonl", selected_labels)
    (output_dir / "split.json").write_text(
        json.dumps({"train": list(split.train), "validation": list(split.validation), "test": list(split.test)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "task_ids": task_ids,
        "trace_count": len(selected_traces),
        "label_count": len(selected_labels),
        "missing_label_trace_ids": sorted(selected_trace_ids - {str(row.get("trace_id", "")) for row in selected_labels}),
        "seed": seed,
        "split": {"train": len(split.train), "validation": len(split.validation), "test": len(split.test)},
        "source": {
            "traces_path": str(traces_path),
            "traces_sha256": _sha256(traces_path),
            "labels_path": str(labels_path),
            "labels_sha256": _sha256(labels_path),
        },
    }
    (output_dir / "subset_provenance.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize one fixed formal-ablation training subset")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--regime", required=True, choices=("code_math", "sql", "openqa"))
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=70)
    parser.add_argument("--validation-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=15)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = materialize_subset(
        task_ids=list(manifest["regimes"][args.regime]["task_ids"]),
        traces_path=args.traces,
        labels_path=args.labels,
        output_dir=args.output_dir,
        seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    result["regime"] = args.regime
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
