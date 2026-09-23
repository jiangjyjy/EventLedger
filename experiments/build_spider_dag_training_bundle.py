from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scheme(row: dict) -> str:
    operator_set = (row.get("metadata") or {}).get("operator_set")
    if operator_set == "spider_dag_paired_structural_cf_v1":
        return "paired_structural_shapley"
    if operator_set == "spider_dag_selector_v2":
        return "selector_direct_delta"
    raise ValueError(f"unsupported Spider credit operator set: {operator_set}")


def build_training_bundle(traces_path: Path, paired_path: Path, selector_path: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(traces_path, output_dir / "traces.jsonl")
    labels: list[dict] = []
    abstained = 0
    for path in (paired_path, selector_path):
        for row in _rows(path):
            if row.get("abstained"):
                abstained += 1
                continue
            metadata = dict(row.get("metadata") or {})
            metadata["credit_scheme"] = _scheme(row)
            row = dict(row)
            row["metadata"] = metadata
            labels.append(row)
    labels.sort(key=lambda row: (row["trace_id"], row["event_id"], row["operator_name"]))
    (output_dir / "credit_labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8")
    manifest = {"traces": sum(1 for _ in _rows(traces_path)), "credit_labels": len(labels), "abstained_excluded": abstained}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Spider DAG training bundle from v2 counterfactual labels")
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--paired-labels", required=True, type=Path)
    parser.add_argument("--selector-labels", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_training_bundle(args.traces, args.paired_labels, args.selector_labels, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
