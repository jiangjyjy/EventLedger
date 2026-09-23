from __future__ import annotations

import json
from pathlib import Path


BASE = Path("artifacts/spider_dag_full100_20260812_01")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    excluded = {row["task_id"] for row in read_jsonl(BASE / "factual_informative10_v1.jsonl")}
    source_paths = sorted(BASE.glob("factual_slice[0-9][0-9].jsonl")) + [BASE / "factual_remaining30.jsonl"]
    traces = [row for path in source_paths for row in read_jsonl(path)]
    if len(traces) != 100 or len({trace["task_id"] for trace in traces}) != 100:
        raise RuntimeError("expected exactly 100 unique factual traces")
    remaining = [trace for trace in traces if trace["task_id"] not in excluded]
    if len(remaining) != 90:
        raise RuntimeError(f"expected 90 traces after exclusion, found {len(remaining)}")
    output = BASE / "factual_remaining90_excluding_informative10_v1.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for trace in remaining:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
    manifest = {
        "run_id": "spider_dag_counterfactual_remaining90_v1",
        "source_trace_count": 100,
        "selected_count": 90,
        "excluded_task_ids": sorted(excluded),
        "expected_counterfactual_jobs": 360,
        "selection_rule": "all factual traces excluding informative10 already counterfactually evaluated",
    }
    (BASE / "factual_remaining90_excluding_informative10_v1.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"traces": len(remaining), "expected_jobs": 360, "output": str(output)}))


if __name__ == "__main__":
    main()
