from __future__ import annotations

import json
import re
from pathlib import Path

from carve.counterfactuals.spider_operators import mutate_spider_event
from carve.datasets.spider import SpiderCase
from carve.schemas import Event
from carve.verifiers.spider import SpiderVerifier


OPERATORS = [
    "predicate_drop",
    "join_condition_delete",
    "aggregation_replace",
    "group_by_drop",
    "distinct_toggle",
    "order_direction_flip",
    "limit_change",
    "projection_column_drop",
]


def applicable(sql: str) -> list[str]:
    checks = {
        "predicate_drop": r"\bwhere\b",
        "join_condition_delete": r"\bjoin\b.+?\bon\b",
        "aggregation_replace": r"\b(count|sum|avg|min|max)\s*\(",
        "group_by_drop": r"\bgroup\s+by\b",
        "distinct_toggle": r"\bselect\s+distinct\b",
        "order_direction_flip": r"\border\s+by\b.+?\b(asc|desc)\b",
        "limit_change": r"\blimit\s+\d+",
        "projection_column_drop": r"\bselect\s+[^*].+?\s+from\b",
    }
    return [name for name, pattern in checks.items() if re.search(pattern, sql, re.IGNORECASE | re.DOTALL)]


def main() -> None:
    base = Path("artifacts/spider_dag_full100_20260812_01")
    selected = {
        "student_transcripts_tracking-dev-515",
        "world_1-dev-712",
        "tvshow-dev-595",
        "poker_player-dev-665",
        "pets_1-dev-53",
    }
    traces = []
    for path in sorted(base.glob("factual_slice*.jsonl")) + [base / "factual_remaining30.jsonl"]:
        traces.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    cases = {row["case_id"]: SpiderCase(**{**row, "database_path": Path(row["database_path"])}) for row in (json.loads(line) for line in Path("artifacts/spider_subset100_20260810_v4/cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    report = []
    verifier = SpiderVerifier()
    for trace in traces:
        if trace["task_id"] not in selected:
            continue
        event = next(item for item in trace["events"] if item["agent_role"] == "sql_writer_a")
        original = event["content"]
        original_score = verifier.verify(original, cases[trace["task_id"]]).score
        row = {"task_id": trace["task_id"], "original_score": original_score, "operators": []}
        for name in applicable(original):
            synthetic = Event.from_dict(event)
            synthetic.agent_role = "sql_writer"
            try:
                mutated = mutate_spider_event(synthetic, name, __import__("random").Random(17)).content
                mutated_score = verifier.verify(mutated, cases[trace["task_id"]]).score
                row["operators"].append({"operator": name, "applicable": True, "changed": mutated != original, "mutated_score": mutated_score, "semantic_delta": mutated_score - original_score, "before": original, "after": mutated})
            except ValueError:
                row["operators"].append({"operator": name, "applicable": False, "reason": "operator_not_implemented"})
        report.append(row)
    out = base / "structural_cf_v2_preflight5.json"
    out.write_text(json.dumps({"selection": sorted(selected), "operators": OPERATORS, "rows": report, "api_calls": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(report), "output": str(out), "api_calls": 0}))


if __name__ == "__main__":
    main()
