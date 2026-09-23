from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NQOpenQACase:
    task_id: str
    question: str
    answers: tuple[str, ...]
    contexts: tuple[tuple[str, str], ...]
    evidence_ids: tuple[str, ...]


def load_nq_openqa_jsonl(path: str | Path, *, limit: int | None = None, offset: int = 0) -> list[NQOpenQACase]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = rows[offset:] if limit is None else rows[offset : offset + limit]
    cases = []
    for row in selected:
        metadata = row.get("metadata") or {}
        contexts = tuple((str(context.get("title", "")), str(context.get("text", ""))) for context in metadata.get("contexts", []))
        answers = tuple(str(answer) for answer in (row.get("reference") or {}).get("answers", []))
        if not answers or not contexts:
            raise ValueError(f"NQ OpenQA record lacks answers or DPR contexts: {row.get('task_id')}")
        cases.append(NQOpenQACase(str(row["task_id"]), str(row["prompt"]), answers, contexts, tuple(str(item) for item in metadata.get("evidence_ids", []))))
    return cases
