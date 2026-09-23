from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Task, Trace
from carve.verifiers.spider import SpiderVerifier


@dataclass(frozen=True)
class SpiderRunnerConfig:
    seed: int = 0
    model: str = "glm-5.2"
    split: str = "trace"


def _prompt(role: str, case: SpiderCase, context: str) -> str:
    base = f"Question: {case.question}\nSQLite schema:\n{case.schema}"
    return {
        "planner": f"Plan a SQL solution. Do not output SQL yet.\n{base}\n{context}",
        "sql_writer": f"Return only one executable SQLite SELECT query, without markdown.\n{base}\n{context}",
        "reviewer_reviser": f"Review candidate A and public execution status. Return only one complete replacement SQLite SELECT query.\n{base}\n{context}",
        "stopper": f"Choose candidate_a, candidate_b, or abstain. Output only the choice.\n{context}",
    }[role]


def _sql(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return value[:-1].rstrip() if value.endswith(";") else value


class SpiderRunner:
    def __init__(self, client: Any, verifier: SpiderVerifier | None = None):
        self.client = client
        self.verifier = verifier or SpiderVerifier()

    def run(self, task: Task, case: SpiderCase, config: SpiderRunnerConfig | None = None) -> Trace:
        config = config or SpiderRunnerConfig()
        trace_id = f"{task.task_id}-spider-{config.seed}-{uuid.uuid4().hex[:8]}"
        events: list[Event] = []

        def call(role: str, context: str) -> str:
            return str(self.client.complete(role, _prompt(role, case, context), seed=config.seed + len(events)))

        def add(event_id: str, event_type: str, role: str, content: str, parent: str, metadata: dict[str, Any] | None = None) -> None:
            events.append(Event(event_id, trace_id, task.task_id, len(events), event_type, role, f"{role}-1", content, [parent] if parent else [], model=config.model, metadata=metadata or {}))

        planner = call("planner", "")
        add("e1", "assign", "planner", planner, "")
        add("e2", "obs", "schema_inspector", case.schema, "e1")
        candidate_a = _sql(call("sql_writer", planner))
        add("e3", "revise", "sql_writer", candidate_a, "e2")
        public_a = self.verifier.verify(candidate_a, case)
        add("e4", "tool", "public_sql_verifier_a", json.dumps(public_a.details, sort_keys=True), "e3", {"verifier_score": public_a.score, "verifier_success": public_a.success, "tool_name": "sql_verifier"})
        candidate_b = _sql(call("reviewer_reviser", f"candidate_a={candidate_a}\npublic_passed={public_a.success}"))
        add("e5", "revise", "reviewer_reviser", candidate_b, "e4")
        public_b = self.verifier.verify(candidate_b, case)
        add("e6", "tool", "public_sql_verifier_b", json.dumps(public_b.details, sort_keys=True), "e5", {"verifier_score": public_b.score, "verifier_success": public_b.success, "tool_name": "sql_verifier"})
        choice = call("stopper", f"candidate_a={public_a.success}; candidate_b={public_b.success}").strip().splitlines()[0].lower()
        if choice not in {"candidate_a", "candidate_b", "abstain"}:
            choice = "abstain"
        add("e7", "stop", "stopper", choice, "e6", {"choice": choice})
        final = candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""
        add("e8", "aggregate", "final_resolver", final, "e7", {"choice": choice})
        hidden = self.verifier.verify(final, case) if final else None
        add("e9", "tool", "hidden_sql_verifier", json.dumps(hidden.details if hidden else {"tests_passed": False}, sort_keys=True), "e8", {"verifier_score": hidden.score if hidden else 0.0, "verifier_success": bool(hidden and hidden.success), "tool_name": "hidden_sql_verifier"})
        return Trace(trace_id, task.task_id, task.dataset, config.split, events, final, verifier_score=hidden.score if hidden else 0.0, success=bool(hidden and hidden.success), manifest={"telemetry": {"api_calls": 4}, "stop_choice": choice, "db_id": case.db_id})
