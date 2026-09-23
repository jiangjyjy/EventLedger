from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Task, Trace
from carve.verifiers.spider import SpiderVerifier


@dataclass(frozen=True)
class SpiderDAGRunnerConfig:
    seed: int = 0
    model: str = "glm-5.2"
    split: str = "trace"


def _sql(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return value[:-1].rstrip() if value.endswith(";") else value


def spider_dag_prompt(role: str, case: SpiderCase, context: dict[str, Any]) -> str:
    base = f"Question: {case.question}\nSQLite schema:\n{case.schema}"
    if role == "planner":
        return f"Plan a SQLite solution. Do not output SQL.\n{base}"
    if role == "sql_writer_a":
        return f"Independently derive one executable SQLite SELECT query. Return SQL only, without markdown.\n{base}"
    if role == "sql_writer_b":
        return f"Independently derive a robust executable SQLite SELECT query. Return SQL only, without markdown.\n{base}"
    if role == "selector":
        branches = []
        for name in ("a", "b"):
            candidate = context.get(f"candidate_{name}")
            status = context.get(f"public_{name}")
            if candidate is not None:
                branches.append(f"candidate_{name}: {candidate}\npublic_sql_execution_passed: {status}")
        plan = context.get("plan")
        plan_text = f"Optional plan:\n{plan}\n" if plan else ""
        return (
            "Select the best available candidate. Return exactly candidate_a, candidate_b, or abstain. "
            "A missing candidate is unavailable and must not be selected.\n"
            f"{base}\n{plan_text}Available branches:\n" + "\n\n".join(branches)
        )
    raise ValueError(f"unsupported Spider DAG role: {role}")


class SpiderDAGRunner:
    def __init__(self, client: Any, verifier: SpiderVerifier | None = None):
        self.client = client
        self.verifier = verifier or SpiderVerifier()

    def run(self, task: Task, case: SpiderCase, config: SpiderDAGRunnerConfig | None = None) -> Trace:
        config = config or SpiderDAGRunnerConfig()
        trace_id = f"{task.task_id}-spider-dag-{config.seed}-{uuid.uuid4().hex[:8]}"
        events: list[Event] = []

        def call(role: str, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            content = str(self.client.complete(role, spider_dag_prompt(role, case, context), seed=config.seed + len(events)))
            telemetry = self.client.last_completion_telemetry() if hasattr(self.client, "last_completion_telemetry") else {}
            return content, telemetry

        def add(event_id: str, event_type: str, role: str, content: str, parents: list[str], metadata: dict[str, Any] | None = None) -> None:
            events.append(Event(event_id, trace_id, task.task_id, len(events), event_type, role, f"{role}-1", content, parents, model=config.model, metadata=metadata or {}))

        plan, plan_telemetry = call("planner", {})
        add("e1", "assign", "planner", plan, [], {"telemetry": plan_telemetry})
        writer_a, writer_a_telemetry = call("sql_writer_a", {})
        candidate_a = _sql(writer_a)
        add("e2", "revise", "sql_writer_a", candidate_a, [], {"telemetry": writer_a_telemetry})
        writer_b, writer_b_telemetry = call("sql_writer_b", {})
        candidate_b = _sql(writer_b)
        add("e3", "revise", "sql_writer_b", candidate_b, [], {"telemetry": writer_b_telemetry})
        public_a = self.verifier.verify(candidate_a, case)
        add("e4", "tool", "public_sql_verifier_a", json.dumps(public_a.details, sort_keys=True), ["e2"], {"verifier_score": public_a.score, "verifier_success": public_a.success, "tool_name": "sql_verifier", "branch": "a"})
        public_b = self.verifier.verify(candidate_b, case)
        add("e5", "tool", "public_sql_verifier_b", json.dumps(public_b.details, sort_keys=True), ["e3"], {"verifier_score": public_b.score, "verifier_success": public_b.success, "tool_name": "sql_verifier", "branch": "b"})
        context = {"plan": plan, "candidate_a": candidate_a, "candidate_b": candidate_b, "public_a": public_a.success, "public_b": public_b.success}
        selector_output, selector_telemetry = call("selector", context)
        choice = selector_output.strip().splitlines()[0].lower()
        if choice not in {"candidate_a", "candidate_b", "abstain"}:
            choice = "abstain"
        add("e6", "aggregate", "selector", choice, ["e1", "e2", "e3", "e4", "e5"], {"choice": choice, "telemetry": selector_telemetry})
        final = candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""
        add("e7", "aggregate", "final_resolver", final, ["e6"], {"choice": choice})
        hidden = self.verifier.verify(final, case) if final else None
        add("e8", "tool", "hidden_sql_verifier", json.dumps(hidden.details if hidden else {"tests_passed": False}, sort_keys=True), ["e7"], {"verifier_score": hidden.score if hidden else 0.0, "verifier_success": bool(hidden and hidden.success), "tool_name": "hidden_sql_verifier"})
        event_telemetry = [plan_telemetry, writer_a_telemetry, writer_b_telemetry, selector_telemetry]
        return Trace(trace_id, task.task_id, task.dataset, config.split, events, final, verifier_score=hidden.score if hidden else 0.0, success=bool(hidden and hidden.success), manifest={"telemetry": {"api_calls": sum(int(item.get("api_calls", 0)) for item in event_telemetry), "api_request_attempts": sum(int(item.get("api_request_attempts", item.get("api_calls", 0))) for item in event_telemetry), "input_tokens": sum(int(item.get("input_tokens") or 0) for item in event_telemetry), "output_tokens": sum(int(item.get("output_tokens") or 0) for item in event_telemetry), "latency_ms": sum(float(item.get("wall_clock_latency_ms") or 0.0) for item in event_telemetry)}, "graph": "spider_parallel_dag_v1", "db_id": case.db_id})
