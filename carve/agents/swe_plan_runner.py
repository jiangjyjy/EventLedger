from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from carve.schemas import Event, Task, Trace
from carve.swe_derived.plan_only import RepairSpec, parse_repair_plan, verify_repair_plan


@dataclass(frozen=True)
class PlanRunnerConfig:
    seed: int = 0
    model: str = "glm-5.2"
    split: str = "trace"
    format_retries: int = 1


def _prompt(role: str, task: str, context: str) -> str:
    schema = '{"diagnosis":"...","files":["..."],"symbols":["..."],"changes":["..."],"regression_tests":["..."],"stop_reason":"..."}'
    prompts = {
        "planner": f"Plan how to diagnose this software bug. Do not write code or JSON.\nTask: {task}\nContext: {context}",
        "repair_strategist": f"Return only one repair-plan JSON matching this schema: {schema}\nTask: {task}\nContext: {context}",
        "reviewer_reviser": f"Review candidate A and its public score. Return only a complete replacement JSON using schema: {schema}\nTask: {task}\nContext: {context}",
        "stopper": f"Choose candidate_a, candidate_b, or abstain. Output only the choice.\nContext: {context}",
    }
    return prompts[role]


class SWEPlanRunner:
    def __init__(self, client: Any):
        self.client = client

    def run(self, task: Task, spec: RepairSpec, config: PlanRunnerConfig | None = None) -> Trace:
        config = config or PlanRunnerConfig()
        trace_id = f"{task.task_id}-plan-{config.seed}-{uuid.uuid4().hex[:8]}"
        events: list[Event] = []
        api_calls = 0
        format_retries = 0

        def call(role: str, context: str) -> str:
            nonlocal api_calls
            api_calls += 1
            return str(self.client.complete(role, _prompt(role, task.prompt, context), seed=config.seed + len(events)))

        def call_plan(role: str, context: str) -> str:
            nonlocal format_retries
            for attempt in range(config.format_retries + 1):
                raw = call(role, context if attempt == 0 else context + "\nPrevious response was invalid JSON. Return only valid schema JSON.")
                try:
                    return json.dumps(parse_repair_plan(raw), sort_keys=True)
                except (ValueError, json.JSONDecodeError):
                    if attempt >= config.format_retries:
                        raise
                    format_retries += 1
            raise AssertionError("unreachable")

        def add(event_id: str, event_type: str, role: str, content: str, parent: str, metadata: dict[str, Any] | None = None) -> None:
            events.append(Event(event_id, trace_id, task.task_id, len(events), event_type, role, f"{role}-1", content, [parent] if parent else [], model=config.model, metadata=metadata or {}))

        planner = call("planner", "")
        add("e1", "assign", "planner", planner, "")
        add("e2", "obs", "repo_inspector", "task context inspected", "e1")
        candidate_a = call_plan("repair_strategist", planner)
        add("e3", "revise", "repair_strategist", candidate_a, "e2")
        public_a = verify_repair_plan(candidate_a, spec, hidden=False)
        add("e4", "tool", "public_plan_verifier_a", json.dumps(public_a.details, sort_keys=True), "e3", {"verifier_score": public_a.score, "verifier_success": public_a.success, "tool_name": "public_plan_verifier"})
        candidate_b = call_plan("reviewer_reviser", f"candidate_a={candidate_a}\npublic_score={public_a.score}")
        add("e5", "revise", "reviewer_reviser", candidate_b, "e4")
        public_b = verify_repair_plan(candidate_b, spec, hidden=False)
        add("e6", "tool", "public_plan_verifier_b", json.dumps(public_b.details, sort_keys=True), "e5", {"verifier_score": public_b.score, "verifier_success": public_b.success, "tool_name": "public_plan_verifier"})
        choice = call("stopper", f"candidate_a={public_a.score}; candidate_b={public_b.score}").strip().splitlines()[0].lower()
        if choice not in {"candidate_a", "candidate_b", "abstain"}:
            choice = "abstain"
        add("e7", "stop", "stopper", choice, "e6", {"choice": choice})
        final = candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else ""
        add("e8", "aggregate", "final_resolver", final, "e7", {"choice": choice})
        hidden = verify_repair_plan(final, spec, hidden=True) if final else None
        add("e9", "tool", "hidden_plan_verifier", json.dumps(hidden.details if hidden else {"tests_passed": False}, sort_keys=True), "e8", {"verifier_score": hidden.score if hidden else 0.0, "verifier_success": bool(hidden and hidden.success), "tool_name": "hidden_plan_verifier"})
        telemetry = {"api_calls": api_calls, "logical_api_roles": 4, "format_retries": format_retries}
        return Trace(trace_id, task.task_id, task.dataset, config.split, events, final, verifier_score=hidden.score if hidden else 0.0, success=bool(hidden and hidden.success), manifest={"telemetry": telemetry, "stop_choice": choice})
