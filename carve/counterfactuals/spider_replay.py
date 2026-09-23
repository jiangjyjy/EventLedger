from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from carve.agents.spider_runner import _prompt, _sql
from carve.counterfactuals.replay import ReplayEngine
from carve.datasets.spider import SpiderCase
from carve.schemas import Event, Intervention, Trace
from carve.verifiers.spider import SpiderVerifier


def continue_spider(trace: Trace, intervention: Intervention, case: SpiderCase, client: Any, seed: int, verifier: SpiderVerifier | None = None) -> Trace:
    verifier = verifier or SpiderVerifier()
    target = trace.get_event(intervention.target_event_id)
    events = list(intervention.prefix_events) + ([intervention.replacement_event] if intervention.replacement_event else [])
    original_by_role = {event.agent_role: event for event in trace.events}
    candidate_a = next((event.content for event in events if event.agent_role == "sql_writer"), original_by_role["sql_writer"].content)
    candidate_b = next((event.content for event in events if event.agent_role == "reviewer_reviser"), original_by_role["reviewer_reviser"].content)
    planner = next((event.content for event in events if event.agent_role == "planner"), original_by_role["planner"].content)
    schema = next((event.content for event in events if event.agent_role == "schema_inspector"), case.schema)
    prompt_case = replace(case, schema=schema)
    public_a_success = next((bool(event.metadata.get("verifier_success")) for event in events if event.agent_role == "public_sql_verifier_a"), False)
    public_b_success = next((bool(event.metadata.get("verifier_success")) for event in events if event.agent_role == "public_sql_verifier_b"), False)

    def append(original: Event, content: str, metadata: dict[str, Any] | None = None) -> None:
        events.append(original.clone(content=content, parents=[events[-1].event_id] if events else [], metadata={**original.metadata, **(metadata or {}), "counterfactual_reexecuted": True}))

    def complete(role: str, context: str) -> str:
        return str(client.complete(role, context, seed=seed))

    for original in trace.events[trace.events.index(target) + 1:]:
        role = original.agent_role
        if role == "sql_writer":
            candidate_a = _sql(complete(role, _prompt(role, prompt_case, planner))); append(original, candidate_a)
        elif role == "public_sql_verifier_a":
            score = verifier.verify(candidate_a, case); append(original, json.dumps(score.details, sort_keys=True), {"verifier_success": score.success, "verifier_score": score.score})
            public_a_success = score.success
        elif role == "reviewer_reviser":
            candidate_b = _sql(complete(role, _prompt(role, prompt_case, f"candidate_a={candidate_a}\npublic_passed={public_a_success}"))); append(original, candidate_b)
        elif role == "public_sql_verifier_b":
            score = verifier.verify(candidate_b, case); append(original, json.dumps(score.details, sort_keys=True), {"verifier_success": score.success, "verifier_score": score.score})
            public_b_success = score.success
        elif role == "stopper":
            context = f"candidate_a={public_a_success}; candidate_b={public_b_success}"
            choice = complete(role, _prompt(role, prompt_case, context)).strip().splitlines()[0].lower(); append(original, choice if choice in {"candidate_a", "candidate_b", "abstain"} else "abstain", {"choice": choice})
        elif role == "final_resolver":
            choice = next(event.content for event in reversed(events) if event.agent_role == "stopper"); append(original, candidate_a if choice == "candidate_a" else candidate_b if choice == "candidate_b" else "")
        elif role == "hidden_sql_verifier":
            answer = next(event.content for event in reversed(events) if event.agent_role == "final_resolver"); score = verifier.verify(answer, case) if answer else None; append(original, json.dumps(score.details if score else {"tests_passed": False}, sort_keys=True), {"verifier_success": bool(score and score.success), "verifier_score": score.score if score else 0.0})
        else:
            append(original, original.content)
    answer = next((event.content for event in reversed(events) if event.agent_role == "final_resolver"), "")
    hidden = next((event for event in reversed(events) if event.agent_role == "hidden_sql_verifier"), None)
    return trace.clone_with_events(events, final_answer=answer, verifier_score=float(hidden.metadata.get("verifier_score", 0.0)) if hidden else 0.0, success=bool(hidden and hidden.metadata.get("verifier_success")), manifest={**trace.manifest, "replay_mode": "behavior_policy_continuation", "crn_seeds": {"sql_writer": seed, "reviewer_reviser": seed, "stopper": seed}})


def replay_engine(case: SpiderCase, client: Any, verifier: SpiderVerifier | None = None) -> ReplayEngine:
    return ReplayEngine(lambda trace, _seed: float(trace.verifier_score or 0.0), behavior_policy="spider_api", continuation_policy=lambda trace, intervention, seed: continue_spider(trace, intervention, case, client, seed, verifier))
