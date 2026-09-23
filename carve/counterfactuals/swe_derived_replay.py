from __future__ import annotations

import json
from typing import Any

from carve.agents.swe_derived_runner import _patch_from_completion, parse_stopper, resolve_candidate
from carve.counterfactuals.replay import ReplayEngine
from carve.schemas import Event, Intervention, Trace
from carve.swe_derived.contracts import DerivedCase
from carve.verifiers.swe_derived import SWEDerivedVerifier


DOWNSTREAM_API = {
    "planner": ("patcher", "reviewer_reviser", "stopper"),
    "repo_inspector": ("patcher", "reviewer_reviser", "stopper"),
    "patcher": ("reviewer_reviser", "stopper"),
    "public_tester_a": ("reviewer_reviser", "stopper"),
    "reviewer_reviser": ("stopper",),
    "public_tester_b": ("stopper",),
    "stopper": (),
}
ROLE_SEED_OFFSET = {"planner": 0, "patcher": 1, "reviewer_reviser": 2, "stopper": 3}


def continue_swe_derived(trace: Trace, intervention: Intervention, case: DerivedCase, client: Any, seed: int, verifier: SWEDerivedVerifier | None = None) -> Trace:
    verifier = verifier or SWEDerivedVerifier()
    target = trace.get_event(intervention.target_event_id)
    prefix = list(intervention.prefix_events)
    events = prefix + ([intervention.replacement_event] if intervention.replacement_event else [])
    by_role = {event.agent_role: event for event in trace.events}
    candidate_a = next((event.content for event in events if event.agent_role == "patcher"), by_role["patcher"].content)
    candidate_b = next((event.content for event in events if event.agent_role == "reviewer_reviser"), by_role["reviewer_reviser"].content)

    def append(original: Event, content: str, metadata: dict | None = None) -> None:
        events.append(original.clone(content=content, parents=[events[-1].event_id] if events else [], metadata={**original.metadata, **(metadata or {}), "counterfactual_reexecuted": True}))

    def complete(role: str, context: str) -> str:
        return str(client.complete(role, context, seed=seed + ROLE_SEED_OFFSET[role]))

    start = trace.events.index(target) + 1
    for original in trace.events[start:]:
        role = original.agent_role
        if role == "patcher":
            candidate_a = _patch_from_completion(complete("patcher", "counterfactual continuation"))
            append(original, candidate_a)
        elif role == "public_tester_a":
            score = verifier.verify_public(candidate_a, case)
            append(original, json.dumps(score.details, sort_keys=True), {"verifier_success": score.success, "verifier_score": score.score})
        elif role == "reviewer_reviser":
            candidate_b = _patch_from_completion(complete("reviewer_reviser", f"candidate_a={candidate_a}"))
            append(original, candidate_b)
        elif role == "public_tester_b":
            score = verifier.verify_public(candidate_b, case)
            append(original, json.dumps(score.details, sort_keys=True), {"verifier_success": score.success, "verifier_score": score.score})
        elif role == "stopper":
            append(original, parse_stopper(complete("stopper", f"candidate_a={candidate_a}; candidate_b={candidate_b}")))
        elif role == "final_resolver":
            choice = next(event.content for event in reversed(events) if event.agent_role == "stopper")
            append(original, resolve_candidate(choice, candidate_a, candidate_b))
        elif role == "hidden_verifier":
            answer = next(event.content for event in reversed(events) if event.agent_role == "final_resolver")
            score = verifier.verify_hidden(answer, case) if answer else None
            append(original, json.dumps(score.details if score else {}, sort_keys=True), {"verifier_success": bool(score and score.success), "verifier_score": score.score if score else 0.0})
        else:
            append(original, original.content)
    answer = next((event.content for event in reversed(events) if event.agent_role == "final_resolver"), "")
    hidden = next((event for event in reversed(events) if event.agent_role == "hidden_verifier"), None)
    return trace.clone_with_events(events, final_answer=answer, verifier_score=float(hidden.metadata.get("verifier_score", 0.0)) if hidden else 0.0, success=bool(hidden and hidden.metadata.get("verifier_success")), manifest={**trace.manifest, "replay_mode": "behavior_policy_continuation", "crn_seeds": {role: seed + offset for role, offset in ROLE_SEED_OFFSET.items()}})


def replay_engine(case: DerivedCase, client: Any, verifier: SWEDerivedVerifier | None = None) -> ReplayEngine:
    return ReplayEngine(lambda replayed, _seed: float(replayed.verifier_score or 0.0), behavior_policy="swe_derived_api", continuation_policy=lambda trace, intervention, seed: continue_swe_derived(trace, intervention, case, client, seed, verifier))
