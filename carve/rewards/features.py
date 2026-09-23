from __future__ import annotations

from carve.schemas import Event


def state_value_estimate(state_snapshot: dict) -> float:
    candidates = len(state_snapshot.get("candidate_answers", []))
    tools = state_snapshot.get("tool_results", [])
    successful_tools = sum(1 for tool in tools if tool.get("success"))
    critiques = len(state_snapshot.get("critiques", []))
    revisions = len(state_snapshot.get("revisions", []))
    stopped_bonus = 0.05 if state_snapshot.get("stopped") else 0.0
    return float(0.05 * candidates + 0.3 * successful_tools + 0.03 * revisions - 0.02 * critiques + stopped_bonus)


def cost_components(event: Event) -> dict[str, float]:
    return {
        "api_cost_usd": float(event.cost_usd),
        "token_cost": float(0.000001 * (event.tokens_in + event.tokens_out)),
        "latency_cost": float(0.000001 * event.latency_ms),
        "tool_call_cost": float(0.001 if event.type == "tool" else 0.0),
    }


def graph_neighborhood_metadata(event: Event, previous_events: list[Event]) -> dict:
    parent_ids = set(event.parents)
    parent_events = [prev for prev in previous_events if prev.event_id in parent_ids]
    return {
        "parents": list(event.parents),
        "parent_event_types": [parent.type for parent in parent_events],
        "previous_event_count": len(previous_events),
    }


def redundancy_score(event: Event, previous_events: list[Event]) -> float:
    words = set(event.content.lower().split())
    if not words or not previous_events:
        return 0.0
    best = 0.0
    for prev in previous_events:
        other = set(prev.content.lower().split())
        if other:
            best = max(best, len(words & other) / len(words | other))
    return float(best)


def grounding_score(event: Event) -> float:
    text = event.content.lower()
    return 1.0 if any(k in text for k in ["test", "verified", "evidence", "because", "assert"]) else 0.0


def contradiction_score(event: Event) -> float:
    text = event.content.lower()
    return 1.0 if any(k in text for k in ["contradiction", "impossible", "inconsistent"]) else 0.0
