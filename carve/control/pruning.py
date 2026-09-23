from __future__ import annotations

from carve.schemas import Trace


def _answer_sink_event_ids(trace: Trace) -> set[str]:
    for event_type in ("aggregate", "revise", "msg"):
        candidates = [event for event in trace.events if event.type == event_type]
        if candidates:
            return {candidates[-1].event_id}
    return set()


def _score_for_event(trace: Trace, event_id: str, event_scores: dict[str, float]) -> float:
    return event_scores.get(f"{trace.trace_id}::{event_id}", event_scores.get(event_id, 0.0))


def prune_negative_events(trace: Trace, event_scores: dict[str, float], threshold: float = 0.0) -> Trace:
    answer_sink_ids = _answer_sink_event_ids(trace)
    kept = [
        event
        for event in trace.events
        if _score_for_event(trace, event.event_id, event_scores) >= threshold
        or event.type == "stop"
        or event.event_id in answer_sink_ids
    ]
    kept_ids = {e.event_id for e in kept}
    repaired = [e.clone(parents=[p for p in e.parents if p in kept_ids]) for e in kept]
    return trace.clone_with_events(repaired, manifest={**trace.manifest, "control": "prune_negative"})
