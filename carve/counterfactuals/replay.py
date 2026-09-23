from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from carve.schemas import Intervention, Trace
from carve.schemas.events import stable_hash
from carve.verifiers import CodeVerifier, MathVerifier


@dataclass
class ReplayResult:
    score: float
    replayed_trace: Trace
    metadata: dict


class ReplayEngine:
    def __init__(
        self,
        scorer: Callable[[Trace, int], float],
        behavior_policy: str = "frozen_behavior_policy",
        continuation_policy: Callable[[Trace, Intervention, int], Trace] | None = None,
    ):
        self.scorer = scorer
        self.behavior_policy = behavior_policy
        self.continuation_policy = continuation_policy

    def replay(self, trace: Trace, intervention: Intervention, seed: int) -> ReplayResult:
        prefix_hash = stable_hash([event.to_dict() for event in intervention.prefix_events])
        if self.continuation_policy is not None:
            replayed = self.continuation_policy(trace, intervention, seed)
            replay_mode = "behavior_policy_continuation"
        else:
            replayed = self._structural_replay(trace, intervention)
            replay_mode = "structural_event_replay"
        score = float(self.scorer(replayed, seed))
        return ReplayResult(
            score=score,
            replayed_trace=replayed,
            metadata={
                "seed": seed,
                "counterfactual_seed": seed,
                "behavior_policy": self.behavior_policy,
                "replay_mode": replay_mode,
                "prefix_held_fixed": self._prefix_held_fixed(replayed, intervention),
                "prefix_hash": prefix_hash,
                "prefix_valid": prefix_hash == intervention.metadata.get("prefix_hash"),
                "target_event_id": intervention.target_event_id,
                "operator_name": intervention.operator_name,
                "num_prefix_events": len(intervention.prefix_events),
                "num_replayed_events": len(replayed.events),
                "prefix_state_hash": intervention.prefix_events[-1].state_after_hash if intervention.prefix_events else (trace.state_snapshots[0]["state_hash"] if trace.state_snapshots else None),
                "terminal_state_hash": replayed.state_snapshots[-1]["state_hash"] if replayed.state_snapshots else None,
            },
        )

    def replay_score(self, trace: Trace, intervention: Intervention, seed: int) -> float:
        return self.replay(trace, intervention, seed).score

    def factual_replay(self, trace: Trace, event_id: str, seed: int) -> ReplayResult:
        prefix = trace.prefix_before(event_id)
        target = trace.get_event(event_id)
        factual_intervention = Intervention(
            target_event_id=event_id,
            operator_name="factual",
            replacement_event=target,
            deleted=False,
            prefix_events=prefix,
            metadata={"prefix_hash": stable_hash([event.to_dict() for event in prefix])},
        )
        if self.continuation_policy is not None:
            replayed = self.continuation_policy(trace, factual_intervention, seed)
            replay_mode = "behavior_policy_continuation"
            score = float(self.scorer(replayed, seed))
        else:
            replayed = trace
            replay_mode = "factual_trace_readout"
            factual = trace.verifier_score if trace.verifier_score is not None else trace.oracle_score
            if factual is None:
                factual = 1.0 if trace.success else 0.0
            score = float(factual)
        prefix_hash = stable_hash([event.to_dict() for event in prefix])
        return ReplayResult(
            score=score,
            replayed_trace=replayed,
            metadata={
                "seed": seed,
                "factual_seed": seed,
                "behavior_policy": self.behavior_policy,
                "replay_mode": replay_mode,
                "prefix_held_fixed": self._prefix_held_fixed(replayed, factual_intervention),
                "prefix_hash": prefix_hash,
                "prefix_valid": True,
                "target_event_id": event_id,
                "operator_name": "factual",
                "num_prefix_events": len(prefix),
                "num_replayed_events": len(replayed.events),
                "prefix_state_hash": prefix[-1].state_after_hash if prefix else (trace.state_snapshots[0]["state_hash"] if trace.state_snapshots else None),
                "terminal_state_hash": replayed.state_snapshots[-1]["state_hash"] if replayed.state_snapshots else None,
            },
        )

    @staticmethod
    def _structural_replay(trace: Trace, intervention: Intervention) -> Trace:
        events = list(intervention.prefix_events)
        if intervention.replacement_event is not None:
            events.append(intervention.replacement_event)
        deleted_or_replaced = {intervention.target_event_id}
        target_seen = False
        for event in trace.events:
            if event.event_id == intervention.target_event_id:
                target_seen = True
                continue
            if target_seen:
                if event.type == "aggregate" and any(parent in deleted_or_replaced for parent in event.parents):
                    new_aggregate = ReplayEngine._reaggregate_event(trace, event, events, intervention)
                    if new_aggregate is None:
                        deleted_or_replaced.add(event.event_id)
                    else:
                        events.append(new_aggregate)
                        deleted_or_replaced.add(event.event_id)
                    continue
                if any(parent in deleted_or_replaced for parent in event.parents):
                    if intervention.replacement_event is None:
                        deleted_or_replaced.add(event.event_id)
                        continue
                    new_parents = [p for p in event.parents if p not in deleted_or_replaced]
                    if event.type != "stop":
                        new_parents.append(intervention.replacement_event.event_id)
                    if event.type == "stop" and not new_parents and events:
                        new_parents = [events[-1].event_id]
                    events.append(event.clone(parents=new_parents))
                else:
                    events.append(event)
        final_answer = ReplayEngine._final_answer_from_events(trace, events)
        score_updates = ReplayEngine._score_updates(trace, final_answer)
        return trace.clone_with_events(events, final_answer=final_answer, **score_updates)

    @staticmethod
    def _reaggregate_event(trace: Trace, original: object, events: list, intervention: Intervention):
        final_answer = ReplayEngine._final_answer_from_events(trace, events)
        if not final_answer:
            return None
        surviving_parent_ids = {event.event_id for event in events}
        parents = [parent for parent in original.parents if parent in surviving_parent_ids]
        if not parents and events:
            parents = [events[-1].event_id]
        return original.clone(
            content=final_answer,
            parents=parents,
            metadata={
                **original.metadata,
                "counterfactual_reaggregate": True,
                "target_event_id": intervention.target_event_id,
                "operator_name": intervention.operator_name,
                "surviving_candidate_event_ids": [
                    event.event_id for event in events if event.type in {"msg", "revise", "aggregate"}
                ],
            },
        )

    @staticmethod
    def _final_answer_from_events(trace: Trace, events: list) -> str:
        task = trace.manifest.get("task", {})
        if trace.dataset in {"humaneval", "mbpp"}:
            for event in reversed(events):
                if event.type in {"aggregate", "revise", "msg"}:
                    return event.content
            return ""
        for event in reversed(events):
            if event.type == "aggregate":
                return event.content
        for event in reversed(events):
            if event.type in {"revise", "msg"}:
                return event.content
        return events[-1].content if events else ""

    @staticmethod
    def _score_updates(trace: Trace, final_answer: str) -> dict:
        task = trace.manifest.get("task", {})
        if trace.dataset in {"humaneval", "mbpp"}:
            score = CodeVerifier().verify(final_answer, task.get("tests"))
            return {"verifier_score": score.score, "success": score.success}
        if trace.dataset == "gsm8k":
            score = MathVerifier().verify(final_answer, task.get("reference"))
            return {"verifier_score": score.score, "success": score.success}
        return {}

    @staticmethod
    def _prefix_held_fixed(replayed: Trace, intervention: Intervention) -> bool:
        if len(replayed.events) < len(intervention.prefix_events):
            return False
        for factual, replayed_event in zip(intervention.prefix_events, replayed.events):
            if factual.to_dict() != replayed_event.to_dict():
                return False
        return True
