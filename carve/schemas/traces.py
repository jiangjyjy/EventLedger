from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import Event, stable_hash


@dataclass
class Task:
    task_id: str
    dataset: str
    prompt: str
    reference: str | None = None
    tests: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Score:
    score: float
    success: bool
    details: dict[str, Any] = field(default_factory=dict)
    stderr: str | None = None
    runtime_ms: float = 0.0


@dataclass
class Trace:
    trace_id: str
    task_id: str
    dataset: str
    split: str
    events: list[Event]
    final_answer: str
    verifier_score: float | None = None
    oracle_score: float | None = None
    success: bool | None = None
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    manifest: dict[str, Any] = field(default_factory=dict)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_tokens == 0:
            self.total_tokens = sum(e.tokens_in + e.tokens_out for e in self.events)
        if self.total_cost_usd == 0.0:
            self.total_cost_usd = sum(e.cost_usd for e in self.events)
        if not self.state_snapshots:
            self.state_snapshots = build_state_snapshots(self.events, final_answer=self.final_answer)
        self._align_event_state_hashes()
        self.validate_graph()

    def validate_graph(self) -> None:
        seen: set[str] = set()
        for event in self.events:
            if event.event_id in seen:
                raise ValueError(f"duplicate event id: {event.event_id}")
            for parent in event.parents:
                if parent not in seen:
                    raise ValueError(f"missing parent {parent} for event {event.event_id}")
            seen.add(event.event_id)

    def get_event(self, event_id: str) -> Event:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)

    def _align_event_state_hashes(self) -> None:
        if len(self.state_snapshots) != len(self.events) + 1:
            return
        for index, event in enumerate(self.events):
            event.state_before_hash = self.state_snapshots[index]["state_hash"]
            event.state_after_hash = self.state_snapshots[index + 1]["state_hash"]

    def prefix_before(self, event_id: str) -> list[Event]:
        prefix = []
        for event in self.events:
            if event.event_id == event_id:
                return prefix
            prefix.append(event)
        raise KeyError(event_id)

    def clone_with_events(self, events: list[Event], **updates: Any) -> "Trace":
        data = {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "split": self.split,
            "events": events,
            "final_answer": self.final_answer,
            "verifier_score": self.verifier_score,
            "oracle_score": self.oracle_score,
            "success": self.success,
            "total_tokens": sum(e.tokens_in + e.tokens_out for e in events),
            "total_cost_usd": sum(e.cost_usd for e in events),
            "manifest": dict(self.manifest),
            "state_snapshots": build_state_snapshots(events, final_answer=updates.get("final_answer", self.final_answer)),
        }
        data.update(updates)
        return Trace(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "dataset": self.dataset,
            "split": self.split,
            "events": [e.to_dict() for e in self.events],
            "final_answer": self.final_answer,
            "verifier_score": self.verifier_score,
            "oracle_score": self.oracle_score,
            "success": self.success,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "manifest": self.manifest,
            "state_snapshots": self.state_snapshots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trace":
        copy = dict(data)
        copy["events"] = [Event.from_dict(e) for e in data["events"]]
        return cls(**copy)


def build_state_snapshots(events: list[Event], final_answer: str = "") -> list[dict[str, Any]]:
    state: dict[str, Any] = {
        "spawned_agents": [],
        "assigned_roles": [],
        "delegated_roles": [],
        "candidate_answers": [],
        "tool_results": [],
        "observations": [],
        "critiques": [],
        "revisions": [],
        "aggregate_event_id": None,
        "final_candidate": "",
        "stopped": False,
        "stop_reason": None,
        "budget_spent": 0.0,
        "tokens": 0,
        "event_ids": [],
    }
    snapshots = [_snapshot(0, None, state)]
    for index, event in enumerate(events, start=1):
        state = _advance_state(state, event, fallback_final_answer=final_answer)
        snapshots.append(_snapshot(index, event.event_id, state))
    return snapshots


def _advance_state(state: dict[str, Any], event: Event, fallback_final_answer: str = "") -> dict[str, Any]:
    next_state = json_like_copy(state)
    next_state["event_ids"].append(event.event_id)
    next_state["tokens"] += event.tokens_in + event.tokens_out
    next_state["budget_spent"] += event.cost_usd
    if event.type == "spawn":
        for role in event.metadata.get("spawned_roles", []):
            if role not in next_state["spawned_agents"]:
                next_state["spawned_agents"].append(role)
    elif event.type == "assign":
        next_state["assigned_roles"].append(
            {
                "event_id": event.event_id,
                "agent_role": event.agent_role,
                "roles": event.metadata.get("assigned_roles") or event.metadata.get("planned_next_roles", []),
                "content": event.content,
            }
        )
    elif event.type == "delegate":
        next_state["delegated_roles"].append(
            {
                "event_id": event.event_id,
                "roles": event.metadata.get("delegated_roles", []),
                "content": event.content,
            }
        )
    elif event.type == "msg":
        next_state["candidate_answers"].append({"event_id": event.event_id, "agent_role": event.agent_role, "content": event.content})
    elif event.type == "tool":
        next_state["tool_results"].append(
            {
                "event_id": event.event_id,
                "input_event_id": event.metadata.get("tool_input_event_id"),
                "score": event.metadata.get("verifier_score"),
                "success": event.metadata.get("verifier_success"),
                "tool_name": event.metadata.get("tool_name"),
                "content": event.content,
            }
        )
    elif event.type == "obs":
        next_state["observations"].append({"event_id": event.event_id, "content": event.content, "parents": list(event.parents)})
    elif event.type == "critique":
        next_state["critiques"].append({"event_id": event.event_id, "content": event.content, "parents": list(event.parents)})
    elif event.type == "revise":
        revision = {"event_id": event.event_id, "agent_role": event.agent_role, "content": event.content, "parents": list(event.parents)}
        next_state["revisions"].append(revision)
        next_state["candidate_answers"].append(revision)
    elif event.type == "aggregate":
        next_state["aggregate_event_id"] = event.event_id
        next_state["final_candidate"] = event.content or fallback_final_answer
    elif event.type == "stop":
        next_state["stopped"] = True
        next_state["stop_reason"] = event.content
        if not next_state["final_candidate"]:
            next_state["final_candidate"] = fallback_final_answer
    return next_state


def _snapshot(state_index: int, after_event_id: str | None, state: dict[str, Any]) -> dict[str, Any]:
    payload = json_like_copy(state)
    payload.update(
        {
            "state_index": state_index,
            "after_event_id": after_event_id,
        }
    )
    payload["state_hash"] = stable_hash(payload)
    return payload


def json_like_copy(value: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(value)
