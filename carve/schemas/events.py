from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Literal

EventType = Literal[
    "spawn",
    "assign",
    "delegate",
    "msg",
    "tool",
    "obs",
    "critique",
    "revise",
    "aggregate",
    "stop",
]

VALID_EVENT_TYPES = {
    "spawn",
    "assign",
    "delegate",
    "msg",
    "tool",
    "obs",
    "critique",
    "revise",
    "aggregate",
    "stop",
}


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Event:
    event_id: str
    trace_id: str
    task_id: str
    t: int
    type: EventType
    agent_role: str
    agent_id: str
    content: str
    parents: list[str] = field(default_factory=list)
    state_before_hash: str = ""
    state_after_hash: str = ""
    model: str | None = None
    prompt_hash: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in VALID_EVENT_TYPES:
            raise ValueError(f"invalid event type: {self.type}")
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.t < 0:
            raise ValueError("event index t must be non-negative")
        if not self.state_before_hash:
            self.state_before_hash = stable_hash({"event_id": self.event_id, "t": self.t, "before": self.parents})
        if not self.state_after_hash:
            self.state_after_hash = stable_hash({"event_id": self.event_id, "content": self.content, "after": self.t})

    def clone(self, **updates: Any) -> "Event":
        return replace(copy.deepcopy(self), **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "t": self.t,
            "type": self.type,
            "agent_role": self.agent_role,
            "agent_id": self.agent_id,
            "content": self.content,
            "parents": list(self.parents),
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "model": self.model,
            "prompt_hash": self.prompt_hash,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(**data)


@dataclass
class Intervention:
    target_event_id: str
    operator_name: str
    replacement_event: Event | None
    deleted: bool
    prefix_events: list[Event]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_event_id": self.target_event_id,
            "operator_name": self.operator_name,
            "replacement_event": self.replacement_event.to_dict() if self.replacement_event else None,
            "deleted": self.deleted,
            "prefix_events": [e.to_dict() for e in self.prefix_events],
            "metadata": self.metadata,
        }
