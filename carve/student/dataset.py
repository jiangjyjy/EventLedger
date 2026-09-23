from __future__ import annotations

import hashlib
from dataclasses import dataclass

from carve.schemas import Trace
from carve.schemas.events import VALID_EVENT_TYPES
from .embeddings import EmbeddingBackend

EVENT_TYPE_TO_ID = {name: idx for idx, name in enumerate(sorted(VALID_EVENT_TYPES))}
ROLE_TO_ID = {
    "aggregator": 0,
    "critic": 1,
    "planner": 2,
    "repo_inspector": 3,
    "researcher": 4,
    "reviser": 5,
    "solver": 6,
    "stopper": 7,
    "tester": 8,
}
EDGE_TYPE_TO_ID = {
    "control": 0,
    "delegation": 1,
    "message": 2,
    "tool_observation": 3,
    "critique_revision": 4,
    "aggregation": 5,
    "stopping": 6,
}


@dataclass
class GraphArrays:
    node_type: list[int]
    role_id: list[int]
    node_features: list[list[float]]
    text_features: list[list[float]]
    edge_index: list[tuple[int, int]]
    edge_type: list[int]
    event_ids: list[str]


def role_to_id(role: str) -> int:
    lowered = role.lower()
    for key, idx in ROLE_TO_ID.items():
        if key in lowered:
            return idx
    return len(ROLE_TO_ID)


def edge_type_for_parent(parent_type: str, child_type: str) -> int:
    pair = (parent_type, child_type)
    if "delegate" in pair:
        return EDGE_TYPE_TO_ID["delegation"]
    if pair == ("tool", "obs"):
        return EDGE_TYPE_TO_ID["tool_observation"]
    if child_type in {"critique", "revise"} or parent_type in {"critique", "revise"}:
        return EDGE_TYPE_TO_ID["critique_revision"]
    if child_type == "aggregate" or parent_type == "aggregate":
        return EDGE_TYPE_TO_ID["aggregation"]
    if child_type == "stop" or parent_type == "stop":
        return EDGE_TYPE_TO_ID["stopping"]
    if child_type == "msg" or parent_type == "msg":
        return EDGE_TYPE_TO_ID["message"]
    return EDGE_TYPE_TO_ID["control"]


def hashed_text_features(text: str, dim: int = 8) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i] / 255.0) * 2.0 - 1.0 for i in range(dim)]


def graph_arrays_from_trace(trace: Trace, embedding_backend: EmbeddingBackend | None = None) -> GraphArrays:
    id_to_idx = {event.event_id: i for i, event in enumerate(trace.events)}
    id_to_type = {event.event_id: event.type for event in trace.events}
    node_type = [EVENT_TYPE_TO_ID[event.type] for event in trace.events]
    role_id = [role_to_id(event.agent_role) for event in trace.events]
    node_features = [
            [
                event.t / max(1, len(trace.events) - 1),
                float(event.tokens_in),
                float(event.tokens_out),
                event.cost_usd,
                event.latency_ms,
            ]
            for event in trace.events
        ]
    if embedding_backend is None:
        text_features = [hashed_text_features(event.content) for event in trace.events]
    else:
        text_features = embedding_backend.embed([event.content for event in trace.events])
    edges: list[tuple[int, int]] = []
    edge_type: list[int] = []
    for event in trace.events:
        dst = id_to_idx[event.event_id]
        for parent in event.parents:
            edges.append((id_to_idx[parent], dst))
            edge_type.append(edge_type_for_parent(id_to_type[parent], event.type))
    return GraphArrays(
        node_type=node_type,
        role_id=role_id,
        node_features=node_features,
        text_features=text_features,
        edge_index=edges,
        edge_type=edge_type,
        event_ids=[e.event_id for e in trace.events],
    )
