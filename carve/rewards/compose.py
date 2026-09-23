from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from carve.schemas import Event, RewardLabel


@dataclass
class RewardWeights:
    lambda_delta: float = 1.0
    gamma: float = 0.99
    lambda_ground: float = 0.2
    lambda_redundancy: float = 0.2
    lambda_contradiction: float = 0.3
    lambda_cost: float = 0.1
    lambda_stop: float = 1.0


def compose_reward(
    event: Event,
    delta: float,
    phi_before: float,
    phi_after: float,
    grounding: float,
    redundancy: float,
    contradiction: float,
    stop_reward: float,
    weights: RewardWeights,
    cost_components: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
    disable_potential_shaping: bool = False,
) -> RewardLabel:
    progress = 0.0 if disable_potential_shaping else weights.gamma * phi_after - phi_before
    cost_components = cost_components or {
        "api_cost_usd": event.cost_usd,
        "token_cost": 0.000001 * (event.tokens_in + event.tokens_out),
        "latency_cost": 0.000001 * event.latency_ms,
        "tool_call_cost": 0.001 if event.type == "tool" else 0.0,
    }
    cost_penalty = float(sum(cost_components.values()))
    total = (
        weights.lambda_delta * delta
        + progress
        + weights.lambda_ground * grounding
        - weights.lambda_redundancy * redundancy
        - weights.lambda_contradiction * contradiction
        - weights.lambda_cost * cost_penalty
        + weights.lambda_stop * stop_reward
    )
    return RewardLabel(
        event_id=event.event_id,
        delta=float(delta),
        progress=float(progress),
        grounding=float(grounding),
        redundancy_penalty=float(redundancy),
        contradiction_penalty=float(contradiction),
        cost_penalty=float(cost_penalty),
        stop_reward=float(stop_reward),
        total_reward=float(total),
        weights=asdict(weights),
        trace_id=event.trace_id,
        task_id=event.task_id,
        metadata={
            "phi_before": float(phi_before),
            "phi_after": float(phi_after),
            "potential_source": "disabled" if disable_potential_shaping else "state_value_estimate",
            "cost_components": {key: float(value) for key, value in cost_components.items()},
            **(metadata or {}),
        },
    )
