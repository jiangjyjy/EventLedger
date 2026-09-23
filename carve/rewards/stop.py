from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StopSignal:
    saved_cost: float
    forgone_gain: float
    stop_reward: float
    continued_score: float
    factual_score: float


def stopping_reward(saved_cost: float, continued_gain: float) -> float:
    return float(saved_cost - max(0.0, continued_gain))


def compute_stop_signal(
    event_type: str,
    factual_score: float,
    counterfactual_scores: list[float],
    factual_cost: float,
    counterfactual_costs: list[float],
    beta_cost: float = 1.0,
) -> StopSignal:
    continued_score = sum(counterfactual_scores) / len(counterfactual_scores) if counterfactual_scores else factual_score
    continued_cost = sum(counterfactual_costs) / len(counterfactual_costs) if counterfactual_costs else factual_cost
    if event_type == "stop":
        saved_cost = max(0.0, continued_cost - factual_cost)
        forgone_gain = max(0.0, continued_score - factual_score)
    else:
        saved_cost = max(0.0, factual_cost - continued_cost)
        forgone_gain = max(0.0, factual_score - continued_score)
    reward = stopping_reward(beta_cost * saved_cost, forgone_gain)
    return StopSignal(
        saved_cost=float(saved_cost),
        forgone_gain=float(forgone_gain),
        stop_reward=float(reward),
        continued_score=float(continued_score),
        factual_score=float(factual_score),
    )
