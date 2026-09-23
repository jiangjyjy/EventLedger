from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class CreditLabel:
    trace_id: str
    event_id: str
    operator_family: str
    operator_name: str
    factual_score: float
    counterfactual_scores: list[float]
    delta_mean: float
    delta_std: float
    num_rollouts: int
    abstained: bool
    score_source: Literal["verifier", "oracle"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleScore:
    raw_mean: float
    calibrated_score: float
    committee_scores: list[float]
    dispersion: float
    abstain: bool
    calibration_version: str


@dataclass
class RewardLabel:
    event_id: str
    delta: float
    progress: float
    grounding: float
    redundancy_penalty: float
    contradiction_penalty: float
    cost_penalty: float
    stop_reward: float
    total_reward: float
    weights: dict[str, float]
    trace_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RLSample:
    task_id: str
    state_text: str
    action_text: str
    event_type: str
    reward: float
    advantage: float
    old_logprob: float | None
    ref_logprob: float | None
    metadata: dict[str, Any] = field(default_factory=dict)
