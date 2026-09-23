from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from carve.schemas import Event, Trace
from carve.schemas.events import stable_hash

from .mbpp_operators import MBPP_CODE_OPERATORS, applicable_mbpp_operators
from .operators import OPERATOR_SETS, TYPE_TO_OPERATORS, compatible_with_operator
from .swebench_operators import SWEBENCH_PATCH_OPERATOR_SET, applicable_swebench_operators


@dataclass(frozen=True)
class CounterfactualJob:
    event_id: str
    event_type: str
    operator_name: str
    leverage: float
    expected_abs_delta: float = 0.0
    uncertainty: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


def event_leverage(event: Event, total_events: int) -> float:
    type_weight = {
        "stop": 3.0,
        "aggregate": 2.8,
        "critique": 2.5,
        "revise": 2.3,
        "tool": 2.0,
        "obs": 1.8,
        "msg": 1.6,
        "delegate": 1.4,
        "assign": 0.8,
        "spawn": 0.6,
    }.get(event.type, 1.0)
    position = event.t / max(1, total_events - 1)
    cost = 0.001 * (event.tokens_in + event.tokens_out) + event.cost_usd
    parent_bonus = 0.05 * len(event.parents)
    return float(type_weight + 0.25 * position + cost + parent_bonus)


def predicted_leverage(event: Event, total_events: int, prior_scores: dict[str, dict[str, float]] | None = None, zeta: float = 1.0) -> tuple[float, float, float, dict[str, Any]]:
    prior = (prior_scores or {}).get(event.event_id, {})
    if "expected_abs_delta" in prior or "uncertainty" in prior:
        expected_abs_delta = float(prior.get("expected_abs_delta", 0.0))
        uncertainty = float(prior.get("uncertainty", 0.0))
        return (
            expected_abs_delta + zeta * uncertainty,
            expected_abs_delta,
            uncertainty,
            {
                "leverage_formula": "expected_abs_delta + zeta * uncertainty",
                "zeta": float(zeta),
                "source": "prior_scores",
            },
        )
    heuristic = event_leverage(event, total_events)
    return (
        heuristic,
        heuristic,
        0.0,
        {
            "leverage_formula": "heuristic_expected_abs_delta + zeta * uncertainty",
            "zeta": float(zeta),
            "source": "heuristic",
        },
    )


def compatible_operators(event_type: str, operator_set: str = "default") -> list[str]:
    mapping = TYPE_TO_OPERATORS if operator_set == "default" else OPERATOR_SETS[operator_set]
    operators = set(mapping.get(event_type, set()))
    operators.update(op for op in mapping.get("non_stop", set()) if compatible_with_operator(event_type, op, operator_set=operator_set))
    priority = {
        "corrupt_patch_hunk": 0,
        "drop_patch_hunk": 1,
        "wrong_patch_target": 2,
        "strip_patch_context": 3,
        "wrong_return_code": 0,
        "syntax_error_code": 1,
        "empty_code": 2,
        "force_wrong_aggregate": 0,
        "drop_verified_candidate": 1,
        "wrong_final_number_aggregate_gsm8k": 0,
        "choose_inconsistent_aggregate_gsm8k": 1,
        "wrong_final_number_msg_gsm8k": 0,
        "corrupt_arithmetic_msg_gsm8k": 1,
    }
    return sorted(operators, key=lambda op: (priority.get(op, 10), op))


def applicable_operators(event: Event, operator_set: str = "default", seed: int = 0) -> list[str]:
    operators = compatible_operators(event.type, operator_set=operator_set)
    if operator_set not in {"mbpp_v1", "swebench_v1"}:
        return operators
    if operator_set == "mbpp_v1":
        code_operators = set(MBPP_CODE_OPERATORS)
        applicable_code = set(applicable_mbpp_operators(event.content))
    else:
        code_operators = SWEBENCH_PATCH_OPERATOR_SET
        applicable_code = set(applicable_swebench_operators(event.content))
    domain = [operator for operator in operators if operator not in code_operators or operator in applicable_code]
    stop = [operator for operator in domain if operator == "force_stop"]
    domain = [operator for operator in domain if operator != "force_stop"]
    if domain and operator_set == "mbpp_v1":
        rotation = int(stable_hash({"task_id": event.task_id, "event_id": event.event_id, "seed": seed}), 16) % len(domain)
        domain = domain[rotation:] + domain[:rotation]
    return domain + stop


def select_counterfactual_jobs(
    trace: Trace,
    top_m: int | None = None,
    operators_per_event: int | None = None,
    primary_events: int | None = None,
    tail_operators_per_event: int | None = None,
    event_selection: str = "top_m",
    seed: int = 0,
    prior_scores: dict[str, dict[str, float]] | None = None,
    zeta: float = 1.0,
    operator_set: str = "default",
) -> list[CounterfactualJob]:
    candidates = [
        (event, *predicted_leverage(event, len(trace.events), prior_scores=prior_scores, zeta=zeta))
        for event in trace.events
        if applicable_operators(event, operator_set=operator_set, seed=seed)
    ]
    if event_selection == "random":
        random.Random(seed).shuffle(candidates)
    elif event_selection == "type_stratified":
        candidates = _type_stratified_candidates(candidates)
    else:
        candidates.sort(key=lambda item: (item[1], item[0].t), reverse=True)
    if top_m is not None:
        candidates = candidates[:top_m]
    jobs: list[CounterfactualJob] = []
    for event_index, (event, leverage, expected_abs_delta, uncertainty, metadata) in enumerate(candidates):
        ops = applicable_operators(event, operator_set=operator_set, seed=seed)
        per_event_limit = operators_per_event
        is_tail_event = (
            primary_events is not None
            and tail_operators_per_event is not None
            and event_index >= primary_events
        )
        if is_tail_event:
            per_event_limit = tail_operators_per_event
        if per_event_limit is not None:
            stop_ops = [op for op in ops if op == "force_stop"]
            domain_ops = [op for op in ops if op != "force_stop"]
            if is_tail_event:
                ops = domain_ops[:per_event_limit]
            else:
                ops = domain_ops[:per_event_limit] + stop_ops
        for op in ops:
            jobs.append(CounterfactualJob(event.event_id, event.type, op, leverage, expected_abs_delta, uncertainty, {**dict(metadata), "operator_set": operator_set}))
    return jobs


def _type_stratified_candidates(candidates: list[tuple[Event, float, float, float, dict[str, Any]]]) -> list[tuple[Event, float, float, float, dict[str, Any]]]:
    type_order = [
        "msg",
        "revise",
        "critique",
        "delegate",
        "aggregate",
        "tool",
        "obs",
        "stop",
        "assign",
        "spawn",
    ]
    by_type: dict[str, list[tuple[Event, float, float, float, dict[str, Any]]]] = {}
    for item in candidates:
        by_type.setdefault(item[0].type, []).append(item)
    for items in by_type.values():
        items.sort(key=lambda item: (item[1], item[0].t), reverse=True)
    ordered: list[tuple[Event, float, float, float, dict[str, Any]]] = []
    while any(by_type.values()):
        for event_type in type_order:
            items = by_type.get(event_type)
            if items:
                ordered.append(items.pop(0))
        for event_type in sorted(set(by_type) - set(type_order)):
            items = by_type.get(event_type)
            if items:
                ordered.append(items.pop(0))
    return ordered
