"""Shared, provenance-aware helpers for the formal RQ2 subset ablation.

This module deliberately keeps the three domain verifiers outside the scoring
policy.  The policy only converts saved counterfactual labels into event or
branch scores; the final outcome is always computed by the domain verifier.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable

from carve.scoring.credit_value import effective_credit


VARIANT_NAMES = (
    "full_carve",
    "no_typed_operators",
    "no_crn_pairing",
    "no_leave_one_out",
    "random_budgeted_selection",
    "no_potential_shaping",
    "no_stopping_reward",
    "no_oracle_calibration",
    "no_conformal_abstention",
    "no_ranking_loss",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label_source: str
    decision_policy: str
    requires_recollection: bool = False
    requires_student_training: bool = False


VARIANTS = {
    "full_carve": VariantSpec("full_carve", "effective_credit", "typed_top_m"),
    "no_typed_operators": VariantSpec("no_typed_operators", "raw_credit", "message_only"),
    "no_crn_pairing": VariantSpec("no_crn_pairing", "raw_credit", "typed_top_m", True),
    "no_leave_one_out": VariantSpec("no_leave_one_out", "raw_credit", "typed_top_m"),
    "random_budgeted_selection": VariantSpec("random_budgeted_selection", "effective_credit", "random_budget"),
    "no_potential_shaping": VariantSpec("no_potential_shaping", "raw_credit", "typed_top_m"),
    "no_stopping_reward": VariantSpec("no_stopping_reward", "effective_credit", "no_stop"),
    "no_oracle_calibration": VariantSpec("no_oracle_calibration", "effective_credit", "raw_oracle"),
    "no_conformal_abstention": VariantSpec("no_conformal_abstention", "effective_credit", "no_abstention"),
    "no_ranking_loss": VariantSpec("no_ranking_loss", "student_rank_off", "student_control", requires_student_training=True),
}


def variant_spec(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown formal ablation variant: {name}") from exc


def _raw_value(row: dict[str, Any]) -> float:
    return float(row.get("delta_mean", 0.0))


def aggregate_label_scores(
    rows: Iterable[dict[str, Any]],
    *,
    variant: str,
    trace_ids: set[str] | None = None,
    seed: int = 0,
    budget_per_trace: int = 3,
) -> dict[str, float]:
    """Aggregate saved labels into trace-scoped event scores.

    ``no_leave_one_out`` intentionally uses the unadjusted rollout delta.
    Variants that need new random rollouts are marked ``requires_recollection``
    by the registry and must not be presented as collected no-CRN evidence.
    """

    spec = variant_spec(variant)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trace_id = str(row.get("trace_id", ""))
        if trace_ids is not None and trace_id not in trace_ids:
            continue
        if row.get("abstained", False):
            continue
        if row.get("operator_family") == "stop" or row.get("operator_name") in {"force_stop", "force_continue"}:
            if spec.decision_policy == "no_stop":
                continue
            continue
        grouped[(trace_id, str(row["event_id"]))].append(row)

    scores: dict[str, float] = {}
    selected: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if spec.decision_policy == "random_budget":
        for key in grouped:
            selected[key[0]].append(key)
        for trace_id, keys in selected.items():
            keys.sort(key=lambda item: _stable_seed(seed, item[1]))
            selected[trace_id] = keys[:budget_per_trace]

    for (trace_id, event_id), group in grouped.items():
        if spec.decision_policy == "message_only" and not any(
            row.get("operator_family") in {"msg", "sql_writer_a", "sql_writer_b", "reader", "router"}
            for row in group
        ):
            continue
        if spec.decision_policy == "random_budget" and (trace_id, event_id) not in selected[trace_id]:
            continue
        values = []
        for row in group:
            values.append(effective_credit(row) if spec.label_source == "effective_credit" else _raw_value(row))
        scores[f"{trace_id}::{event_id}"] = sum(float(value or 0.0) for value in values) / max(1, len(values))
    return scores


def _stable_seed(seed: int, value: str) -> int:
    digest = sha256(f"{int(seed)}::{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def mean_or_na(values: list[float]) -> float | str:
    return sum(values) / len(values) if values else "N/A"


def average_regime_scores(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    """Average per-task verifier scores and expose count/provenance."""

    scores = [float(row["verifier_score"]) for row in rows if row.get("verifier_score") is not None]
    successes = [int(bool(row.get("success"))) for row in rows]
    return {
        "verifier_score": mean_or_na(scores),
        "success_rate": mean_or_na([float(value) for value in successes]),
        "tasks": len(rows),
    }


def validate_formal_contract(manifest: dict[str, Any], required_size: int = 100) -> None:
    regimes = manifest.get("regimes", {})
    required = {"code_math", "sql", "openqa"}
    if set(regimes) != required:
        raise ValueError(f"formal manifest regimes must be {sorted(required)}")
    for regime, config in regimes.items():
        task_ids = config.get("task_ids", [])
        if len(task_ids) != required_size or len(set(task_ids)) != required_size:
            raise ValueError(f"{regime} must contain exactly {required_size} unique task IDs")
