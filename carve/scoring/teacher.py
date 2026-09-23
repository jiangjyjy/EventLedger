from __future__ import annotations

import random

from carve.counterfactuals.crn import paired_seeds
from carve.counterfactuals.operators import OPERATOR_FAMILY, apply_operator
from carve.counterfactuals.replay import ReplayEngine
from carve.schemas import CreditLabel, Trace


def estimate_credit(
    trace: Trace,
    event_id: str,
    operator_name: str,
    replay_engine: ReplayEngine,
    k: int,
    seed: int,
    score_source: str,
    use_crn: bool = True,
    operator_set: str = "default",
) -> CreditLabel:
    rng = random.Random(seed)
    intervention = apply_operator(trace, event_id, operator_name, rng, operator_set=operator_set)
    factual = trace.verifier_score if score_source == "verifier" else trace.oracle_score
    if factual is None:
        factual = 1.0 if trace.success else 0.0
    replay_records = []
    factual_scores = []
    counterfactual_scores = []
    paired_differences = []
    factual_costs = []
    counterfactual_costs = []
    for pair in paired_seeds(seed, k, use_crn=use_crn):
        try:
            factual_replay = replay_engine.factual_replay(trace, event_id, pair.factual_seed)
            counterfactual_replay = replay_engine.replay(trace, intervention, pair.counterfactual_seed)
        except Exception as exc:
            replay_records.append(
                {
                    "factual_seed": pair.factual_seed,
                    "counterfactual_seed": pair.counterfactual_seed,
                    "crn_shared_downstream_seed": pair.factual_seed == pair.counterfactual_seed,
                    "target_event_id": event_id,
                    "operator_name": operator_name,
                    "operator_set": operator_set,
                    "failed": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            break
        metadata = dict(counterfactual_replay.metadata)
        metadata["factual_seed"] = pair.factual_seed
        metadata["counterfactual_seed"] = pair.counterfactual_seed
        metadata["crn_shared_downstream_seed"] = pair.factual_seed == pair.counterfactual_seed
        metadata["factual_score"] = float(factual_replay.score)
        metadata["counterfactual_score"] = float(counterfactual_replay.score)
        metadata["paired_difference"] = float(factual_replay.score - counterfactual_replay.score)
        metadata["factual_replay"] = factual_replay.metadata
        metadata["counterfactual_replay"] = counterfactual_replay.metadata
        metadata["factual_cost"] = factual_replay.replayed_trace.total_cost_usd
        metadata["counterfactual_cost"] = counterfactual_replay.replayed_trace.total_cost_usd
        metadata["operator_set"] = operator_set
        replay_records.append(metadata)
        factual_costs.append(float(factual_replay.replayed_trace.total_cost_usd))
        counterfactual_costs.append(float(counterfactual_replay.replayed_trace.total_cost_usd))
        factual_scores.append(float(factual_replay.score))
        counterfactual_scores.append(float(counterfactual_replay.score))
        paired_differences.append(float(factual_replay.score - counterfactual_replay.score))
    failed_rollouts = sum(1 for record in replay_records if record.get("failed"))
    if failed_rollouts:
        return CreditLabel(
            trace_id=trace.trace_id,
            event_id=event_id,
            operator_family="stop" if operator_name == "force_stop" else OPERATOR_FAMILY[operator_name],
            operator_name=operator_name,
            factual_score=float(factual),
            counterfactual_scores=[float(s) for s in counterfactual_scores],
            delta_mean=0.0,
            delta_std=0.0,
            num_rollouts=len(factual_scores),
            abstained=True,
            score_source=score_source,  # type: ignore[arg-type]
            metadata={
                "seed": seed,
                "estimator": "paired_perturb_rollout",
                "intervention": intervention.to_dict(),
                "replays": replay_records,
                "factual_cost": sum(factual_costs) / len(factual_costs) if factual_costs else float(trace.total_cost_usd),
                "factual_costs": factual_costs,
                "factual_scores": factual_scores,
                "counterfactual_costs": counterfactual_costs,
                "paired_differences": paired_differences,
                "use_crn": use_crn,
                "crn_coupled": use_crn,
                "factual_trace_score": float(factual),
                "abstain_reason": "replay_failed",
                "failed_rollouts": failed_rollouts,
                "requested_rollouts": k,
                "operator_set": operator_set,
            },
        )
    mean_factual = sum(factual_scores) / len(factual_scores) if factual_scores else float(factual)
    mean_counterfactual = sum(counterfactual_scores) / len(counterfactual_scores) if counterfactual_scores else 0.0
    delta = sum(paired_differences) / len(paired_differences) if paired_differences else float(factual) - mean_counterfactual
    variance = sum((d - delta) ** 2 for d in paired_differences) / len(paired_differences) if paired_differences else 0.0
    return CreditLabel(
        trace_id=trace.trace_id,
        event_id=event_id,
        operator_family="stop" if operator_name == "force_stop" else OPERATOR_FAMILY[operator_name],
        operator_name=operator_name,
        factual_score=float(mean_factual),
        counterfactual_scores=[float(s) for s in counterfactual_scores],
        delta_mean=delta,
        delta_std=float(variance**0.5),
        num_rollouts=k,
        abstained=False,
        score_source=score_source,  # type: ignore[arg-type]
        metadata={
            "seed": seed,
            "estimator": "paired_perturb_rollout",
            "intervention": intervention.to_dict(),
            "replays": replay_records,
            "factual_cost": sum(factual_costs) / len(factual_costs) if factual_costs else float(trace.total_cost_usd),
            "factual_costs": factual_costs,
            "factual_scores": factual_scores,
            "counterfactual_costs": counterfactual_costs,
            "paired_differences": paired_differences,
            "use_crn": use_crn,
            "crn_coupled": use_crn,
            "factual_trace_score": float(factual),
            "operator_set": operator_set,
        },
    )
