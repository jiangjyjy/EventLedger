from __future__ import annotations

from carve.schemas import CreditLabel
STOP_OPERATOR_NAMES = {"force_stop", "force_continue"}


def _is_stop_label(label: CreditLabel) -> bool:
    return label.operator_family == "stop" or label.operator_name in STOP_OPERATOR_NAMES




def conservation_error(deltas: list[float], outcome: float, empty_baseline: float) -> float:
    return float(sum(deltas) - (outcome - empty_baseline))


def apply_family_baseline_and_rescale(labels: list[CreditLabel], outcome: float, empty_baseline: float) -> list[CreditLabel]:
    """Aggregate operator labels per event before conservation rescaling."""
    active = [label for label in labels if not label.abstained and not _is_stop_label(label)]
    by_event: dict[tuple[str, str], list[CreditLabel]] = {}
    for label in active:
        key = (label.trace_id, label.event_id)
        by_event.setdefault(key, []).append(label)

    event_raw: dict[tuple[str, str], float] = {}
    for key, group in by_event.items():
        event_raw[key] = sum(float(label.delta_mean) for label in group) / len(group)

    target_total = float(outcome - empty_baseline)
    raw_total = sum(event_raw.values())
    zero_tolerance = 1e-12 * max(1.0, abs(target_total), sum(abs(value) for value in event_raw.values()))
    raw_zero_unrescalable = raw_total == 0.0 and target_total != 0.0
    near_zero_unrescalable = 0.0 < abs(raw_total) <= zero_tolerance and target_total != 0.0

    if not raw_zero_unrescalable and not near_zero_unrescalable and raw_total:
        scale = target_total / raw_total
        status = "rescaled"
        conservation_satisfied = True
    elif target_total == 0.0:
        scale = 1.0
        status = "zero_target"
        conservation_satisfied = True
    else:
        scale = 0.0
        status = "raw_zero_unrescalable" if raw_zero_unrescalable else "near_zero_unrescalable"
        conservation_satisfied = False

    error_before = conservation_error(list(event_raw.values()), outcome, empty_baseline)
    rescaled_values = [float(value * scale) for value in event_raw.values()]
    error_after = conservation_error(rescaled_values, outcome, empty_baseline)

    for key, group in by_event.items():
        event_delta = event_raw[key]
        rescaled_delta = float(event_delta * scale)
        for label in group:
            label.metadata.update(
                {
                    "credit_aggregation_unit": "event",
                    "event_label_count": len(group),
                    "event_raw_delta": float(event_delta),
                    "event_operator_names": sorted(item.operator_name for item in group),
                    "family_loo_baseline": 0.0,
                    "baseline_method": "event_mean",
                    "adjusted_delta": float(event_delta),
                    "empty_baseline": float(empty_baseline),
                    "target_conservation_total": target_total,
                    "conservation_scale": float(scale),
                    "rescaled_delta": rescaled_delta,
                    "conservation_error_before": error_before,
                    "conservation_error_after": error_after,
                    "conservation_status": status,
                    "conservation_satisfied": conservation_satisfied,
                }
            )

    for label in labels:
        if label.abstained:
            label.metadata.update(
                {
                    "credit_aggregation_unit": "event",
                    "family_loo_baseline": 0.0,
                    "baseline_method": "event_mean",
                    "adjusted_delta": 0.0,
                    "empty_baseline": float(empty_baseline),
                    "target_conservation_total": target_total,
                    "conservation_scale": 0.0,
                    "rescaled_delta": 0.0,
                    "conservation_error_before": error_before,
                    "conservation_error_after": error_after,
                    "conservation_status": "abstained_excluded",
                    "conservation_satisfied": False,
                }
            )
        elif _is_stop_label(label):
            label.metadata.update(
                {
                    "credit_aggregation_unit": "stop",
                    "credit_channel": "stop",
                    "baseline_method": "separate_stop_signal",
                    "adjusted_delta": 0.0,
                    "conservation_scale": 0.0,
                    "rescaled_delta": 0.0,
                    "conservation_status": "excluded_stop_channel",
                    "conservation_satisfied": True,
                }
            )
    return labels
