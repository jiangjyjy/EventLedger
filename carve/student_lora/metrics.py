from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _pearson(predictions: Sequence[float], targets: Sequence[float]) -> float:
    if len(predictions) < 2:
        return 0.0
    pred_mean = _mean(predictions)
    target_mean = _mean(targets)
    numerator = sum((pred - pred_mean) * (target - target_mean) for pred, target in zip(predictions, targets, strict=True))
    pred_var = sum((pred - pred_mean) ** 2 for pred in predictions)
    target_var = sum((target - target_mean) ** 2 for target in targets)
    denominator = math.sqrt(pred_var * target_var)
    return numerator / denominator if denominator > 0.0 else 0.0


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + end - 1) / 2.0 + 1.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def regression_and_ranking_metrics(
    predictions: Iterable[float],
    targets: Iterable[float],
    preferred_scores: Iterable[float] = (),
    rejected_scores: Iterable[float] = (),
) -> dict[str, float]:
    pred_values = _as_float_list(predictions)
    target_values = _as_float_list(targets)
    if len(pred_values) != len(target_values):
        raise ValueError("predictions and targets must have equal length")
    if pred_values:
        errors = [pred - target for pred, target in zip(pred_values, target_values, strict=True)]
        mae = sum(abs(error) for error in errors) / len(errors)
        rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    else:
        mae = 0.0
        rmse = 0.0
    preferred = _as_float_list(preferred_scores)
    rejected = _as_float_list(rejected_scores)
    if len(preferred) != len(rejected):
        raise ValueError("preferred and rejected scores must have equal length")
    ranking_accuracy = (
        sum(float(left > right) for left, right in zip(preferred, rejected, strict=True)) / len(preferred)
        if preferred
        else 0.0
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(_pearson(pred_values, target_values)),
        "spearman": float(_pearson(_rank(pred_values), _rank(target_values))),
        "ranking_accuracy": float(ranking_accuracy),
        "ranking_pairs": float(len(preferred)),
        "examples": float(len(pred_values)),
    }


def mc_dropout_std(samples: Iterable[Iterable[float]]) -> list[float]:
    rows = [_as_float_list(row) for row in samples]
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("MC-dropout samples must have equal widths")
    if len(rows) == 1:
        return [0.0] * width
    means = [_mean([row[index] for row in rows]) for index in range(width)]
    return [
        math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / (len(rows) - 1))
        for index in range(width)
    ]
