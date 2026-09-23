from __future__ import annotations

import json
import math
from pathlib import Path

from carve.schemas import Trace

from .dataset import EDGE_TYPE_TO_ID, EVENT_TYPE_TO_ID, ROLE_TO_ID, graph_arrays_from_trace
from .embeddings import EmbeddingBackend, HashEmbeddingBackend
from .evaluate import regression_metrics
from .model import RelationalGraphStudent


def fit_linear_head(features: list[list[float]], targets: list[float], l2: float = 1e-3) -> list[float]:
    if not features:
        return []
    # Lightweight gradient descent fallback for dependency-free smoke training.
    dim = len(features[0])
    weights = [0.0 for _ in range(dim)]
    lr = 0.01
    for _ in range(200):
        grad = [l2 * w for w in weights]
        for row, target in zip(features, targets, strict=True):
            pred = sum(w * x for w, x in zip(weights, row, strict=True))
            err = pred - target
            for i, x in enumerate(row):
                grad[i] += err * x / len(features)
        weights = [w - lr * g for w, g in zip(weights, grad, strict=True)]
    return weights


def huber_loss_and_grad(error: float, delta: float = 1.0) -> tuple[float, float]:
    abs_error = abs(error)
    if abs_error <= delta:
        return 0.5 * error * error, error
    return delta * (abs_error - 0.5 * delta), delta if error > 0.0 else -delta


def bradley_terry_loss_and_grad(preferred_score: float, rejected_score: float) -> tuple[float, float, float]:
    margin = preferred_score - rejected_score
    if margin >= 0:
        loss = math.log1p(math.exp(-margin))
    else:
        loss = -margin + math.log1p(math.exp(margin))
    grad_margin = -1.0 / (1.0 + math.exp(margin))
    return loss, grad_margin, -grad_margin


def train_relational_student(
    traces: list[Trace],
    event_targets: dict[str, float],
    checkpoint_path: str | Path,
    embedding_backend: EmbeddingBackend | None = None,
    epochs: int = 20,
    hidden_dim: int = 32,
    lr: float = 0.01,
    seed: int = 0,
    ranking_pairs: list[tuple[str, str]] | None = None,
    abstained_event_ids: set[str] | None = None,
    huber_delta: float = 1.0,
    ranking_beta: float = 0.2,
) -> dict:
    backend = embedding_backend or HashEmbeddingBackend(dim=64, model_name="hash")
    initialized_from = backend.model_name if backend.model_name != "hash" and not bool(getattr(backend, "used_fallback", False)) else None
    model = RelationalGraphStudent(
        num_event_types=len(EVENT_TYPE_TO_ID),
        num_roles=len(ROLE_TO_ID) + 1,
        num_edge_types=len(EDGE_TYPE_TO_ID),
        hidden_dim=hidden_dim,
        text_dim=backend.dim,
        seed=seed,
    )
    rows: list[list[float]] = []
    targets: list[float] = []
    event_ids: list[str] = []
    abstained = set(abstained_event_ids or set())
    for trace in traces:
        arrays = graph_arrays_from_trace(trace, embedding_backend=backend)
        states = model._propagate(arrays)
        fallback = float(trace.verifier_score if trace.verifier_score is not None else trace.oracle_score or 0.0)
        for event_id, state in zip(arrays.event_ids, states, strict=True):
            scoped_event_id = f"{trace.trace_id}::{event_id}"
            if event_id in abstained or scoped_event_id in abstained:
                continue
            rows.append(state)
            targets.append(float(event_targets.get(scoped_event_id, event_targets.get(event_id, fallback))))
            event_ids.append(scoped_event_id if scoped_event_id in event_targets else event_id)

    filtered_pairs = []
    row_by_event = {event_id: idx for idx, event_id in enumerate(event_ids)}
    for preferred, rejected in ranking_pairs or []:
        if preferred in row_by_event and rejected in row_by_event:
            filtered_pairs.append((row_by_event[preferred], row_by_event[rejected]))

    last_regression_loss = 0.0
    last_ranking_loss = 0.0
    for _ in range(max(0, epochs)):
        if not rows:
            break
        grad = [0.0 for _ in model.out]
        last_regression_loss = 0.0
        for row, target in zip(rows, targets, strict=True):
            pred = sum(w * value for w, value in zip(model.out, row, strict=True))
            loss, clipped = huber_loss_and_grad(pred - target, delta=huber_delta)
            last_regression_loss += loss / len(rows)
            for i, value in enumerate(row):
                grad[i] += clipped * value / len(rows)
        last_ranking_loss = 0.0
        if filtered_pairs:
            for preferred_idx, rejected_idx in filtered_pairs:
                preferred_row = rows[preferred_idx]
                rejected_row = rows[rejected_idx]
                preferred_score = sum(w * value for w, value in zip(model.out, preferred_row, strict=True))
                rejected_score = sum(w * value for w, value in zip(model.out, rejected_row, strict=True))
                loss, grad_preferred, grad_rejected = bradley_terry_loss_and_grad(preferred_score, rejected_score)
                last_ranking_loss += loss / len(filtered_pairs)
                for i, (preferred_value, rejected_value) in enumerate(zip(preferred_row, rejected_row, strict=True)):
                    grad[i] += ranking_beta * (
                        grad_preferred * preferred_value + grad_rejected * rejected_value
                    ) / len(filtered_pairs)
        model.out = [weight - lr * g for weight, g in zip(model.out, grad, strict=True)]

    preds = [sum(w * value for w, value in zip(model.out, row, strict=True)) for row in rows]
    metrics = regression_metrics(preds, targets)
    train_predictions = {event_id: pred for event_id, pred in zip(event_ids, preds, strict=True)}
    payload = {
        "student_name": "CARVE-S",
        "model": "relational_rgAT",
        "initialized_from": initialized_from,
        "architecture": "qwen_embedding_relational_graph_attention",
        "embedding_backend": backend.model_name,
        "graph_inputs": ["typed_events", "causal_parents", "roles", "text_embeddings", "event_features"],
        "graph_head": "relational_graph_attention",
        "deployment_uses": ["event_selection_pruning", "early_stopping", "dense_rewards_for_policy_optimization"],
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "train_steps": epochs,
        "loss": {
            "objective": "Lreg + beta * Lrank",
            "regression_name": "Huber",
            "regression": float(last_regression_loss),
            "ranking_name": "Bradley-Terry",
            "ranking": float(last_ranking_loss),
            "total": float(last_regression_loss + ranking_beta * last_ranking_loss),
            "huber_delta": float(huber_delta),
            "ranking_beta": float(ranking_beta),
            "ranking_pairs": len(filtered_pairs),
            "masked_events": len(abstained),
            "abstention_masked": True,
        },
        "metrics": metrics,
        "train_predictions": train_predictions,
        **model.to_dict(),
    }
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "student_name": "CARVE-S",
        "model": "relational_rgAT",
        "initialized_from": initialized_from,
        "embedding_backend": backend.model_name,
        "train_steps": epochs,
        "regression_loss": float(last_regression_loss),
        "ranking_loss": float(last_ranking_loss),
        "ranking_pairs": len(filtered_pairs),
        "masked_events": len(abstained),
        **metrics,
        "checkpoint": str(path),
    }
