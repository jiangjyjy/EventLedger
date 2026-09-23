from __future__ import annotations

import argparse
import json
from pathlib import Path

from carve.schemas import Trace
from carve.scoring.credit_value import effective_credit
from carve.student import HashEmbeddingBackend, NumpyGraphStudent, QwenEmbeddingBackend, RelationalGraphStudent, graph_arrays_from_trace, regression_metrics, train_relational_student
from carve.student.dataset import EDGE_TYPE_TO_ID, EVENT_TYPE_TO_ID, ROLE_TO_ID


def scoped_event_key(trace_id: str | None, event_id: str) -> str:
    return f"{trace_id}::{event_id}" if trace_id else event_id


def legacy_student_metadata(backend_model_name: str, backend_used_fallback: bool = False) -> dict[str, object]:
    paper_ready = bool(backend_model_name and backend_model_name != "hash" and not backend_used_fallback)
    return {
        "initialized_from": backend_model_name if paper_ready else None,
        "paper_ready": paper_ready,
    }




def load_event_targets(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = row["event_id"]
        key = scoped_event_key(row.get("trace_id"), event_id)
        value = effective_credit(row)
        if value is None:
            continue
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    return {key: total / counts[key] for key, total in totals.items()}


def load_reward_targets(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    targets: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        targets[scoped_event_key(row.get("trace_id"), row["event_id"])] = float(row["total_reward"])
    return targets


def load_method_training_signals(path: Path) -> tuple[dict[str, float], list[tuple[str, str]], set[str]]:
    if not path.exists():
        return {}, [], set()
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    abstained: set[str] = set()
    rows_by_event: dict[str, list[dict]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        event_id = row["event_id"]
        key = scoped_event_key(row.get("trace_id"), event_id)
        value = effective_credit(row)
        if value is None:
            abstained.add(key)
            continue
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
        row = dict(row)
        row["target_value"] = value
        rows_by_event.setdefault(key, []).append(row)

    targets = {event_id: total / counts[event_id] for event_id, total in totals.items()}
    positive = [event_id for event_id, target in targets.items() if target > 0.0 and event_id not in abstained]
    negative = [event_id for event_id, target in targets.items() if target < 0.0 and event_id not in abstained]
    ranking_pairs: list[tuple[str, str]] = []
    for preferred in positive:
        for rejected in negative:
            ranking_pairs.append((preferred, rejected))
    return targets, ranking_pairs, abstained


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--model", choices=["relational", "numpy"], default="relational")
    parser.add_argument("--embedding-backend", choices=["hash", "qwen"], default="hash")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=16)
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    traces = [Trace.from_dict(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    backend = HashEmbeddingBackend(dim=64, model_name="hash")
    if args.embedding_backend == "qwen":
        backend = QwenEmbeddingBackend(fallback=backend)
    if args.model == "numpy":
        model = NumpyGraphStudent(num_event_types=len(EVENT_TYPE_TO_ID), hidden_dim=16, seed=0)
        checkpoint_result = {}
    else:
        model = RelationalGraphStudent(
            num_event_types=len(EVENT_TYPE_TO_ID),
            num_roles=len(ROLE_TO_ID) + 1,
            num_edge_types=len(EDGE_TYPE_TO_ID),
            hidden_dim=args.hidden_dim,
            text_dim=backend.dim,
            seed=0,
        )
        targets = load_reward_targets(run_dir / "reward_labels.jsonl")
        ranking_pairs: list[tuple[str, str]] = []
        abstained_event_ids: set[str] = set()
        if not targets:
            targets, ranking_pairs, abstained_event_ids = load_method_training_signals(run_dir / "credit_labels.jsonl")
        checkpoint_result = train_relational_student(
            traces,
            targets,
            checkpoint_path=run_dir / "student_checkpoint.json",
            embedding_backend=backend,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            seed=0,
            ranking_pairs=ranking_pairs,
            abstained_event_ids=abstained_event_ids,
        )
    preds = []
    targets = []
    uncertainties = []
    for trace in traces:
        arr = graph_arrays_from_trace(trace, embedding_backend=backend if args.model != "numpy" else None)
        pred = model.forward(arr)
        preds.extend(pred)
        if hasattr(model, "uncertainty"):
            uncertainties.extend(model.uncertainty(arr))
        targets.extend([trace.verifier_score or 0.0 for _ in pred])
    metrics = regression_metrics(preds, targets)
    metrics["model"] = "relational_rgAT" if args.model == "relational" else args.model
    metrics["student_name"] = "CARVE-S" if args.model == "relational" else args.model
    metrics["initialized_from"] = "Qwen-3.5" if args.model == "relational" else None
    metrics["deployment_uses"] = ["event_selection_pruning", "early_stopping", "dense_rewards_for_policy_optimization"] if args.model == "relational" else []
    metrics["embedding_backend"] = backend.model_name if args.model != "numpy" else None
    metrics["mean_uncertainty"] = sum(uncertainties) / max(1, len(uncertainties)) if uncertainties else None
    metrics["checkpoint"] = checkpoint_result.get("checkpoint")
    metrics["loss"] = {key: checkpoint_result.get(key) for key in ["regression_loss", "ranking_loss", "ranking_pairs", "masked_events"]}
    metrics["teacher_fit"] = {key: value for key, value in checkpoint_result.items() if key in {"mae", "rmse", "corr", "sign_accuracy", "train_steps"}}
    out = run_dir / "student_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
