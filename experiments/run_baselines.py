from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from carve.schemas import Trace
from carve.scoring.credit_value import effective_credit


def baseline_scores_for_trace(trace: Trace, seed: int = 0) -> dict[str, dict[str, float]]:
    rng = random.Random(f"{seed}:{trace.trace_id}")
    outcome = trace.verifier_score if trace.verifier_score is not None else trace.oracle_score
    outcome = float(outcome if outcome is not None else (1.0 if trace.success else 0.0))
    role_weights = {
        "planner": 0.2,
        "solver": 0.6,
        "coder": 0.6,
        "tester": 0.4,
        "critic": 0.5,
        "reviser": 0.55,
        "aggregator": 0.45,
        "stopper": 0.25,
    }
    baselines = {
        "random": {},
        "length": {},
        "cost": {},
        "uniform_outcome": {},
        "agent_role": {},
        "message_only": {},
        "agent_deletion": {},
        "shapley_proxy": {},
    }
    role_counts: dict[str, int] = {}
    for event in trace.events:
        role_counts[event.agent_role] = role_counts.get(event.agent_role, 0) + 1
    for event in trace.events:
        baselines["random"][event.event_id] = rng.random()
        baselines["length"][event.event_id] = float(event.tokens_in + event.tokens_out + len(event.content.split()))
        baselines["cost"][event.event_id] = float(event.cost_usd + 1e-6 * (event.tokens_in + event.tokens_out) + 1e-6 * event.latency_ms)
        baselines["uniform_outcome"][event.event_id] = outcome / max(1, len(trace.events))
        baselines["agent_role"][event.event_id] = role_weights.get(event.agent_role, 0.1) * outcome
        baselines["message_only"][event.event_id] = outcome if event.type == "msg" else 0.0
        baselines["agent_deletion"][event.event_id] = outcome / max(1, role_counts[event.agent_role])
        type_prior = {
            "stop": 0.75,
            "aggregate": 0.9,
            "critique": 0.8,
            "revise": 0.85,
            "tool": 0.7,
            "obs": 0.65,
            "msg": 0.6,
            "delegate": 0.35,
            "assign": 0.25,
            "spawn": 0.2,
        }.get(event.type, 0.3)
        parent_factor = 1.0 + 0.08 * len(event.parents)
        position_factor = 0.5 + 0.5 * (event.t / max(1, len(trace.events) - 1))
        baselines["shapley_proxy"][event.event_id] = outcome * type_prior * parent_factor * position_factor
    return baselines


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / (vx * vy) ** 0.5


def compare_baseline_to_teacher(baseline: dict[str, float], teacher: dict[str, float], top_k: int = 3) -> dict[str, float]:
    ids = sorted(set(baseline) & set(teacher))
    if not ids:
        return {"spearman": 0.0, "sign_accuracy": 0.0, "top_k_overlap": 0.0, "n": 0.0}
    bx = [baseline[i] for i in ids]
    ty = [teacher[i] for i in ids]
    spearman = _pearson(_rank(bx), _rank(ty))
    sign_accuracy = sum((b >= 0) == (t >= 0) for b, t in zip(bx, ty, strict=True)) / len(ids)
    k = min(top_k, len(ids))
    top_b = {i for i, _ in sorted(((i, baseline[i]) for i in ids), key=lambda x: x[1], reverse=True)[:k]}
    top_t = {i for i, _ in sorted(((i, teacher[i]) for i in ids), key=lambda x: x[1], reverse=True)[:k]}
    overlap = len(top_b & top_t) / k if k else 0.0
    return {"spearman": float(spearman), "sign_accuracy": float(sign_accuracy), "top_k_overlap": float(overlap), "n": float(len(ids))}


def load_teacher_labels(path: Path) -> dict[str, dict[str, float]]:
    labels: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        labels.setdefault(row["trace_id"], {})
        # Average multiple operator labels per event.
        event_id = row["event_id"]
        old = labels[row["trace_id"]].get(event_id)
        value = effective_credit(row)
        if value is None:
            continue
        labels[row["trace_id"]][event_id] = value if old is None else (old + value) / 2.0
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_dir = Path("artifacts/runs") / args.run_id
    traces = [Trace.from_dict(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    teacher = load_teacher_labels(run_dir / "credit_labels.jsonl")

    rows = []
    for trace in traces:
        baselines = baseline_scores_for_trace(trace, seed=args.seed)
        for name, scores in baselines.items():
            metrics = compare_baseline_to_teacher(scores, teacher.get(trace.trace_id, {}), top_k=args.top_k)
            rows.append({"trace_id": trace.trace_id, "baseline": name, **metrics})
    summary: dict[str, dict[str, float]] = {}
    for name in sorted({row["baseline"] for row in rows}):
        subset = [row for row in rows if row["baseline"] == name]
        summary[name] = {
            key: sum(float(row[key]) for row in subset) / len(subset)
            for key in ["spearman", "sign_accuracy", "top_k_overlap", "n"]
        }
    (run_dir / "baseline_metrics.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": args.run_id, "baselines": sorted(summary), "output": str(run_dir / "baseline_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
