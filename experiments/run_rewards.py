from __future__ import annotations

import argparse
import json
from pathlib import Path

from carve.rewards.compose import RewardWeights, compose_reward
from carve.rewards.features import contradiction_score, cost_components, graph_neighborhood_metadata, grounding_score, redundancy_score, state_value_estimate
from carve.schemas import RewardLabel, Trace
from carve.scoring.credit_value import effective_credit
from experiments.run_control import load_stop_signals


def scoped_event_key(trace_id: str | None, event_id: str) -> str:
    return f"{trace_id}::{event_id}" if trace_id else event_id


def load_credit_components(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    sources: dict[str, str] = {}
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
        sources[key] = row.get("score_source", "verifier")
    return {
        key: {"delta": total / counts[key], "score_source": sources.get(key, "verifier")}
        for key, total in totals.items()
    }


def build_reward_labels(
    trace: Trace,
    credit_by_event: dict[str, dict],
    stop_signals: dict[str, dict],
    weights: RewardWeights,
    reward_mode: str = "composed",
    disable_potential_shaping: bool = False,
    disable_stopping_reward: bool = False,
) -> list[RewardLabel]:
    labels: list[RewardLabel] = []
    previous_events = []
    for index, event in enumerate(trace.events):
        delta = float(credit_by_event.get(scoped_event_key(trace.trace_id, event.event_id), credit_by_event.get(event.event_id, {})).get("delta", 0.0))
        before_state = trace.state_snapshots[index] if index < len(trace.state_snapshots) else {}
        after_state = trace.state_snapshots[index + 1] if index + 1 < len(trace.state_snapshots) else before_state
        phi_before = state_value_estimate(before_state)
        phi_after = state_value_estimate(after_state)
        stop_signal = stop_signals.get(scoped_event_key(trace.trace_id, event.event_id), stop_signals.get(event.event_id, {}))
        stop_reward = 0.0 if disable_stopping_reward else float(stop_signal.get("stop_reward", 0.0))
        grounding = grounding_score(event)
        redundancy = redundancy_score(event, previous_events)
        contradiction = contradiction_score(event)
        label = compose_reward(
            event=event,
            delta=delta,
            phi_before=phi_before,
            phi_after=phi_after,
            grounding=grounding,
            redundancy=redundancy,
            contradiction=contradiction,
            stop_reward=stop_reward,
            weights=weights,
            cost_components=cost_components(event),
            disable_potential_shaping=disable_potential_shaping,
            metadata={
                "state_before_hash": event.state_before_hash,
                "state_after_hash": event.state_after_hash,
                "before_state_index": before_state.get("state_index"),
                "after_state_index": after_state.get("state_index"),
                "grounding_source": "local_event_evidence_keywords",
                "redundancy_source": "local_graph_neighborhood_jaccard",
                "contradiction_source": "local_event_contradiction_keywords",
                "grounding_score": grounding,
                "redundancy_score": redundancy,
                "contradiction_score": contradiction,
                "graph_neighborhood": graph_neighborhood_metadata(event, previous_events),
                "stop_signal": stop_signal,
            },
        )
        if reward_mode == "delta_only":
            label.total_reward = float(label.delta)
            label.progress = 0.0
            label.grounding = 0.0
            label.redundancy_penalty = 0.0
            label.contradiction_penalty = 0.0
            label.cost_penalty = 0.0
            label.stop_reward = 0.0
        labels.append(label)
        previous_events.append(event)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--reward-mode", choices=["composed", "delta_only"], default="composed")
    parser.add_argument("--disable-potential-shaping", action="store_true")
    parser.add_argument("--disable-stopping-reward", action="store_true")
    args = parser.parse_args()

    run_dir = Path("artifacts/runs") / args.run_id
    traces = [Trace.from_dict(json.loads(line)) for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    credit_by_event = load_credit_components(run_dir / "credit_labels.jsonl")
    stop_signals = load_stop_signals(run_dir)
    weights = RewardWeights()

    labels: list[RewardLabel] = []
    for trace in traces:
        labels.extend(
            build_reward_labels(
                trace,
                credit_by_event,
                stop_signals,
                weights,
                reward_mode=args.reward_mode,
                disable_potential_shaping=args.disable_potential_shaping,
                disable_stopping_reward=args.disable_stopping_reward,
            )
        )

    out = run_dir / "reward_labels.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label.__dict__, ensure_ascii=False) + "\n")

    summary = {
        "run_id": args.run_id,
        "reward_labels": len(labels),
        "credited_events": len(credit_by_event),
        "stop_signal_count": len(stop_signals),
        "reward_mode": args.reward_mode,
        "disable_potential_shaping": args.disable_potential_shaping,
        "disable_stopping_reward": args.disable_stopping_reward,
        "reward_formula": "lambda_delta*delta + gamma*Phi(G_t)-Phi(G_{t-1}) + lambda_g*grounding - lambda_r*redundancy - lambda_c*contradiction - lambda_cost*cost + lambda_stop*stop_reward",
        "potential_source": "state_value_estimate",
        "cost_components": ["api_cost_usd", "token_cost", "latency_cost", "tool_call_cost"],
        "weights": weights.__dict__,
        "output": str(out),
    }
    (run_dir / "reward_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
