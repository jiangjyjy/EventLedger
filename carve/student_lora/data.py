from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carve.schemas import Trace


SPIDER_DECISION_ROLES = frozenset({"planner", "sql_writer_a", "sql_writer_b", "selector"})


@dataclass(frozen=True)
class TaskSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class SiblingPair:
    preferred: "CreditRow"
    rejected: "CreditRow"


@dataclass(frozen=True)
class CreditRow:
    raw: dict[str, Any]

    @property
    def trace_id(self) -> str | None:
        return self.raw.get("trace_id")

    @property
    def event_id(self) -> str:
        return self.raw["event_id"]

    @property
    def operator_family(self) -> str:
        return self.raw.get("operator_family", "unknown")

    @property
    def operator_name(self) -> str:
        return self.raw.get("operator_name", "unknown")

    @property
    def delta_mean(self) -> float | None:
        return _score(self.raw)

    @property
    def abstained(self) -> bool:
        return bool(self.raw.get("abstained"))


@dataclass(frozen=True)
class EventExample:
    trace_id: str
    task_id: str
    event_id: str
    text: str
    event_type: str
    agent_role: str
    parents: tuple[str, ...]
    numeric_features: tuple[float, ...]
    target: float
    abstained: bool = False
    operator_name: str | None = None
    operator_family: str | None = None
    counterfactual: bool = False


@dataclass(frozen=True)
class DatasetBundle:
    traces: tuple[Trace, ...]
    factual_examples: tuple[EventExample, ...]
    counterfactual_examples: tuple[EventExample, ...]
    sibling_pairs: tuple[SiblingPair, ...]


def scoped_key(trace_id: str | None, event_id: str) -> str:
    return f"{trace_id}::{event_id}" if trace_id else event_id


def make_task_split(task_ids: list[str], seed: int, train_count: int, val_count: int, test_count: int | None = None) -> TaskSplit:
    unique = sorted(set(task_ids))
    if train_count < 0 or val_count < 0 or train_count + val_count > len(unique):
        raise ValueError("requested split sizes exceed the available task count")
    rng = random.Random(seed)
    rng.shuffle(unique)
    train_end = train_count
    val_end = train_end + val_count
    remaining = unique[val_end:]
    if test_count is not None:
        if test_count < 0 or test_count > len(remaining):
            raise ValueError("requested test split size exceeds the remaining task count")
        remaining = remaining[:test_count]
    return TaskSplit(tuple(unique[:train_end]), tuple(unique[train_end:val_end]), tuple(remaining))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_traces(path: Path) -> list[Trace]:
    return [Trace.from_dict(row) for row in _read_jsonl(path)]


def load_reward_targets(path: Path) -> dict[str, float]:
    targets: dict[str, float] = {}
    for row in _read_jsonl(path):
        targets[scoped_key(row.get("trace_id"), row["event_id"])] = float(row["total_reward"])
    return targets


def load_credit_rows(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _intervention(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    return metadata.get("intervention") or {}


def counterfactual_text(row: dict[str, Any]) -> str:
    intervention = _intervention(row)
    replacement = intervention.get("replacement_event")
    if isinstance(replacement, dict) and replacement.get("content") is not None:
        return str(replacement["content"])
    deleted = bool(row.get("deleted") or intervention.get("deleted"))
    if deleted:
        return f"[DELETED operator={row.get('operator_name', 'unknown')}]"
    return f"[OPERATOR operator={row.get('operator_name', 'unknown')}]"


def _score(row: dict[str, Any]) -> float | None:
    value = row.get("delta_mean")
    return float(value) if value is not None else None


def aggregate_event_credits(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return the mean usable intervention credit for each factual event."""
    values: dict[str, list[float]] = {}
    for row in rows:
        credit = CreditRow(row)
        score = credit.delta_mean
        if credit.abstained or score is None:
            continue
        values.setdefault(scoped_key(credit.trace_id, credit.event_id), []).append(score)
    return {key: sum(scores) / len(scores) for key, scores in values.items()}


def build_sibling_pairs(rows: list[dict[str, Any]]) -> list[SiblingPair]:
    groups: dict[tuple[str | None, str, str], list[CreditRow]] = {}
    for raw in rows:
        row = CreditRow(raw)
        if row.abstained:
            continue
        score = row.delta_mean
        if score is None:
            continue
        key = (row.trace_id, row.event_id, row.operator_family)
        groups.setdefault(key, []).append(row)

    pairs: list[SiblingPair] = []
    for group in groups.values():
        for left_index, left in enumerate(group):
            left_score = left.delta_mean
            assert left_score is not None
            for right in group[left_index + 1 :]:
                right_score = right.delta_mean
                assert right_score is not None
                if left_score == right_score:
                    continue
                if left_score > right_score:
                    pairs.append(SiblingPair(left, right))
                else:
                    pairs.append(SiblingPair(right, left))
    return pairs


def _numeric_features(event: Any, event_count: int) -> tuple[float, ...]:
    return (
        event.t / max(1, event_count - 1),
        float(event.tokens_in),
        float(event.tokens_out),
        float(event.latency_ms),
        float(event.cost_usd),
    )


def _is_trainable_factual_event(trace: Trace, event: Any) -> bool:
    # Spider DAG tool/resolver nodes expose execution and final-verifier outcomes.
    return trace.dataset != "spider" or event.agent_role in SPIDER_DECISION_ROLES


def _reward_targets(
    run_dir: Path,
    traces: list[Trace],
    *,
    disable_potential_shaping: bool = False,
    disable_stopping_reward: bool = False,
) -> dict[str, float]:
    """Build reward targets without mutating the source run."""
    from carve.rewards.compose import RewardWeights
    from experiments.run_control import load_stop_signals
    from experiments.run_rewards import build_reward_labels, load_credit_components

    credit_by_event = load_credit_components(run_dir / "credit_labels.jsonl")
    stop_signals = load_stop_signals(run_dir)
    targets: dict[str, float] = {}
    for trace in traces:
        labels = build_reward_labels(
            trace,
            credit_by_event,
            stop_signals,
            RewardWeights(),
            disable_potential_shaping=disable_potential_shaping,
            disable_stopping_reward=disable_stopping_reward,
        )
        for label in labels:
            targets[scoped_key(trace.trace_id, label.event_id)] = float(label.total_reward)
    return targets


def build_examples(run_dir: Path, split: TaskSplit, *, target_source: str = "credit") -> DatasetBundle:
    traces = load_traces(run_dir / "traces.jsonl")
    task_set = set(split.train) | set(split.validation) | set(split.test)
    traces = [trace for trace in traces if trace.task_id in task_set]
    if target_source == "credit":
        reward_targets: dict[str, float] = {}
    elif target_source == "reward_composed":
        reward_targets = _reward_targets(run_dir, traces)
    elif target_source == "reward_no_potential":
        reward_targets = _reward_targets(run_dir, traces, disable_potential_shaping=True)
    elif target_source == "reward_no_stopping":
        reward_targets = _reward_targets(run_dir, traces, disable_stopping_reward=True)
    else:
        raise ValueError(f"unknown student target source: {target_source}")
    trace_ids = {trace.trace_id for trace in traces}
    credit_rows = [row for row in load_credit_rows(run_dir / "credit_labels.jsonl") if row.get("trace_id") in trace_ids]
    event_credits = aggregate_event_credits(credit_rows)
    factual: list[EventExample] = []
    trace_by_id = {trace.trace_id: trace for trace in traces}
    for trace in traces:
        for event in trace.events:
            if not _is_trainable_factual_event(trace, event):
                continue
            target = event_credits.get(
                scoped_key(trace.trace_id, event.event_id),
                reward_targets.get(scoped_key(trace.trace_id, event.event_id), float(trace.verifier_score or 0.0)),
            )
            factual.append(
                EventExample(
                    trace_id=trace.trace_id,
                    task_id=trace.task_id,
                    event_id=event.event_id,
                    text=event.content,
                    event_type=event.type,
                    agent_role=event.agent_role,
                    parents=tuple(event.parents),
                    numeric_features=_numeric_features(event, len(trace.events)),
                    target=target,
                )
            )

    counterfactual: list[EventExample] = []
    for row in credit_rows:
        trace = trace_by_id.get(row.get("trace_id"))
        if trace is None:
            continue
        event = trace.get_event(row["event_id"])
        score = _score(row)
        if score is None:
            continue
        counterfactual.append(
            EventExample(
                trace_id=trace.trace_id,
                task_id=trace.task_id,
                event_id=event.event_id,
                text=counterfactual_text(row),
                event_type=event.type,
                agent_role=event.agent_role,
                parents=tuple(event.parents),
                numeric_features=_numeric_features(event, len(trace.events)),
                target=score,
                abstained=bool(row.get("abstained")),
                operator_name=row.get("operator_name"),
                operator_family=row.get("operator_family"),
                counterfactual=True,
            )
        )
    return DatasetBundle(tuple(traces), tuple(factual), tuple(counterfactual), tuple(build_sibling_pairs(credit_rows)))
