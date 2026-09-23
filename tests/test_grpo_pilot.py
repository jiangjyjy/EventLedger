from experiments.run_grpo_pilot import group_normalize, prune_with_actions
from carve.schemas import Event, Trace
import torch
from experiments import run_grpo_pilot as grpo


def test_group_normalize_is_zero_mean_unit_variance():
    values = group_normalize([1.0, 2.0, 3.0])

    assert abs(sum(values)) < 1e-9
    assert abs(sum(value * value for value in values) / len(values) - 1.0) < 1e-9


def test_last_token_indices_ignore_right_padding():
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])

    assert grpo._last_token_indices(attention_mask).tolist() == [2, 3]


def test_prune_with_actions_keeps_stop_and_repairs_parents():
    trace = Trace(
        "tr",
        "task",
        "humaneval",
        "train",
        [
            Event("e1", "tr", "task", 0, "msg", "solver", "s", "candidate", []),
            Event("e2", "tr", "task", 1, "critique", "critic", "c", "critique", ["e1"]),
            Event("e3", "tr", "task", 2, "aggregate", "aggregator", "a", "answer", ["e2"]),
            Event("e4", "tr", "task", 3, "stop", "stopper", "x", "stop", ["e3"]),
        ],
        "answer",
        1.0,
    )

    pruned = prune_with_actions(trace, [0, 0, 0, 0])
