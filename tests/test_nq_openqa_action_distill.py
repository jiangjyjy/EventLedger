from pathlib import Path

from experiments.nq_openqa_training import pre_generation_action_prompt, serializable_config
from experiments.run_nq_openqa_action_distill import action_target
from experiments.run_nq_openqa_grpo import _group_actions, _select_action
import torch


def test_action_distill_uses_three_fixed_actions_and_pre_generation_state_only():
    state = {"question": "Who sang?", "retrieved_evidence": "[1] Evidence", "router_assignment": "{\"reader_a_indices\": [1]}"}

    prompt = pre_generation_action_prompt(**state)

    assert action_target("use_a") == 0
    assert action_target("use_b") == 1
    assert action_target("use_factual_selector") == 2
    assert "Who sang?" in prompt
    assert "[1] Evidence" in prompt
    assert "reader_a_indices" in prompt
    assert "Candidate A" not in prompt


def test_openqa_config_serialization_converts_path_values():
    config = serializable_config({"output_dir": Path("artifacts/run"), "seed": 81})

    assert config == {"output_dir": "artifacts/run", "seed": 81}


def test_grpo_group_always_compares_all_three_openqa_actions():
    actions = _group_actions(torch.tensor([1.0, 2.0, 3.0]), group_size=4, temperature=1.0)

    assert len(actions) == 4
    assert {0, 1, 2}.issubset(actions)


def test_grpo_low_branch_margin_falls_back_to_factual_selector():
    assert _select_action(torch.tensor([1.0, 1.1, -2.0]), threshold=0.2) == 2
    assert _select_action(torch.tensor([1.0, 1.4, -2.0]), threshold=0.2) == 1
