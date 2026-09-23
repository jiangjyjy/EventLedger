import json
from types import SimpleNamespace

from experiments.run_hcapo_style import (
    _answer_context,
    _build_hindsight_prompt,
    _build_solver_prompt,
    _multi_scale_advantages,
    verifier_name,
)


def _trace():
    events = [
        SimpleNamespace(type="assign", t="e1", event_id="e1", content="plan"),
        SimpleNamespace(type="tool", t="e2", event_id="e2", content="schema"),
        SimpleNamespace(type="aggregate", t="e3", event_id="e3", content="OLD ANSWER"),
    ]
    return SimpleNamespace(task_id="t1", dataset="humaneval", final_answer="OLD ANSWER", manifest={"task": {"prompt": "solve task"}}, events=events)


def test_hindsight_prompt_contains_event_and_outcome_without_gold_answer():
    prompt = _build_hindsight_prompt(_trace(), 0, 1.0)
    assert "current event: plan" in prompt.lower()
    assert "[e1:assign] plan" in prompt.lower()
    assert "outcome=1.0" in prompt
    assert "OLD ANSWER" not in prompt


def test_multi_scale_advantages_combines_step_segment_and_terminal_values():
    values = _multi_scale_advantages([0.2, 0.4, 0.8, 0.1], segment_size=2, terminal_value=1.0)
    assert len(values) == 4
    assert values[2] > values[0]


def test_answer_context_excludes_final_answer_event():
    events = _answer_context(_trace(), [1, 1, 1])
    assert all(event.type != "aggregate" for event in events)


def test_dataset_verifier_routing():
    assert verifier_name("GSM8K") == "MathVerifier"
    assert verifier_name("Spider") == "SpiderVerifier"
    assert verifier_name("OpenQA") == "OpenQAExactMatchVerifier"


def test_solver_prompt_is_not_a_critic_prompt():
    prompt = _build_solver_prompt(_trace(), [])
    assert "Return complete executable Python code only" in prompt
    assert "hindsight critic" not in prompt.lower()
    assert "OLD ANSWER" not in prompt
