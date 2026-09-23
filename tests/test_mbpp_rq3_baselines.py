from types import SimpleNamespace

from experiments.run_mbpp_rq3_baselines import (
    masprm_event_scores,
    outcome_reward,
    ruler_trajectory_score,
    summarize_rows,
)
from experiments.run_mbpp_cofi_pgma_style import _build_regeneration_prompt


def _trace(success=True):
    events = [
        SimpleNamespace(type="msg", metadata={"telemetry": {"api_calls": 1, "input_tokens": 10, "output_tokens": 5, "wall_clock_latency_ms": 20}}, latency_ms=20),
        SimpleNamespace(type="tool", metadata={"verifier_score": 1.0 if success else 0.0, "verifier_success": success}, latency_ms=0),
        SimpleNamespace(type="stop", metadata={"telemetry": {"api_calls": 1, "input_tokens": 4, "output_tokens": 2, "wall_clock_latency_ms": 10}}, latency_ms=10),
    ]
    return SimpleNamespace(trace_id="t1", task_id="1", dataset="humaneval", success=success, verifier_score=1.0 if success else 0.0, events=events)


def test_outcome_reward_uses_only_terminal_verifier_outcome():
    assert outcome_reward(_trace(True)) == 1.0
    assert outcome_reward(_trace(False)) == 0.0


def test_ruler_style_score_is_deterministic_and_trace_scoped():
    trace = _trace(True)
    first = ruler_trajectory_score(trace)
    assert first == ruler_trajectory_score(trace)
    assert 0.0 <= first <= 1.0


def test_masprm_style_emits_one_independent_score_per_event():
    scores = masprm_event_scores(_trace(True))
    assert len(scores) == 3
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[-1] == 1.0


def test_summary_has_efficiency_and_method_fields():
    summary = summarize_rows([{"success": True, "api_calls": 2, "tokens": 12, "tool_calls": 1, "latency_ms": 3, "events": 3}])
    assert summary["success_rate"] == 1.0
    assert summary["mean_api_calls"] == 2.0
    assert summary["mean_tokens"] == 12.0
    assert summary["events_total"] == 3


def test_cofi_regeneration_prompt_does_not_copy_original_answer():
    trace = _trace(True)
    trace.manifest = {"task": {"prompt": "write f", "tests": "assert f(1) == 2"}}
    trace.events[0].content = "old final answer should not be copied"
    prompt = _build_regeneration_prompt(trace, trace.events[:1])
    assert "old final answer should not be copied" not in prompt
    assert "assert f(1) == 2" not in prompt
    assert "write f" in prompt
