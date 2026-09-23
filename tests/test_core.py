import json
import random
import subprocess
import sys
from pathlib import Path
import tempfile

from carve.counterfactuals.crn import paired_seeds
from carve.counterfactuals.operators import apply_operator
from carve.counterfactuals.operators import compatible_with_operator
from carve.counterfactuals.selection import select_counterfactual_jobs
from carve.counterfactuals.replay import ReplayEngine
from carve.agents.api_client import APIClientConfig, OpenAICompatibleClient
from carve.control.early_stop import should_stop
from carve.control.pruning import prune_negative_events
from carve.control.reranking import rerank_traces
from carve.control.rl_export import export_rl_samples
from carve.rewards.stop import compute_stop_signal
from carve.rewards.compose import RewardWeights, compose_reward
from carve.scoring.calibration import CalibratedOracle
from carve.scoring.conservation import conservation_error, apply_family_baseline_and_rescale
from carve.scoring.teacher import estimate_credit
from carve.schemas import CreditLabel, Event, Task, Trace
from carve.student.dataset import graph_arrays_from_trace
from carve.student.model import NumpyGraphStudent, RelationalGraphStudent
from carve.student.embeddings import HashEmbeddingBackend, QwenEmbeddingBackend, embed_trace_events
from carve.student.train import train_relational_student
from carve.verifiers.math import MathVerifier
from carve.verifiers.code import CodeVerifier, extract_python_code
from carve.verifiers.swebench import SWEBenchVerifier, apply_patch_to_repo
from carve.datasets.humaneval import load_humaneval_jsonl
from carve.datasets.mbpp import load_mbpp_jsonl
from carve.datasets.gsm8k import load_gsm8k_jsonl
from carve.datasets.swebench import load_swebench_jsonl
from experiments.run_counterfactuals import build_behavior_continuation_policy, conservation_summary, prs_score_for_trace, verifier_score_for_trace
from experiments.run_baselines import baseline_scores_for_trace, compare_baseline_to_teacher
from experiments.run_judge_reranking import score_traces_with_committee, summarize_judge_reranking
from experiments.run_oracle import compute_oracle_metrics, default_method_judge_specs, build_judge_committee
from experiments.run_control import control_reward_source, load_control_event_scores, load_stop_signals, score_trace_for_control, summarize_controlled_trace
from experiments.run_rewards import build_reward_labels
from experiments.run_counterfactuals import filter_counterfactual_jobs
from experiments.validate_run import validate_run_dir
from experiments.run_ppo_smoke import compute_ppo_smoke_metrics
from experiments.plan_paper_campaign import build_campaign_plan
from experiments.run_paper_campaign import build_run_command, campaign_run_status, execute_campaign_runs, summarize_campaign
from experiments.make_paper_tables import build_paper_tables, render_markdown_tables
from experiments.run_ablations import default_ablation_specs, summarize_ablation_runs
from experiments.prepare_swebench_repos import prepare_swebench_instances
from experiments.prepare_openqa_anchors import build_anchor_template, calibration_report_from_anchors
from experiments.run_humaneval_nostudent_batch import build_command as build_humaneval_batch_command
from experiments.run_humaneval_nostudent_batch import iter_offsets, should_stop_after_failure
from carve.scoring.oracle import APIJudge, DeterministicJudge, JudgeCommittee, load_anchor_scores
from carve.agents.runner import MultiAgentRunner, RunnerConfig
from carve.agents.roles import get_role_specs


def make_trace(score=1.0):
    events = [
        Event(
            event_id="e1",
            trace_id="tr1",
            task_id="task1",
            t=0,
            type="delegate",
            agent_role="planner",
            agent_id="planner-1",
            content="Ask solver to solve.",
            parents=[],
        ),
        Event(
            event_id="e2",
            trace_id="tr1",
            task_id="task1",
            t=1,
            type="msg",
            agent_role="solver",
            agent_id="solver-1",
            content="The answer is 42.",
            parents=["e1"],
            tokens_out=5,
        ),
        Event(
            event_id="e3",
            trace_id="tr1",
            task_id="task1",
            t=2,
            type="critique",
            agent_role="critic",
            agent_id="critic-1",
            content="Check arithmetic.",
            parents=["e2"],
        ),
        Event(
            event_id="e4",
            trace_id="tr1",
            task_id="task1",
            t=3,
            type="aggregate",
            agent_role="aggregator",
            agent_id="agg-1",
            content="Final answer: 42.",
            parents=["e2", "e3"],
        ),
        Event(
            event_id="e5",
            trace_id="tr1",
            task_id="task1",
            t=4,
            type="stop",
            agent_role="stopper",
            agent_id="stop-1",
            content="Stop because answer is verified.",
            parents=["e4"],
        ),
    ]
    return Trace(
        trace_id="tr1",
        task_id="task1",
        dataset="gsm8k",
        split="test",
        events=events,
        final_answer="42",
        verifier_score=score,
        success=score >= 1.0,
    )


def test_trace_validation_rejects_missing_parent():
    trace = make_trace()
    trace.events[1].parents = ["missing"]
    try:
        trace.validate_graph()
    except ValueError as exc:
        assert "missing parent" in str(exc)
    else:
        raise AssertionError("expected missing parent validation error")


def test_counterfactual_operator_preserves_prefix_and_replaces_target_only():
    trace = make_trace()
    intervention = apply_operator(trace, "e2", "nullify", random.Random(0))
    assert [e.event_id for e in intervention.prefix_events] == ["e1"]
    assert intervention.replacement_event is not None
    assert intervention.replacement_event.type == "msg"
    assert intervention.replacement_event.content == ""
    assert intervention.metadata["rng_seed"] is not None


def test_tco_locality_preserves_parents_and_prefix_state():
    trace = make_trace()
    intervention = apply_operator(trace, "e4", "drop_source", random.Random(0))
    target = trace.get_event("e4")
    assert intervention.replacement_event is not None
    assert intervention.replacement_event.type == target.type
    assert intervention.replacement_event.parents == target.parents
    assert intervention.prefix_events == trace.prefix_before("e4")
    assert intervention.metadata["locality"] == "target_event_only"
    assert intervention.metadata["type_compatible"] is True
    assert intervention.replacement_event.metadata["dropped_source_event_id"] in target.parents


def test_tco_stop_family_force_continue_and_force_stop_semantics():
    trace = make_trace()
    stop_cf = apply_operator(trace, "e5", "force_continue", random.Random(0))
    assert stop_cf.deleted is True
    assert stop_cf.replacement_event is None
    assert stop_cf.metadata["operator_family"] == "stop"

    forced_stop = apply_operator(trace, "e2", "force_stop", random.Random(0))
    assert forced_stop.replacement_event is not None
    assert forced_stop.replacement_event.type == "stop"
    assert forced_stop.replacement_event.parents == trace.get_event("e2").parents
    assert forced_stop.replacement_event.metadata["counterfactual"] == "force_stop"
    assert forced_stop.metadata["operator_family"] == "stop"
    assert compatible_with_operator("msg", "force_stop") is True
    assert compatible_with_operator("stop", "force_continue") is True
    assert compatible_with_operator("stop", "force_stop") is False


def test_crn_returns_matched_factual_and_counterfactual_seeds():
    pairs = paired_seeds(base_seed=7, k=3)
    assert len(pairs) == 3
    assert [p.factual_seed for p in pairs] == [p.counterfactual_seed for p in pairs]
    assert len({p.factual_seed for p in pairs}) == 3


def test_no_crn_returns_independent_counterfactual_seeds():
    pairs = paired_seeds(base_seed=7, k=3, use_crn=False)
    assert len(pairs) == 3
    assert any(p.factual_seed != p.counterfactual_seed for p in pairs)


def test_teacher_credit_estimates_factual_minus_counterfactual_mean():
    trace = make_trace(score=1.0)
    engine = ReplayEngine(lambda _trace, _seed: 0.25)
    label = estimate_credit(
        trace=trace,
        event_id="e2",
        operator_name="nullify",
        replay_engine=engine,
        k=4,
        seed=1,
        score_source="verifier",
    )
    assert label.delta_mean == 0.75
    assert label.num_rollouts == 4


def test_teacher_credit_abstains_when_replay_raises():
    trace = make_trace(score=1.0)

    def failing_continuation(_trace, _intervention, _seed):
        raise TimeoutError("unit replay timeout")

    engine = ReplayEngine(
        lambda _trace, _seed: 0.25,
        behavior_policy="unit_policy",
        continuation_policy=failing_continuation,
    )
    label = estimate_credit(
        trace=trace,
        event_id="e2",
        operator_name="nullify",
        replay_engine=engine,
        k=2,
        seed=1,
        score_source="verifier",
    )
    assert label.abstained is True
    assert label.delta_mean == 0.0
    assert label.counterfactual_scores == []
    assert label.metadata["abstain_reason"] == "replay_failed"
    assert label.metadata["failed_rollouts"] == 1
    assert "unit replay timeout" in label.metadata["replays"][0]["error"]


def test_prs_records_restored_prefix_and_frozen_policy_metadata():
    trace = make_trace(score=1.0)
    engine = ReplayEngine(lambda _trace, _seed: 0.25, behavior_policy="unit_policy")
    label = estimate_credit(
        trace=trace,
        event_id="e2",
        operator_name="nullify",
        replay_engine=engine,
        k=2,
        seed=5,
        score_source="verifier",
    )
    replay = label.metadata["replays"][0]
    intervention = label.metadata["intervention"]
    assert replay["prefix_hash"] == intervention["metadata"]["prefix_hash"]
    assert replay["prefix_valid"] is True
    assert replay["behavior_policy"] == "unit_policy"
    assert replay["seed"] == replay["counterfactual_seed"]


def test_prs_estimator_records_paired_factual_counterfactual_readouts_and_crn():
    trace = make_trace(score=1.0)

    def scorer(_trace, seed):
        return {11: 0.9, 22: 0.7}.get(seed, 0.5)

    engine = ReplayEngine(scorer, behavior_policy="unit_policy")
    label = estimate_credit(
        trace=trace,
        event_id="e2",
        operator_name="nullify",
        replay_engine=engine,
        k=2,
        seed=3,
        score_source="verifier",
    )
    records = label.metadata["replays"]
    assert len(records) == 2
    assert all(record["factual_seed"] == record["counterfactual_seed"] for record in records)
    assert all(record["paired_difference"] == record["factual_score"] - record["counterfactual_score"] for record in records)
    assert label.metadata["estimator"] == "paired_perturb_rollout"
    assert label.metadata["crn_coupled"] is True
    assert label.metadata["factual_scores"] == [1.0, 1.0]
    assert label.delta_mean == sum(record["paired_difference"] for record in records) / len(records)


def test_top_m_leverage_predictor_records_expected_delta_and_uncertainty():
    trace = make_trace()
    jobs = select_counterfactual_jobs(
        trace,
        top_m=5,
        operators_per_event=1,
        event_selection="top_m",
        prior_scores={"e2": {"expected_abs_delta": 0.1, "uncertainty": 0.9}},
        zeta=2.0,
    )
    e2_job = next(job for job in jobs if job.event_id == "e2")
    assert e2_job.expected_abs_delta == 0.1
    assert e2_job.uncertainty == 0.9
    assert abs(e2_job.leverage - 1.9) < 1e-12
    assert e2_job.metadata["leverage_formula"] == "expected_abs_delta + zeta * uncertainty"


def test_trace_records_explicit_team_state_snapshots():
    runner = MultiAgentRunner()
    task = Task(
        task_id="state-1",
        dataset="humaneval",
        prompt="def add(a, b):",
        tests="assert add(1, 2) == 3",
    )
    trace = runner.run(task, RunnerConfig(seed=7, planner_mode="dynamic"))
    assert len(trace.state_snapshots) == len(trace.events) + 1
    assert trace.state_snapshots[0]["state_index"] == 0
    assert trace.state_snapshots[0]["after_event_id"] is None
    assert trace.state_snapshots[-1]["state_index"] == len(trace.events)
    assert trace.state_snapshots[-1]["after_event_id"] == trace.events[-1].event_id
    assert trace.state_snapshots[-1]["state_hash"] == trace.events[-1].state_after_hash
    assert trace.state_snapshots[-1]["stopped"] is True
    assert trace.state_snapshots[-1]["final_candidate"] == trace.final_answer
    assert any(snapshot["tool_results"] for snapshot in trace.state_snapshots)
    roundtrip = Trace.from_dict(trace.to_dict())
    assert roundtrip.state_snapshots[-1]["state_hash"] == trace.state_snapshots[-1]["state_hash"]


def test_replay_engine_can_use_behavior_policy_downstream_continuation():
    trace = make_trace(score=1.0)
    intervention = apply_operator(trace, "e2", "nullify", random.Random(0))
    calls = []

    def continuation(source_trace, source_intervention, seed):
        calls.append((source_intervention.target_event_id, seed))
        events = list(source_intervention.prefix_events)
        if source_intervention.replacement_event:
            events.append(source_intervention.replacement_event)
        events.extend(
            [
                Event(
                    event_id="cf-agg",
                    trace_id=source_trace.trace_id,
                    task_id=source_trace.task_id,
                    t=len(events),
                    type="aggregate",
                    agent_role="aggregator",
                    agent_id="aggregator-1",
                    content="counterfactual final answer",
                    parents=[events[-1].event_id],
                ),
                Event(
                    event_id="cf-stop",
                    trace_id=source_trace.trace_id,
                    task_id=source_trace.task_id,
                    t=len(events) + 1,
                    type="stop",
                    agent_role="stopper",
                    agent_id="stopper-1",
                    content="Stop.",
                    parents=["cf-agg"],
                ),
            ]
        )
        return source_trace.clone_with_events(events, final_answer="counterfactual final answer")

    engine = ReplayEngine(lambda _trace, _seed: 0.25, behavior_policy="api_role_conditioned", continuation_policy=continuation)
    replay = engine.replay(trace, intervention, seed=99)
    assert calls == [("e2", 99)]
    assert replay.metadata["replay_mode"] == "behavior_policy_continuation"
    assert replay.metadata["prefix_held_fixed"] is True
    assert replay.replayed_trace.events[0].event_id == trace.events[0].event_id
    assert replay.replayed_trace.events[-1].event_id == "cf-stop"


def test_behavior_replay_continuation_restores_task_and_replays_downstream_policy():
    runner = MultiAgentRunner()
    task = Task(
        task_id="behavior-replay-1",
        dataset="humaneval",
        prompt="def add(a, b):",
        tests="assert add(1, 2) == 3",
    )
    trace = runner.run(task, RunnerConfig(seed=8, planner_mode="dynamic"))
    trace.manifest["task"] = {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "prompt": task.prompt,
        "tests": task.tests,
        "metadata": task.metadata,
    }
    target = next(event for event in trace.events if event.type == "msg")
    intervention = apply_operator(trace, target.event_id, "nullify", random.Random(0))
    engine = ReplayEngine(
        lambda _trace, _seed: 0.0,
        behavior_policy="api_role_conditioned",
        continuation_policy=build_behavior_continuation_policy(use_api=False),
    )
    replay = engine.replay(trace, intervention, seed=123)
    assert replay.metadata["replay_mode"] == "behavior_policy_continuation"
    assert replay.metadata["prefix_held_fixed"] is True
    assert replay.replayed_trace.events[: len(intervention.prefix_events)] == intervention.prefix_events
    assert any(event.metadata.get("counterfactual_replay") for event in replay.replayed_trace.events)
    assert replay.replayed_trace.state_snapshots[-1]["state_hash"] == replay.metadata["terminal_state_hash"]


def make_gsm8k_continuation_trace():
    task = Task("gsm8k_cf", "gsm8k", "A seller has 9 items at $2 each. How much?", reference="18")
    events = [
        Event("e1", "gsm8k_cf-trace-0", task.task_id, 0, "msg", "solver_a", "solver_a-1", "9 * 2 = 18\nFinal answer: 18"),
        Event("e2", "gsm8k_cf-trace-0", task.task_id, 1, "aggregate", "aggregator", "aggregator-1", "Selected_candidate: solver_a\nFinal answer: 18", parents=["e1"]),
        Event("e3", "gsm8k_cf-trace-0", task.task_id, 2, "stop", "stopper", "stopper-1", "Decision: STOP", parents=["e2"]),
    ]
    return Trace(
        trace_id="gsm8k_cf-trace-0",
        task_id=task.task_id,
        dataset=task.dataset,
        split="test",
        events=events,
        final_answer=events[1].content,
        verifier_score=1.0,
        success=True,
        manifest={
            "prompt_version": "gsm8k_stable_v1",
            "planner_mode": "static",
            "max_retries": 0,
            "task": {"task_id": task.task_id, "dataset": task.dataset, "prompt": task.prompt, "reference": task.reference, "metadata": {}},
        },
    )


def test_credit_uses_factual_replay_cost_for_stop_signal():
    trace = make_gsm8k_continuation_trace()
    engine = ReplayEngine(
        prs_score_for_trace,
        behavior_policy="deterministic_downstream",
        continuation_policy=build_behavior_continuation_policy(use_api=False),
    )
    label = estimate_credit(trace, "e1", "force_stop", engine, 1, 0, "verifier", operator_set="gsm8k_v1")
    assert label.metadata["factual_cost"] > trace.total_cost_usd
    assert label.metadata["factual_costs"] == [label.metadata["factual_cost"]]


def test_force_stop_signal_values_saved_cost_against_forgone_gain():
    signal = compute_stop_signal(
        event_type="non_stop",
        factual_score=1.0,
        counterfactual_scores=[1.0],
        factual_cost=0.5,
        counterfactual_costs=[0.2],
    )
    assert signal.saved_cost == 0.3
    assert signal.forgone_gain == 0.0
    assert signal.stop_reward == 0.3


def test_behavior_continuation_replays_only_original_downstream_roles():
    trace = make_gsm8k_continuation_trace()
    intervention = apply_operator(trace, "e2", "wrong_final_number_aggregate_gsm8k", random.Random(0), operator_set="gsm8k_v1")
    replayed = build_behavior_continuation_policy(use_api=False)(trace, intervention, 7)
    suffix = replayed.events[len(intervention.prefix_events) + 1 :]
    assert [event.agent_role for event in suffix] == ["stopper"]
    assert replayed.final_answer.endswith("Final answer: 19")
    assert replayed.verifier_score == 0.0


def test_behavior_continuation_force_stop_truncates_suffix():
    trace = make_gsm8k_continuation_trace()
    intervention = apply_operator(trace, "e1", "force_stop", random.Random(0), operator_set="gsm8k_v1")
    replayed = build_behavior_continuation_policy(use_api=False)(trace, intervention, 7)
    assert len(replayed.events) == 1
    assert replayed.events[-1].type == "stop"
    assert replayed.events[-1].metadata["counterfactual"] == "force_stop"


def test_calibrated_oracle_isotonic_and_abstention():
    oracle = CalibratedOracle(alpha=0.25)
    oracle.fit(raw_scores=[0.1, 0.3, 0.8, 0.9], targets=[0.0, 0.0, 1.0, 1.0], dispersions=[0.01, 0.02, 0.05, 0.2])
    confident = oracle.score([0.8, 0.9], calibration_version="v1")
    uncertain = oracle.score([0.1, 0.9], calibration_version="v1")
    assert confident.calibrated_score > 0.8
    assert confident.abstain is False
    assert uncertain.abstain is True


def test_reward_composition_and_conservation_are_finite():
    event = make_trace().events[1]
    reward = compose_reward(
        event,
        delta=0.5,
        phi_before=0.2,
        phi_after=0.5,
        grounding=1.0,
        redundancy=0.1,
        contradiction=0.0,
        stop_reward=0.0,
        weights=RewardWeights(),
    )
    assert reward.total_reward > 0.0
    assert abs(conservation_error([0.2, 0.3], outcome=1.0, empty_baseline=0.4)) < 1.0


def test_reward_label_builder_composes_method_components():
    trace = make_trace()
    credit_by_event = {
        "e2": {"delta": 0.5, "score_source": "verifier"},
        "e5": {"delta": 0.1, "score_source": "verifier"},
    }
    stop_signals = {"e5": {"stop_reward": 0.2}}
    labels = build_reward_labels(trace, credit_by_event, stop_signals, RewardWeights())
    by_event = {label.event_id: label for label in labels}
    assert by_event["e2"].delta == 0.5
    assert by_event["e2"].grounding == 0.0
    assert by_event["e5"].stop_reward == 0.2
    assert by_event["e5"].total_reward != by_event["e5"].delta
    delta_only = build_reward_labels(trace, credit_by_event, stop_signals, RewardWeights(), reward_mode="delta_only")
    delta_by_event = {label.event_id: label for label in delta_only}
    assert delta_by_event["e5"].total_reward == delta_by_event["e5"].delta


def test_reward_component_ablations_only_remove_the_target_component():
    trace = make_trace()
    credit_by_event = {"e2": {"delta": 0.5, "score_source": "verifier"}}
    stop_signals = {"e5": {"stop_reward": 0.2}}
    weights = RewardWeights(gamma=0.9)

    full = {label.event_id: label for label in build_reward_labels(trace, credit_by_event, stop_signals, weights)}
    no_potential = {
        label.event_id: label
        for label in build_reward_labels(
            trace,
            credit_by_event,
            stop_signals,
            weights,
            disable_potential_shaping=True,
        )
    }
    no_stop = {
        label.event_id: label
        for label in build_reward_labels(
            trace,
            credit_by_event,
            stop_signals,
            weights,
            disable_stopping_reward=True,
        )
    }

    for event_id, full_label in full.items():
        potential_label = no_potential[event_id]
        stop_label = no_stop[event_id]
        assert potential_label.progress == 0.0
        assert potential_label.stop_reward == full_label.stop_reward
        assert potential_label.delta == full_label.delta
        assert potential_label.grounding == full_label.grounding
        assert potential_label.redundancy_penalty == full_label.redundancy_penalty
        assert potential_label.contradiction_penalty == full_label.contradiction_penalty
        assert potential_label.cost_penalty == full_label.cost_penalty
        assert stop_label.stop_reward == 0.0
        assert stop_label.progress == full_label.progress
        assert stop_label.delta == full_label.delta
        assert stop_label.grounding == full_label.grounding
        assert stop_label.redundancy_penalty == full_label.redundancy_penalty
        assert stop_label.contradiction_penalty == full_label.contradiction_penalty
        assert stop_label.cost_penalty == full_label.cost_penalty


def test_reward_progress_uses_explicit_graph_state_potential():
    trace = make_trace()
    credit_by_event = {
        "e2": {"delta": 0.5, "score_source": "verifier"},
        "e4": {"delta": 0.2, "score_source": "verifier"},
    }
    labels = build_reward_labels(trace, credit_by_event, {}, RewardWeights(gamma=0.9))
    by_event = {label.event_id: label for label in labels}
    e2 = by_event["e2"]
    assert e2.metadata["potential_source"] == "state_value_estimate"
    assert e2.metadata["state_before_hash"] == trace.events[1].state_before_hash
    assert e2.metadata["state_after_hash"] == trace.events[1].state_after_hash
    assert e2.metadata["phi_after"] >= e2.metadata["phi_before"]
    assert abs(e2.progress - (0.9 * e2.metadata["phi_after"] - e2.metadata["phi_before"])) < 1e-12


def test_reward_cost_and_stop_metadata_match_method_formula():
    trace = make_trace()
    tool = trace.events[1]
    tool.type = "tool"
    tool.tokens_in = 10
    tool.tokens_out = 5
    tool.latency_ms = 20.0
    tool.metadata["tool_name"] = "unit_tests"
    stop_signals = {
        "e5": {
            "stop_reward": 0.14,
            "saved_cost": 0.14,
            "forgone_gain": 0.0,
            "continued_score": 0.8,
            "factual_score": 0.8,
        }
    }
    labels = build_reward_labels(trace, {}, stop_signals, RewardWeights())
    by_event = {label.event_id: label for label in labels}
    assert by_event["e2"].metadata["cost_components"]["tool_call_cost"] > 0.0
    assert by_event["e2"].cost_penalty == sum(by_event["e2"].metadata["cost_components"].values())
    assert by_event["e5"].metadata["stop_signal"]["saved_cost"] == 0.14
    assert by_event["e5"].metadata["stop_signal"]["forgone_gain"] == 0.0


def test_event_mean_credit_preserves_shared_effect_and_conservation_rescaling():
    labels = [
        CreditLabel("tr", "e1", "msg", "nullify", 1.0, [0.5], 0.5, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e1", "msg", "delete", 1.0, [0.8], 0.2, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e3", "stop", "force_continue", 1.0, [0.9], 0.1, 0.0, 1, False, "verifier"),
    ]
    adjusted = apply_family_baseline_and_rescale(labels, outcome=1.0, empty_baseline=0.4)
    by_event = {label.event_id: label for label in adjusted}
    assert by_event["e1"].metadata["credit_aggregation_unit"] == "event"
    assert by_event["e1"].metadata["event_label_count"] == 2
    assert by_event["e1"].metadata["event_raw_delta"] == 0.35
    assert by_event["e1"].metadata["family_loo_baseline"] == 0.0
    assert by_event["e1"].metadata["rescaled_delta"] == adjusted[1].metadata["rescaled_delta"]
    unique_event_rescaled = {label.event_id: label.metadata["rescaled_delta"] for label in adjusted}
    assert abs(sum(unique_event_rescaled.values()) - 0.6) < 1e-9
    assert all(label.metadata["conservation_satisfied"] is True for label in adjusted)
    assert all("conservation_scale" in label.metadata for label in adjusted)

def test_event_aggregation_does_not_cancel_same_family_effects():
    labels = [
        CreditLabel("tr", "e1", "aggregate", "op_a", 1.0, [0.0], 1.0, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e2", "aggregate", "op_b", 1.0, [0.0], 1.0, 0.0, 1, False, "verifier"),
    ]
    adjusted = apply_family_baseline_and_rescale(labels, outcome=1.0, empty_baseline=0.0)
    assert [label.metadata["rescaled_delta"] for label in adjusted] == [0.5, 0.5]
    assert all(label.metadata["conservation_status"] == "rescaled" for label in adjusted)



def test_conservation_records_unrescalable_zero_signal():
    labels = [
        CreditLabel("tr", "e1", "msg", "nullify", 1.0, [1.0], 0.0, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e2", "stop", "force_continue", 1.0, [1.0], 0.0, 0.0, 1, False, "verifier"),
    ]
    adjusted = apply_family_baseline_and_rescale(labels, outcome=1.0, empty_baseline=0.0)
    assert sum(label.metadata["rescaled_delta"] for label in adjusted) == 0.0
    assert adjusted[0].metadata["conservation_status"] == "raw_zero_unrescalable"
    assert adjusted[1].metadata["conservation_status"] == "excluded_stop_channel"
    assert adjusted[0].metadata["conservation_satisfied"] is False


def test_student_graph_arrays_and_forward_shape():
    trace = make_trace()
    arrays = graph_arrays_from_trace(trace)
    model = NumpyGraphStudent(num_event_types=10, hidden_dim=8, seed=0)
    preds = model.forward(arrays)
    assert len(preds) == len(trace.events)


def test_relational_student_uses_typed_edges_roles_and_uncertainty():
    trace = make_trace()
    arrays = graph_arrays_from_trace(trace)
    assert len(arrays.edge_type) == len(arrays.edge_index)
    assert len(arrays.role_id) == len(trace.events)
    assert len(arrays.text_features) == len(trace.events)
    model = RelationalGraphStudent(
        num_event_types=10,
        num_roles=max(arrays.role_id) + 1,
        num_edge_types=max(arrays.edge_type) + 1,
        hidden_dim=8,
        seed=0,
    )
    preds = model.forward(arrays)
    uncertainty = model.uncertainty(arrays)
    assert len(preds) == len(trace.events)
    assert len(uncertainty) == len(trace.events)
    assert all(value >= 0.0 for value in uncertainty)


def test_embedding_backend_populates_graph_text_features():
    trace = make_trace()
    backend = HashEmbeddingBackend(dim=12, model_name="hash-test")
    arrays = graph_arrays_from_trace(trace, embedding_backend=backend)
    assert len(arrays.text_features[0]) == 12
    embedded = embed_trace_events(trace, backend)
    assert embedded.model_name == "hash-test"
    assert len(embedded.vectors) == len(trace.events)
    qwen = QwenEmbeddingBackend(model_name="Qwen/Qwen3-Embedding-0.6B", fallback=backend)
    qwen_vectors = qwen.embed(["hello", "world"])
    assert len(qwen_vectors) == 2
    assert len(qwen_vectors[0]) == 12


def test_train_relational_student_updates_head_and_checkpoint(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        trace = make_trace(score=1.0)
        labels = {event.event_id: float(i) / 10.0 for i, event in enumerate(trace.events)}
        checkpoint = Path(tmp) / "student_checkpoint.json"
        result = train_relational_student([trace], labels, checkpoint_path=checkpoint, epochs=3, hidden_dim=8, seed=0)
        assert checkpoint.exists()
        assert result["model"] == "relational_rgAT"
        assert result["embedding_backend"] == "hash"
        assert result["train_steps"] == 3
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert "out" in saved
        assert saved["embedding_backend"] == "hash"


def test_train_relational_student_uses_method_losses_and_abstention_mask(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        trace = make_trace(score=1.0)
        labels = {event.event_id: 0.0 for event in trace.events}
        labels["e2"] = 1.0
        labels["e3"] = -1.0
        labels["e5"] = 10.0
        checkpoint = Path(tmp) / "student_method_checkpoint.json"
        result = train_relational_student(
            [trace],
            labels,
            checkpoint_path=checkpoint,
            epochs=30,
            hidden_dim=8,
            seed=0,
            ranking_pairs=[("e2", "e3")],
            abstained_event_ids={"e5"},
            ranking_beta=0.5,
        )
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert saved["loss"]["regression"] >= 0.0
        assert saved["loss"]["ranking"] >= 0.0
        assert saved["loss"]["masked_events"] == 1
        assert saved["loss"]["ranking_pairs"] == 1
        assert result["masked_events"] == 1
        assert result["ranking_pairs"] == 1
        predictions = saved["train_predictions"]
        assert predictions["e2"] > predictions["e3"]
        assert "e5" not in predictions


def test_carve_s_checkpoint_records_distillation_method_metadata(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        trace = make_trace(score=1.0)
        targets = {"e2": 1.0, "e3": -1.0}
        checkpoint = Path(tmp) / "student_checkpoint.json"
        backend = QwenEmbeddingBackend(model_name="Qwen/Qwen3.5-Embedding", fallback=HashEmbeddingBackend(dim=8, model_name="hash-qwen-fallback"))
        result = train_relational_student(
            [trace],
            targets,
            checkpoint_path=checkpoint,
            embedding_backend=backend,
            epochs=2,
            hidden_dim=8,
            ranking_pairs=[("e2", "e3")],
            abstained_event_ids={"e5"},
        )
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        assert saved["student_name"] == "CARVE-S"
        assert saved["initialized_from"] is None
        assert saved["architecture"] == "qwen_embedding_relational_graph_attention"
        assert saved["graph_inputs"] == ["typed_events", "causal_parents", "roles", "text_embeddings", "event_features"]
        assert saved["loss"]["regression_name"] == "Huber"
        assert saved["loss"]["ranking_name"] == "Bradley-Terry"
        assert saved["loss"]["abstention_masked"] is True
        assert saved["deployment_uses"] == ["event_selection_pruning", "early_stopping", "dense_rewards_for_policy_optimization"]
        assert result["student_name"] == "CARVE-S"


def test_control_and_rl_export_interfaces():
    trace_a = make_trace(score=1.0)
    trace_b = make_trace(score=0.0)
    event_scores = {"e1": 0.1, "e2": -0.5, "e3": 0.2, "e4": 0.3, "e5": 0.0}
    pruned = prune_negative_events(trace_a, event_scores)
    assert "e2" not in [e.event_id for e in pruned.events]
    ranked = rerank_traces([trace_b, trace_a], {"tr1": 0.5, "other": 0.0})
    assert ranked[0].trace_id == "tr1"
    assert should_stop(saved_cost=0.4, forgone_gain=0.1, threshold=0.0)
    samples = export_rl_samples(trace_a, event_scores)
    assert samples[0].reward == event_scores[samples[0].metadata["event_id"]]


def test_pruning_preserves_answer_sink_when_its_score_is_negative():
    trace = make_trace(score=1.0)
    event_scores = {event.event_id: -1.0 for event in trace.events}

    pruned = prune_negative_events(trace, event_scores)

    assert any(event.type == "aggregate" for event in pruned.events)
    assert pruned.events[-1].type == "stop"


def test_control_loads_teacher_rewards_for_pruning_and_reranking(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps({"event_id": "e1", "delta_mean": 0.1, "metadata": {"rescaled_delta": 0.4}}) + "\n"
            + json.dumps({"event_id": "e2", "delta_mean": -0.2, "metadata": {"adjusted_delta": -0.3}}) + "\n",
            encoding="utf-8",
        )
        event_scores, metadata = load_control_event_scores(run_dir)
        assert event_scores == {"e1": 0.4, "e2": -0.3}
        assert metadata["score_source"] == "teacher_credit_labels"
        trace = make_trace(score=1.0)
        assert abs(score_trace_for_control(trace, event_scores) - 0.1) < 1e-9
        (run_dir / "reward_labels.jsonl").write_text(
            json.dumps({"event_id": "e1", "total_reward": 1.25}) + "\n",
            encoding="utf-8",
        )
        reward_scores, reward_metadata = load_control_event_scores(run_dir)
        assert reward_scores == {"e1": 1.25}
        assert reward_metadata["score_source"] == "reward_labels"


def test_reward_and_control_keys_are_trace_scoped_for_multi_trace_runs(tmp_path=None):
    trace_a = make_trace(score=1.0)
    trace_b = make_trace(score=0.0)
    trace_b.trace_id = "tr2"
    for event in trace_b.events:
        event.trace_id = "tr2"
    credit_by_event = {
        "tr1::e2": {"delta": 0.5, "score_source": "verifier"},
        "tr2::e2": {"delta": -0.5, "score_source": "verifier"},
    }
    labels_a = build_reward_labels(trace_a, credit_by_event, {}, RewardWeights())
    labels_b = build_reward_labels(trace_b, credit_by_event, {}, RewardWeights())
    assert next(label for label in labels_a if label.event_id == "e2").delta == 0.5
    assert next(label for label in labels_b if label.event_id == "e2").delta == -0.5

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "reward_labels.jsonl").write_text(
            json.dumps({"trace_id": "tr1", "event_id": "e2", "total_reward": 0.5}) + "\n"
            + json.dumps({"trace_id": "tr2", "event_id": "e2", "total_reward": -0.5}) + "\n",
            encoding="utf-8",
        )
        scores, metadata = load_control_event_scores(run_dir)
        assert scores["tr1::e2"] == 0.5
        assert scores["tr2::e2"] == -0.5
        assert metadata["scored_events"] == 2
        assert score_trace_for_control(trace_a, scores) == 0.5
        assert score_trace_for_control(trace_b, scores) == -0.5


def test_stop_counterfactual_signal_uses_force_continue_credit_labels(tmp_path=None):
    signal = compute_stop_signal(
        event_type="stop",
        factual_score=0.8,
        counterfactual_scores=[0.9, 0.7],
        factual_cost=0.30,
        counterfactual_costs=[0.42, 0.46],
        beta_cost=1.0,
    )
    assert abs(signal.saved_cost - 0.14) < 1e-9
    assert abs(signal.forgone_gain - 0.0) < 1e-9
    assert abs(signal.stop_reward - 0.14) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps(
                {
                    "event_id": "e5",
                    "operator_family": "stop",
                    "operator_name": "force_continue",
                    "factual_score": 0.8,
                    "counterfactual_scores": [0.9, 0.7],
                    "metadata": {
                        "factual_cost": 0.3,
                        "counterfactual_costs": [0.42, 0.46],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        signals = load_stop_signals(run_dir)
        assert signals["e5"]["operator_name"] == "force_continue"
        assert abs(signals["e5"]["stop_reward"] - 0.14) < 1e-9


def test_rl_export_records_grpo_group_normalized_ppo_metadata():
    trace = make_trace(score=1.0)
    event_scores = {"e1": -1.0, "e2": 0.0, "e3": 1.0, "e4": 2.0, "e5": 3.0}
    samples = export_rl_samples(trace, event_scores, reward_source="CARVE-S", policy_objective="ppo")
    advantages = [sample.advantage for sample in samples]
    assert abs(sum(advantages) / len(advantages)) < 1e-9
    assert abs((sum(a * a for a in advantages) / len(advantages)) - 1.0) < 1e-9
    assert all(sample.metadata["advantage_normalization"] == "group_zscore" for sample in samples)
    assert all(sample.metadata["grpo_group_id"] == trace.trace_id for sample in samples)
    assert all(sample.metadata["policy_objective"] == "ppo" for sample in samples)
    assert all(sample.metadata["reward_source"] == "CARVE-S" for sample in samples)
    assert all("group_reward_mean" in sample.metadata for sample in samples)
    assert all("group_reward_std" in sample.metadata for sample in samples)
    grpo_samples = export_rl_samples(trace, event_scores, reward_source="CARVE-S", policy_objective="grpo")
    assert all(sample.metadata["policy_objective"] == "grpo" for sample in grpo_samples)
    assert all(sample.metadata["compatible_objectives"] == ["ppo", "grpo"] for sample in grpo_samples)
    assert all(sample.metadata["policy_base_model"] == "Qwen-3.5 orchestrator" for sample in grpo_samples)


def test_ppo_smoke_metrics_compute_clipped_surrogate_and_kl_placeholder():
    samples = export_rl_samples(make_trace(), {"e1": -1.0, "e2": 0.0, "e3": 1.0, "e4": 2.0, "e5": 3.0})
    metrics = compute_ppo_smoke_metrics(samples, clip_epsilon=0.2, kl_beta=0.1)
    assert metrics["samples"] == len(samples)
    assert metrics["objective"] == metrics["surrogate_mean"] - 0.1 * metrics["kl_mean"]
    assert metrics["clip_epsilon"] == 0.2
    assert metrics["policy_update"] == "smoke_no_weight_update"
    assert metrics["old_logprob_available"] is False


def test_math_verifier_exact_numeric_match():
    verifier = MathVerifier()
    score = verifier.verify("The final answer is 42.", reference="42")
    assert score.success is True
    assert score.score == 1.0


def test_code_verifier_extracts_markdown_python_and_runs_tests():
    text = "Here is code:\n```python\ndef add(a, b):\n    return a + b\n```"
    code = extract_python_code(text)
    assert "def add" in code
    score = CodeVerifier().verify(code, "assert add(2, 3) == 5")
    assert score.success is True


def test_swebench_verifier_applies_patch_and_runs_repo_tests():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        (repo / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (repo / "test_buggy.py").write_text(
            "import unittest\nfrom buggy import add\n\n"
            "class AddTest(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n",
            encoding="utf-8",
        )
        patch = (
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        result = SWEBenchVerifier(work_root=Path(tmp) / "work").verify(
            patch,
            tests="python -m unittest test_buggy.py",
            repo_path=str(repo),
        )
        assert result.success is True
        assert result.details["mode"] == "repo_test"
        assert result.details["returncode"] == 0


def test_apply_patch_to_repo_rejects_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        patch = "--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-old\n+new\n"
        try:
            apply_patch_to_repo(repo, patch)
        except FileNotFoundError as exc:
            assert "missing.py" in str(exc)
        else:
            raise AssertionError("expected missing file failure")


def test_prepare_swebench_instances_clones_and_writes_repo_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "source_repo"
        source.mkdir()
        (source / "buggy.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        import subprocess

        subprocess.run(["git", "init"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["git", "config", "user.email", "anonymous@example.com"], cwd=source, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
        subprocess.run(["git", "add", "buggy.py"], cwd=source, check=True)
        subprocess.run(["git", "commit", "-m", "initial"], cwd=source, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        input_path = root / "swe.jsonl"
        output_path = root / "prepared.jsonl"
        cache_dir = root / "repos"
        input_path.write_text(
            json.dumps(
                {
                    "instance_id": "local__repo-1",
                    "problem_statement": "fix",
                    "repo": "local/repo",
                    "repo_url": str(source),
                    "base_commit": commit,
                    "test_command": "python -m unittest",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary = prepare_swebench_instances(input_path, output_path, cache_dir, limit=1)
        row = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
        assert summary["prepared"] == 1
        assert Path(row["repo_path"]).exists()
        assert row["base_commit"] == commit


def test_counterfactual_scorer_uses_dataset_specific_verifier():
    trace = make_trace(score=0.0)
    trace.dataset = "gsm8k"
    trace.final_answer = "Final answer: 42"
    trace.manifest["task"] = {"reference": "42"}
    assert verifier_score_for_trace(trace) == 1.0

    code_trace = make_trace(score=0.0)
    code_trace.dataset = "humaneval"
    code_trace.final_answer = "```python\ndef add(a,b):\n    return a+b\n```"
    code_trace.manifest["task"] = {"tests": "assert add(1, 2) == 3"}
    assert verifier_score_for_trace(code_trace) == 1.0


def test_prs_scorer_preserves_verified_final_answer_when_aggregate_is_markdown():
    task = {
        "task_id": "HumanEval/x",
        "dataset": "humaneval",
        "prompt": "def answer():",
        "tests": "def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    }
    trace = Trace(
        trace_id="tr-prs",
        task_id="HumanEval/x",
        dataset="humaneval",
        split="test",
        events=[
            Event(
                event_id="e1",
                trace_id="tr-prs",
                task_id="HumanEval/x",
                t=0,
                type="msg",
                agent_role="solver",
                agent_id="solver-1",
                content="```python\ndef answer():\n    return 42\n```",
                parents=[],
            ),
            Event(
                event_id="e2",
                trace_id="tr-prs",
                task_id="HumanEval/x",
                t=1,
                type="aggregate",
                agent_role="aggregator",
                agent_id="aggregator-1",
                content="The verified answer is above, and all tests pass.",
                parents=["e1"],
            ),
        ],
        final_answer="```python\ndef answer():\n    return 42\n```",
        verifier_score=1.0,
        success=True,
        manifest={"task": task},
    )
    assert prs_score_for_trace(trace, seed=0) == 1.0


def test_prs_scorer_only_scores_terminal_counterfactual_final_answer():
    task = {
        "task_id": "HumanEval/x",
        "dataset": "humaneval",
        "prompt": "def answer():",
        "tests": "def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    }
    trace = Trace(
        trace_id="tr-prs-terminal",
        task_id="HumanEval/x",
        dataset="humaneval",
        split="test",
        events=[
            Event(
                event_id="e1",
                trace_id="tr-prs-terminal",
                task_id="HumanEval/x",
                t=0,
                type="msg",
                agent_role="solver",
                agent_id="solver-1",
                content="```python\ndef answer():\n    return 42\n```",
                parents=[],
            ),
            Event(
                event_id="e2",
                trace_id="tr-prs-terminal",
                task_id="HumanEval/x",
                t=1,
                type="aggregate",
                agent_role="aggregator",
                agent_id="aggregator-1",
                content="```python\ndef answer():\n    return 0\n```",
                parents=["e1"],
            ),
        ],
        final_answer="```python\ndef answer():\n    return 0\n```",
        verifier_score=0.0,
        success=False,
        manifest={"task": task},
    )
    assert prs_score_for_trace(trace, seed=0) == 0.0


def test_structural_replay_reaggregates_from_surviving_candidates_without_original_answer_leak():
    task = {
        "task_id": "HumanEval/x",
        "dataset": "humaneval",
        "prompt": "def answer():",
        "tests": "def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    }
    trace = Trace(
        trace_id="tr-reagg",
        task_id="HumanEval/x",
        dataset="humaneval",
        split="test",
        events=[
            Event("e1", "tr-reagg", "HumanEval/x", 0, "msg", "solver_a", "solver-a", "```python\ndef answer():\n    return 42\n```"),
            Event("e2", "tr-reagg", "HumanEval/x", 1, "msg", "solver_b", "solver-b", "```python\ndef answer():\n    return 0\n```"),
            Event("e3", "tr-reagg", "HumanEval/x", 2, "aggregate", "aggregator", "agg", "```python\ndef answer():\n    return 42\n```", ["e1", "e2"]),
            Event("e4", "tr-reagg", "HumanEval/x", 3, "stop", "stopper", "stop", "Stop.", ["e3"]),
        ],
        final_answer="```python\ndef answer():\n    return 42\n```",
        verifier_score=1.0,
        success=True,
        manifest={"task": task},
    )
    intervention = apply_operator(trace, "e1", "delete", random.Random(0))
    replay = ReplayEngine(prs_score_for_trace).replay(trace, intervention, seed=0)
    assert replay.replayed_trace.final_answer == "```python\ndef answer():\n    return 0\n```"
    assert replay.score == 0.0


def test_code_specific_destructive_operator_replaces_passing_code_with_failing_code():
    task = {
        "task_id": "HumanEval/x",
        "dataset": "humaneval",
        "prompt": "def answer():",
        "tests": "def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    }
    trace = Trace(
        trace_id="tr-code-op",
        task_id="HumanEval/x",
        dataset="humaneval",
        split="test",
        events=[
            Event("e1", "tr-code-op", "HumanEval/x", 0, "msg", "solver", "solver-1", "```python\ndef answer():\n    return 42\n```"),
        ],
        final_answer="```python\ndef answer():\n    return 42\n```",
        verifier_score=1.0,
        success=True,
        manifest={"task": task},
    )
    intervention = apply_operator(trace, "e1", "wrong_return_code", random.Random(0))
    assert intervention.replacement_event is not None
    assert CodeVerifier().verify(intervention.replacement_event.content, task["tests"]).score == 0.0


def test_typed_counterfactual_job_selection_respects_event_types_and_top_m():
    trace = make_trace()
    jobs = select_counterfactual_jobs(trace, top_m=3, operators_per_event=2)
    assert len({job.event_id for job in jobs}) == 3
    assert all(job.operator_name in {"reroute_agent", "skip_delegate"} for job in jobs if job.event_type == "delegate")
    assert all(job.operator_name in {"delete", "nullify", "reroute"} for job in jobs if job.event_type == "msg")
    assert any(job.event_type == "stop" for job in jobs)
    no_stop = filter_counterfactual_jobs(jobs, stop_counterfactual=False)
    assert all(job.event_type != "stop" for job in no_stop)
    random_jobs = select_counterfactual_jobs(trace, top_m=3, operators_per_event=1, event_selection="random", seed=4)
    top_jobs = select_counterfactual_jobs(trace, top_m=3, operators_per_event=1, event_selection="top_m", seed=4)
    assert [job.event_id for job in random_jobs] != [job.event_id for job in top_jobs]


def test_type_stratified_counterfactual_selection_covers_lower_ranked_event_types():
    trace = Trace(
        trace_id="tr-stratified",
        task_id="task",
        dataset="humaneval",
        split="test",
        events=[
            Event("e1", "tr-stratified", "task", 0, "spawn", "orchestrator", "orch", "spawn"),
            Event("e2", "tr-stratified", "task", 1, "assign", "planner", "planner", "assign", ["e1"]),
            Event("e3", "tr-stratified", "task", 2, "delegate", "orchestrator", "orch", "delegate", ["e2"]),
            Event("e4", "tr-stratified", "task", 3, "msg", "solver_a", "solver", "```python\ndef answer():\n    return 42\n```", ["e3"]),
            Event("e5", "tr-stratified", "task", 4, "tool", "tester", "tester", "pass", ["e4"]),
            Event("e6", "tr-stratified", "task", 5, "obs", "test_observer", "obs", "pass", ["e5"]),
            Event("e7", "tr-stratified", "task", 6, "critique", "critic", "critic", "looks ok", ["e4", "e6"]),
            Event("e8", "tr-stratified", "task", 7, "revise", "reviser", "reviser", "```python\ndef answer():\n    return 42\n```", ["e4", "e7"]),
            Event("e9", "tr-stratified", "task", 8, "aggregate", "aggregator", "agg", "```python\ndef answer():\n    return 42\n```", ["e4", "e8"]),
            Event("e10", "tr-stratified", "task", 9, "stop", "stopper", "stop", "stop", ["e9"]),
        ],
        final_answer="```python\ndef answer():\n    return 42\n```",
        verifier_score=1.0,
        success=True,
        manifest={"task": {"tests": "def check(candidate):\n    assert candidate() == 42\ncheck(answer)"}},
    )
    jobs = select_counterfactual_jobs(trace, top_m=8, operators_per_event=1, event_selection="type_stratified")
    selected_types = {job.event_type for job in jobs}
    assert {"msg", "delegate", "critique", "revise"}.issubset(selected_types)


def test_runner_selects_dataset_specific_oeg_workflows():
    runner = MultiAgentRunner()
    swe_task = Task(
        task_id="swe-1",
        dataset="swebench_lite",
        prompt="Fix the failing repository test.",
        tests="python -m unittest",
        metadata={"repo_path": "/tmp/repo"},
    )
    swe_trace = runner.run(swe_task, RunnerConfig(seed=2, max_turns=12))
    swe_roles = [event.agent_role for event in swe_trace.events]
    assert "repo_inspector" in swe_roles
    assert "patcher" in swe_roles
    assert "patch_reviser" in swe_roles
    assert swe_trace.manifest["workflow"] == "swebench"
    assert any(event.type == "obs" for event in swe_trace.events)
    aggregate = next(event for event in swe_trace.events if event.type == "aggregate")
    assert len(aggregate.parents) >= 3

    open_task = Task(task_id="open-1", dataset="research_synthesis_qa", prompt="Synthesize evidence.")
    open_trace = runner.run(open_task, RunnerConfig(seed=3, max_turns=10))
    open_roles = [event.agent_role for event in open_trace.events]
    assert "researcher_a" in open_roles
    assert "researcher_b" in open_roles
    assert open_trace.manifest["workflow"] == "openqa"
    assert len(next(event for event in open_trace.events if event.type == "aggregate").parents) >= 3


def test_dynamic_runner_plans_branches_tools_and_retries():
    runner = MultiAgentRunner()
    task = Task(
        task_id="dyn-1",
        dataset="humaneval",
        prompt="def add(a, b):",
        tests="assert add(1, 2) == 3",
    )
    trace = runner.run(task, RunnerConfig(seed=4, max_turns=14, planner_mode="dynamic"))
    assert trace.manifest["planner_mode"] == "dynamic"
    planner_events = [event for event in trace.events if event.agent_role == "planner"]
    assert planner_events
    assert any(event.metadata.get("planned_next_roles") for event in planner_events)
    tool_events = [event for event in trace.events if event.type == "tool"]
    assert tool_events
    assert all("tool_name" in event.metadata for event in tool_events)
    assert any("verifier_score" in event.metadata for event in tool_events)
    roles = [event.agent_role for event in trace.events]
    assert "critic" in roles
    assert "reviser" in roles
    assert any(event.metadata.get("retry_of") for event in trace.events)
    assert trace.validate_graph() is None


def test_dynamic_runner_records_full_orchestration_taxonomy():
    runner = MultiAgentRunner()
    task = Task(
        task_id="dyn-taxonomy",
        dataset="humaneval",
        prompt="def add(a, b):",
        tests="assert add(1, 2) == 3",
    )
    trace = runner.run(task, RunnerConfig(seed=5, max_turns=18, planner_mode="dynamic"))
    event_types = {event.type for event in trace.events}
    assert {"spawn", "assign", "delegate", "msg", "tool", "obs", "aggregate", "stop"}.issubset(event_types)

    spawn = next(event for event in trace.events if event.type == "spawn")
    assign = next(event for event in trace.events if event.type == "assign")
    delegate = next(event for event in trace.events if event.type == "delegate")
    first_msg = next(event for event in trace.events if event.type == "msg")
    assert assign.parents == [spawn.event_id]
    assert delegate.parents == [assign.event_id]
    assert first_msg.parents == [delegate.event_id]
    assert trace.manifest["event_taxonomy"] == [
        "spawn",
        "assign",
        "delegate",
        "msg",
        "tool",
        "obs",
        "critique",
        "revise",
        "aggregate",
        "stop",
    ]
    assert trace.validate_graph() is None


def test_dynamic_runner_records_stop_at_behavior_turn_boundary():
    runner = MultiAgentRunner()
    task = Task(
        task_id="dyn-stop-boundary",
        dataset="humaneval",
        prompt="def add(a, b):",
        tests="assert add(1, 2) == 3",
    )
    trace = runner.run(task, RunnerConfig(seed=6, max_turns=12, planner_mode="dynamic"))
    assert trace.events[-1].type == "stop"
    assert trace.events[-1].parents == [next(event for event in trace.events if event.type == "aggregate").event_id]
    assert trace.validate_graph() is None


def test_dynamic_runner_uses_memory_budget_routing_and_early_stop():
    runner = MultiAgentRunner()
    task = Task(
        task_id="dyn-2",
        dataset="gsm8k",
        prompt="What is 40 + 2?",
        reference="42",
    )
    trace = runner.run(
        task,
        RunnerConfig(
            seed=0,
            max_turns=12,
            planner_mode="dynamic",
            max_retries=2,
            max_cost=0.001,
            early_stop_threshold=0.0,
        ),
    )
    assert trace.manifest["planner_state"]["planner_mode"] == "dynamic"
    assert trace.manifest["planner_state"]["memory_size"] > 0
    assert trace.manifest["planner_state"]["budget_remaining"] <= 0.001
    assert any(event.metadata.get("memory_keys") for event in trace.events)
    assert any(event.metadata.get("budget_remaining") is not None for event in trace.events)
    assert any(event.metadata.get("early_stop_reason") for event in trace.events if event.type == "stop")
    tool = next(event for event in trace.events if event.type == "tool")
    obs = next(event for event in trace.events if event.type == "obs")
    assert obs.parents == [tool.event_id]
    assert obs.metadata["observed_tool_event_id"] == tool.event_id
    assert "tool_result" in obs.content


def test_credit_baselines_and_metrics_are_computed_against_teacher_labels():
    trace = make_trace(score=1.0)
    baselines = baseline_scores_for_trace(trace, seed=0)
    assert {
        "random",
        "length",
        "cost",
        "uniform_outcome",
        "agent_role",
        "message_only",
        "agent_deletion",
        "shapley_proxy",
    }.issubset(baselines)
    assert set(baselines["length"]) == {event.event_id for event in trace.events}
    assert baselines["message_only"]["e2"] > 0
    assert baselines["message_only"]["e3"] == 0
    assert baselines["agent_deletion"]["e2"] == baselines["agent_deletion"]["e1"] or baselines["agent_deletion"]["e2"] > 0
    assert baselines["shapley_proxy"]["e4"] > baselines["shapley_proxy"]["e1"]
    teacher = {"e1": 0.0, "e2": 0.8, "e3": -0.2, "e4": 0.1, "e5": 0.0}
    metrics = compare_baseline_to_teacher(baselines["length"], teacher, top_k=2)
    assert "spearman" in metrics
    assert "sign_accuracy" in metrics
    assert "top_k_overlap" in metrics


def test_judge_committee_and_anchor_loading(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "anchors.jsonl"
        path.write_text(
            '{"task_id":"a","raw_score":0.2,"target":0.0,"dispersion":0.01}\n'
            '{"task_id":"b","raw_score":0.8,"target":1.0,"dispersion":0.02}\n',
            encoding="utf-8",
        )
        anchors = load_anchor_scores(path)
        assert anchors.raw_scores == [0.2, 0.8]
        committee = JudgeCommittee([DeterministicJudge("strict"), DeterministicJudge("lenient")], samples_per_judge=2)
        scores = committee.score("Task", "This answer cites evidence because tests pass.")
        assert len(scores) == 4
        assert all(0.0 <= score <= 1.0 for score in scores)


def test_cgo_default_committee_matches_method_and_keeps_dry_run_fallback():
    specs = default_method_judge_specs()
    assert [spec.method_model for spec in specs] == ["GLM-5.1", "gpt-4o"]
    assert specs[0].runtime_model == "GLM-5.1"
    assert len({spec.family for spec in specs}) == 2
    committee, manifest = build_judge_committee(api_judges=False, samples_per_judge=2)
    assert manifest["method_default_models"] == ["GLM-5.1", "gpt-4o"]
    assert manifest["runtime_models"][0] == "GLM-5.1"
    assert manifest["judge_mode"] == "deterministic"
    assert manifest["api_fallback_reason"] == "api_judges_disabled"
    assert len(committee.score("Task", "Answer with evidence because it is verified.")) == 4


def test_openqa_anchor_template_and_calibration_report():
    trace = make_trace(score=1.0)
    trace.dataset = "research_synthesis_qa"
    trace.final_answer = "Evidence shows this because tests verify it."
    committee = JudgeCommittee([DeterministicJudge("strict"), DeterministicJudge("lenient")], samples_per_judge=2)
    rows = build_anchor_template([trace], committee)
    assert rows[0]["trace_id"] == trace.trace_id
    assert rows[0]["target"] is None
    assert "raw_score" in rows[0]
    labeled = [
        {"raw_score": 0.2, "target": 0.0, "dispersion": 0.01},
        {"raw_score": 0.5, "target": 0.5, "dispersion": 0.03},
        {"raw_score": 0.8, "target": 1.0, "dispersion": 0.06},
    ]
    report = calibration_report_from_anchors(labeled, alpha=0.2)
    assert report["num_anchors"] == 3
    assert report["judge_human_corr"] > 0.9
    assert report["expected_calibration_error"] >= 0.0
    assert "conformal_threshold" in report
    assert report["coverage_target"] == 0.8
    assert "bias_epsilon" in report
    assert "delta_bias_bound_2epsilon" in report


def test_cgo_metrics_report_calibrated_transfer_and_retention():
    oracle_rows = [
        {"trace_id": "fac", "calibrated_score": 0.8, "dispersion": 0.01, "abstain": False},
        {"trace_id": "cf", "calibrated_score": 0.4, "dispersion": 0.02, "abstain": False},
        {"trace_id": "abs", "calibrated_score": 0.5, "dispersion": 0.8, "abstain": True},
    ]
    anchor_report = {
        "num_anchors": 4,
        "bias_epsilon": 0.05,
        "conformal_threshold": 0.2,
        "expected_calibration_error": 0.1,
        "judge_human_corr": 0.7,
    }
    metrics = compute_oracle_metrics(
        rows=oracle_rows,
        anchor_report=anchor_report,
        alpha=0.1,
        factual_counterfactual_pairs=[("fac", "cf"), ("fac", "abs")],
    )
    assert metrics["coverage_target"] == 0.9
    assert metrics["delta_bias_bound_2epsilon"] == 0.1
    assert metrics["factual_counterfactual_both_retained_rate"] == 0.5
    assert metrics["retained_set_score_reliability"] == 1.0 - anchor_report["expected_calibration_error"]


def test_judge_only_reranking_scores_and_selects_best_trace():
    weak = make_trace(score=0.0)
    weak.trace_id = "weak"
    weak.final_answer = "Maybe 42."
    strong = make_trace(score=1.0)
    strong.trace_id = "strong"
    strong.final_answer = "The answer is 42 because the evidence verifies the arithmetic tests."
    committee = JudgeCommittee([DeterministicJudge("strict"), DeterministicJudge("lenient")], samples_per_judge=2)
    scored = score_traces_with_committee([weak, strong], committee)
    assert scored["strong"]["raw_mean"] > scored["weak"]["raw_mean"]
    summary = summarize_judge_reranking([weak, strong], scored, mode="deterministic")
    assert summary["selected_trace_id"] == "strong"
    assert summary["selected_success"] is True
    assert summary["mode"] == "deterministic"


def test_control_summary_includes_event_telemetry():
    trace = make_trace(score=1.0)
    trace.events[0].metadata["telemetry"] = {"api_calls": 1, "api_request_attempts": 2, "input_tokens": 8, "output_tokens": 5}
    summary = summarize_controlled_trace(trace)
    assert summary["controlled_api_calls"] == 1
    assert summary["controlled_api_request_attempts"] == 2
    assert summary["controlled_input_tokens"] == 8
    assert summary["controlled_output_tokens"] == 5


def test_paper_table_builder_summarizes_required_sections(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        trace.manifest["telemetry"] = {"api_calls": 3, "api_request_attempts": 4, "input_tokens": 20, "output_tokens": 10}
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "baseline_metrics.json").write_text(
            json.dumps({"summary": {"random": {"spearman": 0.1, "sign_accuracy": 0.5, "top_k_overlap": 0.0, "n": 5.0}}}),
            encoding="utf-8",
        )
        (run_dir / "student_metrics.json").write_text(
            json.dumps({"model": "relational", "mae": 0.2, "rmse": 0.3, "corr": 0.4, "sign_accuracy": 0.8}),
            encoding="utf-8",
        )
        (run_dir / "student_checkpoint.json").write_text(
            json.dumps({"loss": {"regression": 0.1, "ranking": 0.2, "ranking_pairs": 3, "masked_events": 1}}),
            encoding="utf-8",
        )
        (run_dir / "control_summary.json").write_text(
            json.dumps(
                {
                    "score_source": "reward_labels",
                    "rl_samples": 5,
                    "advantage_std": 1.0,
                    "stop_signal_count": 1,
                    "pruned_events": 4,
                    "controlled_success": True,
                    "controlled_tokens": 42,
                    "controlled_cost_usd": 0.01,
                    "controlled_latency_ms": 12.5,
                    "controlled_tool_calls": 2,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "ppo_smoke_summary.json").write_text(
            json.dumps({"samples": 5, "objective": 0.12}),
            encoding="utf-8",
        )
        (run_dir / "reward_summary.json").write_text(
            json.dumps({"reward_labels": 5, "credited_events": 2, "stop_signal_count": 1}),
            encoding="utf-8",
        )
        (run_dir / "judge_reranking.json").write_text(
            json.dumps({"selected_success": True, "selected_score": 0.7, "num_traces": 1}),
            encoding="utf-8",
        )
        (run_dir / "ablation_metrics.json").write_text(
            json.dumps({"rows": [{"ablation": "full_carve", "success_rate": 1.0, "credit_labels": 4, "mean_events": 5.0}]}),
            encoding="utf-8",
        )
        tables = build_paper_tables([run_dir])
        assert "table1_dataset_trace_stats" in tables
        assert "table2_credit_quality" in tables
        assert "table3_control_results" in tables
        assert "table4_student_distillation" in tables
        assert any(row["method"] == "carve_control" and row["score_source"] == "reward_labels" for row in tables["table3_control_results"])
        carve_row = next(row for row in tables["table3_control_results"] if row["method"] == "carve_control")
        assert carve_row["success_rate"] == 1.0
        assert carve_row["mean_tokens"] == 42
        assert carve_row["mean_cost_usd"] == 0.01
        assert carve_row["latency_ms"] == 12.5
        assert carve_row["tool_calls"] == 2
        raw_row = next(row for row in tables["table3_control_results"] if row["method"] == "raw_multi_agent")
        assert raw_row["api_calls"] == 3
        assert raw_row["input_tokens"] == 20
        assert raw_row["output_tokens"] == 10
        assert carve_row["ppo_smoke_samples"] == 5
        assert carve_row["ppo_smoke_objective"] == 0.12
        assert tables["table4_student_distillation"][0]["ranking_pairs"] == 3
        assert tables["table4_student_distillation"][0]["masked_events"] == 1
        assert tables["table4_student_distillation"][0]["regression_loss"] == 0.1
        assert tables["table5_ablations"][0]["ablation"] == "full_carve"
        markdown = render_markdown_tables(tables)
        assert "Table 1" in markdown
        assert "random" in markdown


def test_validate_run_dir_checks_acceptance_gates(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "validation_smoke",
                    "dataset": "gsm8k",
                    "seed": 0,
                    "model": "deterministic",
                    "operator_config": {"k": 1, "top_m": 3},
                    "stages": ["trace_collection", "counterfactuals", "rewards"],
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps(
                {
                    "trace_id": trace.trace_id,
                    "event_id": "e2",
                    "delta_mean": 0.2,
                    "abstained": False,
                    "metadata": {
                        "rescaled_delta": 0.2,
                        "replays": [{"prefix_valid": True}],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "reward_labels.jsonl").write_text(
            json.dumps({"event_id": "e2", "total_reward": 0.3}) + "\n",
            encoding="utf-8",
        )
        report = validate_run_dir(run_dir)
        assert report["passed"] is True
        assert report["parent_resolved_rate"] == 1.0
        assert report["credit_labels"] == 1
        assert report["reward_labels"] == 1



def test_validate_run_dir_allows_sampled_operator_coverage():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "run_id": "sampled_coverage",
                    "dataset": "gsm8k",
                    "seed": 0,
                    "model": "api",
                    "operator_config": {
                        "k": 1,
                        "top_m": 5,
                        "operators_per_event": 2,
                        "stop_counterfactual": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps({"trace_id": trace.trace_id, "event_id": "e2", "delta_mean": 0.0, "abstained": False}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "prs_summary.json").write_text(
            json.dumps(
                {
                    "prs": {
                        "operator_coverage": {
                            "complete_for_compatible_events": False,
                            "missing_compatible_operators": ["drop_call"],
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        report = validate_run_dir(run_dir)
        assert report["passed"] is True
        assert report["operator_coverage_policy"] == "sampled"
        assert report["operator_coverage_complete"] is False
        assert report["operator_coverage_missing"] == ["drop_call"]


def test_paper_campaign_plan_covers_required_datasets_and_gates():
    full = build_campaign_plan("paper_full", scale="full", seed=0)
    datasets = {row["dataset"] for row in full["runs"]}
    assert {"humaneval", "mbpp", "gsm8k", "swebench_lite", "research_synthesis_qa"}.issubset(datasets)
    by_dataset = {row["dataset"]: row for row in full["runs"]}
    assert by_dataset["humaneval"]["limit"] == 164
    assert by_dataset["mbpp"]["limit"] == 500
    assert by_dataset["gsm8k"]["limit"] == 500
    assert by_dataset["swebench_lite"]["limit"] == 300
    assert by_dataset["research_synthesis_qa"]["limit"] == 500
    assert "validate_run" in full["required_stages"]
    assert full["acceptance_gates"]["parent_resolved_rate"] == 0.95
    smoke = build_campaign_plan("paper_smoke", scale="smoke", seed=0)
    assert all(row["limit"] <= 2 for row in smoke["runs"])


def test_paper_campaign_executor_builds_commands_and_summary(tmp_path=None):
    plan = build_campaign_plan("paper_smoke", scale="smoke", seed=0)
    command = build_run_command(plan["runs"][0])
    assert "experiments/run_pilot.py" in command
    assert "--dataset" in command
    assert "--ppo-smoke" in command
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / plan["runs"][0]["run_id"]
        run_dir.mkdir()
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "validation_report.json").write_text(json.dumps({"passed": True, "issues": []}), encoding="utf-8")
        (run_dir / "student_metrics.json").write_text(json.dumps({"model": "relational", "mae": 0.1}), encoding="utf-8")
        summary = summarize_campaign(plan, root, root / "tables")
        assert summary["runs_total"] == 5
        assert summary["runs_found"] == 1
        assert summary["validations_passed"] == 1
        assert "table1_dataset_trace_stats" in summary["paper_tables"]


def test_paper_campaign_command_can_skip_student():
    plan = build_campaign_plan("paper_smoke", scale="smoke", seed=0)
    spec = dict(plan["runs"][0])
    spec["skip_student"] = True
    command = build_run_command(spec)
    assert "--skip-student" in command


def test_control_uses_teacher_reward_source_when_student_skipped(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        labels = build_reward_labels(
            trace,
            {"e2": {"delta": 1.0, "score_source": "verifier", "operator_name": "nullify", "operator_family": "msg"}},
            {},
            RewardWeights(),
        )
        (run_dir / "reward_labels.jsonl").write_text(
            "\n".join(json.dumps(label.__dict__) for label in labels) + "\n",
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(json.dumps({"skip_student": True}), encoding="utf-8")
        event_scores, _ = load_control_event_scores(run_dir)
        assert score_trace_for_control(trace, event_scores) != 0.0
        assert control_reward_source(run_dir) == "teacher_rewards"


def test_run_pilot_passes_teacher_reward_source_when_student_skipped():
    import sys
    import experiments.run_pilot as run_pilot

    commands = []
    old_argv = list(sys.argv)
    old_run = run_pilot.run

    def fake_run(cmd):
        commands.append(cmd)
        run_dir = Path("artifacts/runs/unit_skip_student")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "unit_skip_student"}), encoding="utf-8")

    try:
        sys.argv = [
            "run_pilot.py",
            "--dataset",
            "humaneval",
            "--limit",
            "1",
            "--run-id",
            "unit_skip_student",
            "--skip-student",
        ]
        run_pilot.run = fake_run
        run_pilot.main()
    finally:
        sys.argv = old_argv
        run_pilot.run = old_run
    control_cmds = [cmd for cmd in commands if "experiments/run_control.py" in cmd]
    assert control_cmds
    assert "--reward-source" in control_cmds[0]
    assert "teacher_rewards" in control_cmds[0]


def test_humaneval_nostudent_batch_accepts_explicit_offsets():
    assert list(iter_offsets(start=0, end=10, offsets_csv="10,20,31")) == [10, 20, 31]
    assert list(iter_offsets(start=3, end=6, offsets_csv=None)) == [3, 4, 5]


def test_humaneval_nostudent_batch_can_enable_api_downstream_replay():
    import argparse

    args = argparse.Namespace(
        run_prefix="unit_humaneval_",
        max_retries=1,
        max_cost=10.0,
        early_stop_threshold=0.0,
        prompt_version="code_v2",
        k=3,
        replay_mode="behavior",
        top_m=5,
        operators_per_event=2,
        api_replay=True,
    )
    command = build_humaneval_batch_command(0, args)
    assert "--api" in command
    assert "--api-replay" in command
    assert "--prompt-version" in command
    assert "code_v2" in command


def test_humaneval_nostudent_batch_stops_after_consecutive_failures():
    assert should_stop_after_failure(consecutive_failures=2, max_consecutive_failures=3) is False
    assert should_stop_after_failure(consecutive_failures=3, max_consecutive_failures=3) is True
    assert should_stop_after_failure(consecutive_failures=10, max_consecutive_failures=0) is False


def test_paper_campaign_executor_skips_validated_runs_and_records_failures(tmp_path=None):
    plan = build_campaign_plan("paper_smoke", scale="smoke", seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        completed_dir = root / plan["runs"][0]["run_id"]
        completed_dir.mkdir()
        (completed_dir / "validation_report.json").write_text(json.dumps({"passed": True, "issues": []}), encoding="utf-8")
        assert campaign_run_status(plan["runs"][0], root) == "passed"
        assert campaign_run_status(plan["runs"][1], root) == "missing"

        calls = []

        def fake_runner(cmd):
            calls.append(cmd)
            if "--dataset" in cmd and cmd[cmd.index("--dataset") + 1] == plan["runs"][2]["dataset"]:
                raise RuntimeError("intentional failure")

        result = execute_campaign_runs(
            plan,
            runs_root=root,
            skip_existing=True,
            fail_fast=False,
            command_runner=fake_runner,
        )
        assert result["skipped"] == [plan["runs"][0]["run_id"]]
        assert plan["runs"][2]["run_id"] in result["failed"]
        assert len(calls) == 4


def test_ablation_specs_cover_paper_required_variants():
    specs = default_ablation_specs()
    names = {spec.name for spec in specs}
    assert {
        "full_carve",
        "message_only",
        "no_stop_counterfactual",
        "no_crn",
        "random_event_selection",
        "delta_only_reward",
    }.issubset(names)
    full = next(spec for spec in specs if spec.name == "full_carve")
    assert full.all_compatible is True
    assert full.top_m >= 3


def test_ablation_summary_reads_run_outputs(tmp_path=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "ablation_full_carve"
        run_dir.mkdir()
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "credit_labels.jsonl").write_text('{"trace_id":"tr1","event_id":"e2","delta_mean":0.5}\n', encoding="utf-8")
        (run_dir / "student_metrics.json").write_text(json.dumps({"mae": 0.2, "model": "relational"}), encoding="utf-8")
        rows = summarize_ablation_runs({"full_carve": run_dir})
        assert rows[0]["ablation"] == "full_carve"
        assert rows[0]["success_rate"] == 1.0
        assert rows[0]["credit_labels"] == 1


def test_api_judge_parses_json_and_text_scores():
    prompts = []

    class FakeClient:
        def complete(self, role, prompt, seed):
            prompts.append((role, prompt, seed))
            return '{"score": 0.73, "rationale": "grounded"}'

    judge = APIJudge("api_judge", FakeClient())
    assert judge.score("Task prompt", "Answer text", 2) == 0.73
    assert prompts[0][0] == "api_judge"
    assert "Task prompt" in prompts[0][1]
    assert "Answer text" in prompts[0][1]

    class TextClient:
        def complete(self, role, prompt, seed):
            return "I would assign 4 out of 5."

    assert APIJudge("text_judge", TextClient()).score("Task", "Answer", 0) == 0.8


def test_openai_compatible_client_uses_configured_base_url_and_key():
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

    def opener(request, timeout):
        calls.append((request.full_url, dict(request.header_items()), request.data, timeout))
        return FakeResponse()

    client = OpenAICompatibleClient(
        APIClientConfig(api_key="secret", base_urls=["https://api.example.org"], model="GLM-5.1"),
        opener=opener,
    )
    assert client.complete("solver", "hello", seed=3) == "ok"
    assert calls[0][0] == "https://api.example.org/v1/chat/completions"
    assert calls[0][1]["Authorization"] == "Bearer secret"
    assert calls[0][1]["User-agent"] == "CARVE/1.0"




def test_openai_compatible_client_uses_configured_chat_completions_path():
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}"

    def opener(request, timeout):
        calls.append(request.full_url)
        return FakeResponse()

    client = OpenAICompatibleClient(
        APIClientConfig(
            api_key="secret",
            base_urls=["https://api.openai.com/v1"],
            model="ep-example",
            chat_completions_path="/api/v3/chat/completions",
        ),
        opener=opener,
    )
    assert client.complete("solver", "hello", seed=3) == "ok"
    assert calls[0] == "https://api.openai.com/v1/api/v3/chat/completions"

def test_openai_compatible_client_accepts_reasoning_content_fallback():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"","reasoning_content":"reasoned answer"}}]}'

    client = OpenAICompatibleClient(
        APIClientConfig(api_key="secret", base_urls=["https://api.example.org"], model="GLM-5.2"),
        opener=lambda _request, timeout: FakeResponse(),
    )
    assert client.complete("solver", "hello", seed=3) == "reasoned answer"


def test_openai_compatible_client_records_provider_usage_telemetry():
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":11,"completion_tokens":7}}'

    client = OpenAICompatibleClient(
        APIClientConfig(api_key="secret", base_urls=["https://api.example.org"], model="GLM-5.1"),
        opener=lambda _request, timeout: FakeResponse(),
    )
    assert client.complete("solver", "hello", seed=3) == "ok"
    telemetry = client.last_completion_telemetry()
    assert telemetry["api_calls"] == 1
    assert telemetry["api_request_attempts"] == 1
    assert telemetry["input_tokens"] == 11
    assert telemetry["output_tokens"] == 7
    assert telemetry["token_source"] == "provider_usage"


def test_runner_aggregates_completion_telemetry_into_trace():
    class TelemetryClient:
        def complete(self, role, prompt, seed):
            return "answer"

        def last_completion_telemetry(self):
            return {"api_calls": 1, "api_request_attempts": 2, "input_tokens": 11, "output_tokens": 7, "token_source": "provider_usage", "wall_clock_latency_ms": 5.0}

    runner = MultiAgentRunner(client=TelemetryClient())
    task = Task(task_id="telemetry-task", dataset="gsm8k", prompt="What is 2+2?", reference="4")
    trace = runner.run(task, RunnerConfig(model="api-model", max_turns=2, token_cost=0.01))
    telemetry = trace.manifest["telemetry"]
    assert telemetry["api_calls"] == 2
    assert telemetry["api_request_attempts"] == 4
    assert telemetry["input_tokens"] == 22
    assert telemetry["output_tokens"] == 14
    assert telemetry["tool_calls"] == 0
    assert telemetry["wall_clock_latency_ms"] == 10.0
    assert telemetry["estimated_cost_usd"] == 0.36
    assert trace.events[0].tokens_in == 11
    assert trace.events[0].cost_usd == 0.18


def test_real_dataset_jsonl_loaders_parse_common_formats():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        humaneval_path = tmp_path / "humaneval.jsonl"
        humaneval_path.write_text(
            '{"task_id":"HumanEval/0","prompt":"def add(a,b):","canonical_solution":"return a+b","test":"assert add(1,2)==3"}\n',
            encoding="utf-8",
        )
        mbpp_path = tmp_path / "mbpp.jsonl"
        mbpp_path.write_text(
            '{"task_id":1,"text":"square n","code":"def square(n): return n*n","test_list":["assert square(3)==9"]}\n',
            encoding="utf-8",
        )
        gsm_path = tmp_path / "gsm8k.jsonl"
        gsm_path.write_text('{"question":"1+1?","answer":"#### 2"}\n', encoding="utf-8")
        swe_path = tmp_path / "swebench.jsonl"
        swe_path.write_text(
            '{"instance_id":"repo__pkg-1","problem_statement":"fix bug","repo":"repo/pkg","base_commit":"abc",'
            '"test_command":"python -m unittest","repo_path":"/tmp/repo"}\n',
            encoding="utf-8",
        )
        assert load_humaneval_jsonl(humaneval_path, 1)[0].tests == "assert add(1,2)==3"
        assert "assert square" in load_mbpp_jsonl(mbpp_path, 1)[0].tests
        assert load_gsm8k_jsonl(gsm_path, 1)[0].reference == "2"
        swe_task = load_swebench_jsonl(swe_path, 1)[0]
        assert swe_task.task_id == "repo__pkg-1"
        assert swe_task.tests == "python -m unittest"
        assert swe_task.metadata["repo_path"] == "/tmp/repo"


def test_humaneval_loader_offset_selects_single_task():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "humaneval.jsonl"
        path.write_text(
            "\n".join(
                [
                    '{"task_id":"HumanEval/0","prompt":"def a():","test":"assert a() == 0"}',
                    '{"task_id":"HumanEval/1","prompt":"def b():","test":"assert b() == 1"}',
                    '{"task_id":"HumanEval/2","prompt":"def c():","test":"assert c() == 2"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = load_humaneval_jsonl(path, limit=1, offset=1)
        assert [task.task_id for task in tasks] == ["HumanEval/1"]

def test_gsm8k_loader_offset_selects_single_task():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gsm8k.jsonl"
        path.write_text(
            '{"task_id":"gsm8k_0","question":"q0","answer":"#### 0"}\n{"task_id":"gsm8k_1","question":"q1","answer":"#### 1"}\n{"task_id":"gsm8k_2","question":"q2","answer":"#### 2"}\n',
            encoding="utf-8",
        )
        tasks = load_gsm8k_jsonl(path, limit=1, offset=1)
        assert [task.task_id for task in tasks] == ["gsm8k_1"]


def test_mbpp_loader_offset_selects_requested_slice():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mbpp.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"task_id": f"m{index}", "text": f"task {index}", "test_list": [f"assert f() == {index}"]})
                for index in range(5)
            )
            + "\n",
            encoding="utf-8",
        )
        tasks = load_mbpp_jsonl(path, limit=2, offset=2)
        assert [task.task_id for task in tasks] == ["m2", "m3"]



def test_code_task_final_answer_prefers_verified_candidate_over_markdown_aggregate():
    task = Task(
        "HumanEval/x",
        "humaneval",
        "def answer():",
        tests="def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    )
    events = [
        Event(
            event_id="e1",
            trace_id="tr",
            task_id=task.task_id,
            t=0,
            type="msg",
            agent_role="solver_a",
            agent_id="solver_a-1",
            content="```python\ndef answer():\n    return 42\n```",
            parents=[],
        ),
        Event(
            event_id="e2",
            trace_id="tr",
            task_id=task.task_id,
            t=1,
            type="aggregate",
            agent_role="aggregator",
            agent_id="aggregator-1",
            content="The verified answer is above, and all tests pass.",
            parents=["e1"],
        ),
    ]
    assert MultiAgentRunner()._final_answer(events, task).startswith("```python")


def test_counterfactual_terminal_readout_does_not_recover_verified_history():
    task = Task(
        "HumanEval/x",
        "humaneval",
        "def answer():",
        tests="def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    )
    events = [
        Event(
            event_id="e1",
            trace_id="tr",
            task_id=task.task_id,
            t=0,
            type="msg",
            agent_role="solver_a",
            agent_id="solver_a-1",
            content="```python\ndef answer():\n    return 42\n```",
            parents=[],
        ),
        Event(
            event_id="e2",
            trace_id="tr",
            task_id=task.task_id,
            t=1,
            type="aggregate",
            agent_role="aggregator",
            agent_id="aggregator-1",
            content="```python\ndef answer():\n    return 0\n```",
            parents=["e1"],
        ),
    ]
    assert MultiAgentRunner()._final_answer(events, task, final_answer_policy="terminal_readout") == events[-1].content


def test_counterfactual_replay_prompt_marks_restored_state_not_resolve_from_scratch():
    from experiments.run_counterfactuals import build_counterfactual_replay_prompt

    task = Task("HumanEval/x", "humaneval", "def answer():", tests="assert True")
    event = Event("e1", "tr", "HumanEval/x", 0, "msg", "solver", "solver-1", "candidate")
    prompt = build_counterfactual_replay_prompt(task, [event])
    assert "Restored orchestration state" in prompt
    assert "Do not restart from scratch" in prompt
    assert "Terminal readout policy" in prompt


def test_trace_collection_scores_humaneval_with_code_verifier():
    from experiments.run_trace_collection import score_trace_for_task

    task = Task(
        "HumanEval/x",
        "humaneval",
        "def answer():",
        tests="def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    )
    trace = Trace(
        trace_id="tr",
        task_id=task.task_id,
        dataset=task.dataset,
        split="smoke",
        events=[],
        final_answer="def answer():\n    return 42\n",
    )
    score = score_trace_for_task(trace, task)
    assert score.score == 1.0
    assert score.success is True


def test_code_v2_prompt_set_preserves_default_and_adds_evidence_fields():
    default_roles = get_role_specs("default")
    code_v2_roles = get_role_specs("code_v2")
    assert "Solve independently as solver A" in default_roles["solver_a"].prompt_template
    assert "Candidate: solver_a" in code_v2_roles["solver_a"].prompt_template
    assert "Selected_candidate:" in code_v2_roles["aggregator"].prompt_template
    assert "Decision: STOP or CONTINUE" in code_v2_roles["stopper"].prompt_template


def test_mbpp_v1_prompt_set_has_public_test_contract():
    roles = get_role_specs("mbpp_v1")
    assert "public tests" in roles["planner"].prompt_template.lower()
    assert "exact function name" in roles["solver_a"].prompt_template.lower()
    assert "Candidate: solver_a" in roles["solver_a"].prompt_template
    assert "passed the programmatic verifier" in roles["aggregator"].prompt_template
    assert "Decision: STOP or CONTINUE" in roles["stopper"].prompt_template


def test_mbpp_runner_uses_dedicated_workflow_and_exposes_public_tests():
    prompts = []

    class CapturingClient:
        def complete(self, role, prompt, seed):
            prompts.append((role, prompt, seed))
            return "Next_roles: solver_a, solver_b\nRationale: collect independent candidates"

    task = Task(
        "2",
        "mbpp",
        "Write a function to find the shared elements from two tuples.",
        tests="assert set(similar_elements((1, 2), (2, 3))) == {2}",
    )
    runner = MultiAgentRunner(client=CapturingClient(), roles=get_role_specs("mbpp_v1"))
    trace = runner.run(task, RunnerConfig(seed=0, max_turns=1, prompt_version="mbpp_v1"))

    assert trace.manifest["workflow"] == "mbpp_code"
    assert trace.events[0].metadata["mechanism_name"] == "mbpp_code.assign.planner"
    assert "Public tests:" in prompts[0][1]
    assert "similar_elements" in prompts[0][1]
    assert "def similar_elements" not in prompts[0][1]


def test_mbpp_v1_prompt_is_exposed_by_experiment_entry_points():
    for module in ("experiments.run_trace_collection", "experiments.run_pilot"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "mbpp_v1" in result.stdout


def test_runner_records_prompt_version_in_manifest_and_events():
    task = Task(
        "HumanEval/x",
        "humaneval",
        "def answer():",
        tests="def check(candidate):\n    assert candidate() == 42\ncheck(answer)",
    )
    runner = MultiAgentRunner(roles=get_role_specs("code_v2"))
    trace = runner.run(task, RunnerConfig(seed=0, planner_mode="dynamic", prompt_version="code_v2"))
    assert trace.manifest["prompt_version"] == "code_v2"
    assert all(event.metadata.get("prompt_version") == "code_v2" for event in trace.events)


def test_conservation_reports_near_zero_event_signal_as_unrescalable():
    labels = [
        CreditLabel("tr", "e1", "msg", "nullify", 1.0, [0.9], 0.1, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e2", "tool", "drop_call", 1.0, [0.8], 0.2, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e3", "tool", "drop_call", 1.0, [1.3], -0.3, 0.0, 1, False, "verifier"),
    ]
    adjusted = apply_family_baseline_and_rescale(labels, outcome=1.0, empty_baseline=0.0)
    assert all(abs(label.metadata["rescaled_delta"]) < 1e-12 for label in adjusted)
    assert all(label.metadata["conservation_status"] == "near_zero_unrescalable" for label in adjusted)
    assert all(label.metadata["conservation_satisfied"] is False for label in adjusted)


def test_conservation_excludes_abstained_labels_from_event_aggregation():
    labels = [
        CreditLabel("tr", "e1", "msg", "nullify", 1.0, [0.5], 0.5, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e2", "msg", "delete", 1.0, [0.75], 0.25, 0.0, 1, False, "verifier"),
        CreditLabel("tr", "e3", "msg", "reroute", 1.0, [], 0.0, 0.0, 0, True, "verifier"),
    ]
    adjusted = apply_family_baseline_and_rescale(labels, outcome=1.0, empty_baseline=0.0)
    by_event = {label.event_id: label for label in adjusted}
    assert by_event["e1"].metadata["family_loo_baseline"] == 0.0
    assert by_event["e2"].metadata["family_loo_baseline"] == 0.0
    assert by_event["e1"].metadata["event_label_count"] == 1
    assert by_event["e3"].metadata["conservation_status"] == "abstained_excluded"


def test_effective_credit_falls_back_to_raw_when_conservation_is_unsatisfied():
    from carve.scoring.credit_value import effective_credit

    row = {"delta_mean": 0.75, "abstained": False, "metadata": {"rescaled_delta": 0.0, "conservation_satisfied": False}}
    assert effective_credit(row) == 0.75
    assert effective_credit({**row, "abstained": True}) is None
    assert effective_credit({**row, "operator_family": "stop"}) is None


def test_effective_credit_rejects_explosive_conservation_scale():
    from carve.scoring.credit_value import effective_credit

    row = {
        "delta_mean": 0.5,
        "abstained": False,
        "metadata": {"rescaled_delta": 1.2e16, "conservation_scale": 3.6e16, "conservation_satisfied": True},
    }
    assert effective_credit(row) == 0.5


def test_credit_consumers_share_effective_credit_and_skip_abstentions():
    from experiments.run_rewards import load_credit_components
    from experiments.run_baselines import load_teacher_labels
    from experiments.run_student import load_event_targets

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "credit_labels.jsonl"
        rows = [
            {"trace_id": "tr1", "event_id": "e1", "delta_mean": 0.75, "abstained": False, "score_source": "verifier", "metadata": {"rescaled_delta": 0.0, "conservation_satisfied": False}},
            {"trace_id": "tr1", "event_id": "e2", "delta_mean": 9.0, "abstained": True, "score_source": "verifier", "metadata": {"rescaled_delta": 9.0, "conservation_satisfied": True}},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        assert load_credit_components(path) == {"tr1::e1": {"delta": 0.75, "score_source": "verifier"}}
        assert load_teacher_labels(path) == {"tr1": {"e1": 0.75}}
        assert load_event_targets(path) == {"tr1::e1": 0.75}


def test_control_reverifies_each_gsm8k_trace_after_pruning():
    from experiments.run_control import evaluate_controlled_traces

    events = [
        Event("e1", "gsm-tr", "gsm-task", 0, "msg", "solver", "solver-1", "Final answer: 17", []),
        Event("e2", "gsm-tr", "gsm-task", 1, "aggregate", "aggregator", "aggregator-1", "Final answer: 18", ["e1"]),
        Event("e3", "gsm-tr", "gsm-task", 2, "stop", "stopper", "stopper-1", "Stop", ["e2"]),
    ]
    trace = Trace("gsm-tr", "gsm-task", "gsm8k", "test", events, "Final answer: 18", 1.0, None, True, manifest={"task": {"reference": "18"}})
    controlled, summary = evaluate_controlled_traces([trace], {"gsm-tr::e1": 1.0, "gsm-tr::e2": -1.0, "gsm-tr::e3": 0.0})
    assert controlled[0].final_answer == "Final answer: 18"
    assert controlled[0].success is True
    assert controlled[0].verifier_score == 1.0
    assert summary["controlled_success_rate"] == 1.0
    assert summary["removed_events_total"] == 0
    assert summary["kept_events_total"] == 3


def test_control_aggregates_metrics_over_all_tasks():
    from experiments.run_control import evaluate_controlled_traces

    traces = []
    for index, answer in enumerate(("18", "19")):
        event = Event("e1", f"tr{index}", f"task{index}", 0, "aggregate", "aggregator", "aggregator-1", f"Final answer: {answer}", [], tokens_in=10, tokens_out=5)
        traces.append(Trace(f"tr{index}", f"task{index}", "gsm8k", "test", [event], event.content, 1.0, None, True, manifest={"task": {"reference": answer}}))
    controlled, summary = evaluate_controlled_traces(traces, {"tr0::e1": 1.0, "tr1::e1": 1.0})
    assert len(controlled) == 2
    assert summary["controlled_success_rate"] == 1.0
    assert summary["controlled_mean_tokens"] == 15.0


def test_table3_uses_aggregate_control_metrics():
    from experiments.make_paper_tables import table3_control_results

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "control_summary.json").write_text(json.dumps({"control_mode": "offline_structural_pruning_reverified", "controlled_traces": 20, "controlled_success_rate": 0.85, "controlled_mean_tokens": 123.0, "controlled_mean_cost_usd": 0.02, "controlled_mean_latency_ms": 45.0, "controlled_mean_tool_calls": 0.5, "controlled_mean_api_calls": 0.0, "controlled_mean_input_tokens": 0.0, "controlled_mean_output_tokens": 0.0, "mean_removed_events": 2.5}), encoding="utf-8")
        rows = table3_control_results([run_dir])
        carve_row = next(row for row in rows if row["method"] == "carve_control")
        assert carve_row["success_rate"] == 0.85
        assert carve_row["mean_tokens"] == 123.0
        assert carve_row["removed_events"] == 2.5
        assert carve_row["controlled_traces"] == 20


def test_behavior_continuation_force_continue_generates_post_stop_roles():
    trace = make_gsm8k_continuation_trace()
    intervention = apply_operator(trace, "e3", "force_continue", random.Random(0), operator_set="gsm8k_v1")
    replayed = build_behavior_continuation_policy(use_api=False)(trace, intervention, 7)
    generated = [event for event in replayed.events if event.metadata.get("force_continue_generated")]
    assert generated
    assert generated[-1].type == "stop"
    assert any(event.type == "aggregate" for event in generated)
    assert replayed.total_cost_usd > trace.prefix_before("e3")[-1].cost_usd


def test_behavior_continuation_delay_observation_hides_tool_result():
    task = Task("gsm8k_delay", "gsm8k", "What is 9 * 2?", reference="18")
    events = [
        Event("e1", "delay-tr", task.task_id, 0, "tool", "tester", "tester-1", "TOOL_SECRET_18", []),
        Event("e2", "delay-tr", task.task_id, 1, "obs", "test_observer", "observer-1", "TOOL_SECRET_18", ["e1"]),
        Event("e3", "delay-tr", task.task_id, 2, "aggregate", "aggregator", "aggregator-1", "Final answer: 18", ["e2"]),
        Event("e4", "delay-tr", task.task_id, 3, "stop", "stopper", "stopper-1", "Stop", ["e3"]),
    ]
    trace = Trace("delay-tr", task.task_id, task.dataset, "test", events, events[2].content, 1.0, None, True, manifest={"prompt_version": "gsm8k_stable_v1", "task": {"prompt": task.prompt, "reference": task.reference}})
    intervention = apply_operator(trace, "e1", "delay_observation", random.Random(0), operator_set="gsm8k_v1")
    replayed = build_behavior_continuation_policy(use_api=False)(trace, intervention, 7)
    observation = next(event for event in replayed.events if event.metadata.get("source_event_id") == "e2")
    assert observation.metadata["delayed_observation"] is True
    assert "TOOL_SECRET_18" not in observation.content


def test_operator_limit_keeps_force_stop_in_addition_to_domain_operators():
    from experiments.run_counterfactuals import operator_coverage

    trace = make_gsm8k_continuation_trace()
    jobs = select_counterfactual_jobs(trace, top_m=8, operators_per_event=3, event_selection="type_stratified", operator_set="gsm8k_v1")
    aggregate_ops = {job.operator_name for job in jobs if job.event_type == "aggregate"}
    assert aggregate_ops == {
        "wrong_final_number_aggregate_gsm8k",
        "choose_inconsistent_aggregate_gsm8k",
        "premature_aggregate_gsm8k",
        "force_stop",
    }
    coverage = operator_coverage([trace], jobs, "gsm8k_v1")
    assert "premature_aggregate_gsm8k" in coverage["compatible_operators"]
    assert coverage["missing_compatible_operators"] == []
    assert coverage["complete_for_compatible_events"] is True



def test_tail_event_operator_budget_keeps_coverage_with_bounded_cost():
    trace = make_gsm8k_continuation_trace()
    for index, event_type in enumerate(["assign", "obs", "tool", "revise", "critique"], start=4):
        trace.events.append(Event(
            f"e{index}", trace.trace_id, trace.task_id, index - 1, event_type,
            "agent", f"agent-{index}", "content", parents=["e1"],
        ))
    jobs = select_counterfactual_jobs(
        trace,
        top_m=8,
        operators_per_event=2,
        primary_events=5,
        tail_operators_per_event=1,
        event_selection="type_stratified",
        operator_set="gsm8k_v1",
    )
    by_event = {}
    for job in jobs:
        by_event.setdefault(job.event_id, []).append(job)
    assert len(by_event) == 8
    assert len(jobs) == 18
    counts = [len(rows) for rows in by_event.values()]
    assert sorted(counts) == [1, 1, 1, 3, 3, 3, 3, 3]
    assert any(job.event_type == "stop" and job.operator_name == "force_continue" for job in jobs)


def test_counterfactual_resume_detects_completed_traces_and_rejects_second_writer():
    from experiments.run_counterfactuals import completed_trace_ids, counterfactual_run_lock

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        labels = run_dir / "credit_labels.jsonl"
        labels.write_text(json.dumps({"trace_id": "tr1", "event_id": "e1"}) + "\n", encoding="utf-8")
        assert completed_trace_ids(labels) == {"tr1"}
        with counterfactual_run_lock(run_dir):
            try:
                with counterfactual_run_lock(run_dir):
                    assert False, "second writer should not enter"
            except RuntimeError as exc:
                assert "already active" in str(exc)
        assert not (run_dir / ".counterfactuals.lock").exists()


def test_conservation_summary_excludes_incomplete_trace_from_main_metric():
    complete = make_trace(score=1.0)
    incomplete = make_trace(score=1.0)
    incomplete.trace_id = "tr2"
    incomplete.task_id = "task2"
    labels = [
        CreditLabel("tr1", "e1", "msg", "nullify", 1.0, [0.8], 0.2, 0.0, 1, False, "verifier", metadata={"rescaled_delta": 1.0, "conservation_satisfied": True}),
        CreditLabel("tr2", "e1", "msg", "nullify", 1.0, [], 0.0, 0.0, 0, True, "verifier", metadata={"rescaled_delta": 0.0, "conservation_satisfied": False}),
    ]

    summary = conservation_summary([complete, incomplete], labels)

    assert summary["factual_total"] == 1.0
    assert summary["rescaled_delta_sum"] == 1.0
    assert summary["error_after_rescale"] == 0.0
    assert summary["coverage"] == {"eligible_traces": 1, "total_traces": 2, "coverage_rate": 0.5, "excluded_traces": 1}
    assert summary["all_trace_diagnostic"]["factual_total"] == 2.0


def test_trace_collection_appends_completed_traces_incrementally():
    from experiments.run_trace_collection import append_trace_record, load_trace_records

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traces.jsonl"
        first = make_trace(score=1.0)
        append_trace_record(path, first)
        assert len(load_trace_records(path)) == 1

        second = make_trace(score=0.0)
        second.trace_id = "tr2"
        second.task_id = "task2"
        append_trace_record(path, second)
        rows = load_trace_records(path)
        assert [trace.trace_id for trace in rows] == ["tr1", "tr2"]


def test_validation_rejects_explosive_credit_and_missing_controlled_traces():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trace = make_trace(score=1.0)
        trace.manifest["task"] = {"reference": "42"}
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "manifest.json").write_text(json.dumps({"run_id": "x", "dataset": "gsm8k", "seed": 0, "model": "api", "operator_config": {}}), encoding="utf-8")
        (run_dir / "credit_labels.jsonl").write_text(json.dumps({"trace_id": trace.trace_id, "event_id": "e1", "delta_mean": 0.5, "abstained": False, "metadata": {"rescaled_delta": 1.2e16, "conservation_scale": 3.6e16, "conservation_satisfied": True}}) + "\n", encoding="utf-8")
        (run_dir / "control_summary.json").write_text(json.dumps({"controlled_traces": 1}), encoding="utf-8")
        report = validate_run_dir(run_dir)
        assert any("explosive conservation scale" in issue for issue in report["issues"])
        assert any("controlled_traces.jsonl" in issue for issue in report["issues"])


def test_judge_reranking_does_not_rank_different_tasks_against_each_other():
    first = make_trace(score=1.0)
    second = make_trace(score=0.0)
    second.trace_id = "tr2"
    second.task_id = "different-task"
    scored = {first.trace_id: {"raw_mean": 0.9}, second.trace_id: {"raw_mean": 0.1}}
    summary = summarize_judge_reranking([first, second], scored, mode="deterministic")
    assert summary["status"] == "insufficient_candidates"
    assert summary["eligible_tasks"] == 0
    assert summary["selected_trace_ids"] == {}


def test_gsm8k_full_batch_command_uses_full_readiness_configuration():
    import argparse
    from experiments.run_gsm8k_full_batch import build_command

    args = argparse.Namespace(
        run_prefix="GSM8K/full_glm51_task",
        k=3,
        top_m=8,
        operators_per_event=3,
        max_retries=1,
        max_cost=10.0,
        early_stop_threshold=0.0,
    )
    command = build_command(17, args)
    joined = " ".join(command)
    assert "--dataset gsm8k" in joined
    assert "--run-id GSM8K/full_glm51_task0017" in joined
    assert "--prompt-version gsm8k_stable_v1" in joined
    assert "--operator-set gsm8k_v1" in joined
    assert "--k 3" in joined
    assert "--top-m 8" in joined
    assert "--operators-per-event 3" in joined
    assert "--event-selection type_stratified" in joined


    assert "--api-replay" in command
def test_counterfactual_job_checkpoint_loads_and_deduplicates():
    from experiments.run_counterfactuals import (
        counterfactual_job_key,
        load_job_checkpoint,
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "credit_jobs.jsonl"
        row = {
            "job_key": counterfactual_job_key("tr1", "e2", "op_a", "mbpp_v1"),
            "trace_id": "tr1",
            "event_id": "e2",
            "operator_name": "op_a",
            "operator_set": "mbpp_v1",
            "label": {
                "trace_id": "tr1",
                "event_id": "e2",
                "operator_family": "msg",
                "operator_name": "op_a",
                "factual_score": 1.0,
                "counterfactual_scores": [0.0],
                "delta_mean": 1.0,
                "delta_std": 0.0,
                "num_rollouts": 1,
                "abstained": False,
                "score_source": "verifier",
                "metadata": {},
            },
        }
        path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
        loaded = load_job_checkpoint(path)
        assert list(loaded) == [row["job_key"]]
        assert loaded[row["job_key"]]["delta_mean"] == 1.0



def test_retry_abstained_resume_state_reopens_only_failed_traces():
    from experiments.run_counterfactuals import prepare_retry_abstained_resume

    existing_rows = [
        {"trace_id": "tr1", "event_id": "e1", "abstained": False},
        {"trace_id": "tr1", "event_id": "e2", "abstained": True},
        {"trace_id": "tr2", "event_id": "e1", "abstained": False},
    ]
    checkpoints = {
        "tr1::e1::mbpp_v1::ok": {"abstained": False},
        "tr1::e2::mbpp_v1::failed": {"abstained": True},
        "tr2::e1::mbpp_v1::ok": {"abstained": False},
    }

    retained, retryable, completed = prepare_retry_abstained_resume(existing_rows, checkpoints)

    assert len(retained) == 2
    assert set(retryable) == {"tr1::e1::mbpp_v1::ok", "tr2::e1::mbpp_v1::ok"}
    assert completed == {"tr2"}


def test_code_verifier_times_out_infinite_counterfactual_code():
    score = CodeVerifier(timeout_s=0.01).verify("while True: pass", "assert True")

    assert score.score == 0.0
    assert score.success is False
    assert "timed out" in (score.stderr or "").lower()


def test_retry_abstained_resume_uses_failed_checkpoint_trace_ids():
    from experiments.run_counterfactuals import prepare_retry_abstained_resume

    existing_rows = [
        {"trace_id": "tr1", "event_id": "e1", "abstained": False},
        {"trace_id": "tr2", "event_id": "e1", "abstained": False},
    ]
    checkpoints = {
        "tr1::e1::mbpp_v1::ok": {"trace_id": "tr1", "abstained": False},
        "tr1::e2::mbpp_v1::failed": {"trace_id": "tr1", "abstained": True},
        "tr2::e1::mbpp_v1::ok": {"trace_id": "tr2", "abstained": False},
    }

    retained, retryable, completed = prepare_retry_abstained_resume(existing_rows, checkpoints)

    assert len(retained) == 2
    assert set(retryable) == {"tr1::e1::mbpp_v1::ok", "tr2::e1::mbpp_v1::ok"}
    assert completed == {"tr2"}


def test_resume_job_labels_reuses_existing_formal_labels():
    from experiments.run_counterfactuals import resume_job_labels

    existing_rows = [
        {"trace_id": "tr1", "event_id": "e1", "operator_name": "op_a", "abstained": False},
    ]

    labels = resume_job_labels(existing_rows, {}, "mbpp_v1")

    assert labels["tr1::e1::mbpp_v1::op_a"]["trace_id"] == "tr1"
