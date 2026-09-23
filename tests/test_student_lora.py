from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from carve.schemas import Event, Trace
from carve.student_lora.data import build_examples, build_sibling_pairs, counterfactual_text, make_task_split
from carve.student_lora.losses import bradley_terry_loss, masked_huber_loss
from carve.student_lora.metrics import regression_and_ranking_metrics
from carve.student_lora.train import RankingBatch, ScoreBatch, train_step
from carve.student_lora.model import (
    RelationalGraphRewardHead,
    place_model_on_device,
    validate_trainable_parameter_names,
)


def credit(
    trace_id: str,
    event_id: str,
    family: str,
    operator: str,
    delta: float,
    *,
    abstained: bool = False,
    replacement_content: str | None = None,
    deleted: bool = False,
) -> dict:
    replacement = None
    if replacement_content is not None:
        replacement = {"content": replacement_content}
    return {
        "trace_id": trace_id,
        "event_id": event_id,
        "operator_family": family,
        "operator_name": operator,
        "delta_mean": delta,
        "abstained": abstained,
        "metadata": {
            "intervention": {
                "replacement_event": replacement,
                "deleted": deleted,
            }
        },
    }


def test_task_split_is_deterministic_and_disjoint():
    task_ids = [f"HumanEval/{i}" for i in range(20)]
    split = make_task_split(task_ids, seed=0, train_count=16, val_count=2)
    assert len(split.train) == 16
    assert len(split.validation) == 2
    assert len(split.test) == 2
    assert not (set(split.train) & set(split.validation))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))
    assert split == make_task_split(task_ids, seed=0, train_count=16, val_count=2)


def test_sibling_pairs_stay_within_trace_event_and_family():
    rows = [
        credit("tr1", "e2", "msg", "wrong_return", 1.0),
        credit("tr1", "e2", "msg", "empty", 0.0),
        credit("tr1", "e2", "stop", "force_stop", -1.0),
        credit("tr2", "e2", "msg", "empty", -1.0),
    ]
    pairs = build_sibling_pairs(rows)
    assert [(p.preferred.operator_name, p.rejected.operator_name) for p in pairs] == [("wrong_return", "empty")]


def test_tied_and_abstained_siblings_are_excluded():
    rows = [
        credit("tr1", "e2", "msg", "a", 0.0),
        credit("tr1", "e2", "msg", "b", 0.0),
        credit("tr1", "e2", "msg", "c", 1.0, abstained=True),
    ]
    assert build_sibling_pairs(rows) == []


def test_counterfactual_variant_uses_replacement_content_or_deletion_marker():
    replacement = credit("tr1", "e2", "msg", "wrong", -1.0, replacement_content="return None")
    deletion = credit("tr1", "e2", "msg", "remove", -1.0, deleted=True)
    assert counterfactual_text(replacement) == "return None"
    assert counterfactual_text(deletion) == "[DELETED operator=remove]"


def test_masked_huber_ignores_abstained_targets():
    pred = torch.tensor([0.0, 100.0], requires_grad=True)
    target = torch.tensor([1.0, 0.0])
    loss = masked_huber_loss(pred, target, torch.tensor([True, False]), delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.5))


def test_bradley_terry_prefers_larger_score():
    good = bradley_terry_loss(torch.tensor([2.0]), torch.tensor([-1.0]))
    bad = bradley_terry_loss(torch.tensor([-1.0]), torch.tensor([2.0]))
    assert good < bad


def test_empty_ranking_batch_returns_differentiable_zero():
    anchor = torch.tensor([1.0], requires_grad=True)
    loss = bradley_terry_loss(anchor[:0], anchor[:0])
    loss.backward()
    assert anchor.grad is not None


def test_parent_edges_change_child_prediction():
    torch.manual_seed(0)
    head = RelationalGraphRewardHead(input_dim=8, hidden_dim=8, num_event_types=10, num_roles=10, num_relations=7, dropout=0.0)
    nodes = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    no_edges = head(nodes, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.zeros(3, 5), torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.long))
    with_edge = head(nodes, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.zeros(3, 5), torch.tensor([[0], [2]]), torch.tensor([0]))
    assert not torch.isclose(no_edges[2], with_edge[2])


def test_relation_type_changes_child_prediction():
    torch.manual_seed(0)
    head = RelationalGraphRewardHead(input_dim=8, hidden_dim=8, num_event_types=10, num_roles=10, num_relations=7, dropout=0.0)
    nodes = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    edge_index = torch.tensor([[0], [2]])
    common = (nodes, torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2]), torch.zeros(3, 5), edge_index)
    control_score = head(*common, torch.tensor([0]))
    stopping_score = head(*common, torch.tensor([6]))
    assert not torch.isclose(control_score[2], stopping_score[2])


def test_trainable_contract_rejects_unfrozen_base_parameter():
    params = [
        ("base.layers.0.weight", True),
        ("base.layers.0.lora_A.weight", True),
        ("graph_head.out.weight", True),
    ]
    try:
        validate_trainable_parameter_names(params)
    except ValueError as exc:
        assert "unexpected trainable base parameter" in str(exc)
    else:
        raise AssertionError("expected an unfrozen-base validation error")


def test_trainable_contract_accepts_lora_and_heads_only():
    params = [
        ("base.layers.0.weight", False),

        ("base.layers.0.lora_A.weight", True),
        ("graph_head.out.weight", True),
    ]
    validate_trainable_parameter_names(params)

def test_graph_head_accepts_bf16_backbone_states_and_float_features():
    torch.manual_seed(0)
    head = RelationalGraphRewardHead(input_dim=8, hidden_dim=8, num_event_types=10, num_roles=10, num_relations=7, dropout=0.0)
    states = torch.randn(3, 8, dtype=torch.bfloat16)
    scores = head(
        states,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1, 2]),
        torch.zeros(3, 5, dtype=torch.float32),
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0, dtype=torch.long),
    )
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()

def test_place_model_on_device_moves_parameters():
    model = torch.nn.Linear(2, 2)
    placed = place_model_on_device(model, torch.device("cpu"))
    assert placed is model
    assert next(model.parameters()).device.type == "cpu"

class FakeScoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def score_graph(self, **inputs):
        return inputs["scores"] * self.weight


def test_one_training_step_combines_regression_and_ranking():
    model = FakeScoringModel()
    factual = ScoreBatch(
        model_inputs={"scores": torch.tensor([0.0, 1.0])},
        targets=torch.tensor([0.0, 1.0]),
        target_mask=torch.tensor([True, True]),
    )
    ranking = RankingBatch(
        model_inputs={"scores": torch.tensor([2.0, -1.0])},
        preferred_indices=torch.tensor([0]),
        rejected_indices=torch.tensor([1]),
    )
    result = train_step(model, factual, ranking, beta=0.2, huber_delta=1.0)
    assert torch.isfinite(result.total_loss)
    assert torch.isclose(result.total_loss, result.regression_loss + 0.2 * result.ranking_loss)


def test_metrics_are_task_split_scoped_and_finite():
    metrics = regression_and_ranking_metrics([0.0, 1.0], [0.0, 1.0], [1.0], [0.0])
    assert metrics["mae"] == 0.0
    assert metrics["ranking_accuracy"] == 1.0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values() if isinstance(value, float))

def test_task_split_supports_explicit_test_count():
    split = make_task_split([f"HumanEval/{i}" for i in range(20)], seed=0, train_count=16, val_count=2, test_count=2)
    assert len(split.train) == 16
    assert len(split.validation) == 2
    assert len(split.test) == 2
    assert not (set(split.train) & set(split.test))
    assert not (set(split.validation) & set(split.test))

def test_legacy_student_does_not_claim_qwen35_initialization_when_hash_fallback_is_used():
    from experiments.run_student import legacy_student_metadata

    metadata = legacy_student_metadata(backend_model_name="hash", backend_used_fallback=True)
    assert metadata["initialized_from"] is None
    assert metadata["paper_ready"] is False

def test_build_examples_filters_credit_rows_to_task_split():
    def make_trace(trace_id: str, task_id: str) -> Trace:
        event = Event(
            event_id="e1",
            trace_id=trace_id,
            task_id=task_id,
            t=0,
            type="msg",
            agent_role="solver",
            agent_id="solver-1",
            content="candidate",
        )
        return Trace(trace_id, task_id, "humaneval", "test", [event], "candidate", verifier_score=1.0, success=True)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        traces = [make_trace("tr-selected", "task-selected"), make_trace("tr-other", "task-other")]
        (run_dir / "traces.jsonl").write_text("".join(json.dumps(trace.to_dict()) + "\n" for trace in traces), encoding="utf-8")
        rewards = [
            {"trace_id": trace.trace_id, "event_id": "e1", "total_reward": 1.0}
            for trace in traces
        ]
        (run_dir / "reward_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rewards), encoding="utf-8")
        credits = []
        for trace in traces:
            credits.extend(
                [
                    {"trace_id": trace.trace_id, "event_id": "e1", "operator_family": "msg", "operator_name": "a", "delta_mean": 1.0},
                    {"trace_id": trace.trace_id, "event_id": "e1", "operator_family": "msg", "operator_name": "b", "delta_mean": 0.0},
                ]
            )
        (run_dir / "credit_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in credits), encoding="utf-8")
        split = make_task_split(["task-selected", "task-other"], seed=0, train_count=1, val_count=0, test_count=0)
        bundle = build_examples(run_dir, split)
        assert [trace.task_id for trace in bundle.traces] == list(split.train)
        assert len(bundle.sibling_pairs) == 1
        expected_trace_id = {"task-selected": "tr-selected", "task-other": "tr-other"}[split.train[0]]
        assert bundle.sibling_pairs[0].preferred.trace_id == expected_trace_id


def test_build_examples_uses_mean_credit_as_the_labeled_event_target():
    event = Event("e2", "tr1", "task1", 0, "revise", "sql_writer_a", "writer-1", "SELECT 1")
    trace = Trace("tr1", "task1", "spider", "trace", [event], "SELECT 1", verifier_score=1.0, success=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "credit_labels.jsonl").write_text(
            "".join(
                json.dumps({"trace_id": "tr1", "event_id": "e2", "operator_family": "sql_writer_a", "operator_name": name, "delta_mean": value, "abstained": False}) + "\n"
                for name, value in (("one", 0.5), ("two", 1.0))
            ),
            encoding="utf-8",
        )
        bundle = build_examples(run_dir, make_task_split(["task1"], seed=0, train_count=1, val_count=0, test_count=0))

    assert bundle.factual_examples[0].target == 0.75


def test_build_examples_reward_target_sources_change_factual_targets():
    events = [
        Event("e1", "tr1", "task1", 0, "msg", "solver", "solver-1", "candidate one"),
        Event("e2", "tr1", "task1", 1, "msg", "solver", "solver-2", "candidate two"),
    ]
    trace = Trace("tr1", "task1", "gsm8k", "trace", events, "candidate two", verifier_score=1.0, success=True)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n", encoding="utf-8")
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps({"trace_id": "tr1", "event_id": "e1", "operator_family": "msg", "operator_name": "a", "delta_mean": 0.5}) + "\n",
            encoding="utf-8",
        )
        split = make_task_split(["task1"], seed=0, train_count=1, val_count=0, test_count=0)
        credit_bundle = build_examples(run_dir, split, target_source="credit")
        composed_bundle = build_examples(run_dir, split, target_source="reward_composed")
        no_potential_bundle = build_examples(run_dir, split, target_source="reward_no_potential")
        assert credit_bundle.factual_examples[0].target == 0.5
        composed_by_event = {example.event_id: example.target for example in composed_bundle.factual_examples}
        no_potential_by_event = {example.event_id: example.target for example in no_potential_bundle.factual_examples}
        assert composed_by_event["e1"] == no_potential_by_event["e1"] == 0.5
        assert composed_by_event["e2"] != no_potential_by_event["e2"]

def test_student_control_scores_are_trace_scoped():
    from experiments.evaluate_student_lora_control import filter_event_scores_for_traces

    traces = [
        Trace("tr-test", "task-test", "humaneval", "test", [], "", verifier_score=1.0, success=True),
        Trace("tr-train", "task-train", "humaneval", "train", [], "", verifier_score=1.0, success=True),
    ]
    scores = {"tr-test::e1": 0.4, "tr-train::e1": 0.9}
    assert filter_event_scores_for_traces([traces[0]], scores) == {"tr-test::e1": 0.4}


def test_teacher_control_scores_use_credit_labels_not_rewards():
    from experiments.run_control import load_teacher_credit_scores

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "reward_labels.jsonl").write_text(
            json.dumps({"trace_id": "tr1", "event_id": "e1", "total_reward": 99.0}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "credit_labels.jsonl").write_text(
            json.dumps({
                "trace_id": "tr1",
                "event_id": "e1",
                "operator_family": "msg",
                "operator_name": "a",
                "delta_mean": 2.0,
            }) + "\n",
            encoding="utf-8",
        )
        scores, metadata = load_teacher_credit_scores(run_dir)
        assert scores == {"tr1::e1": 2.0}
        assert metadata["score_source"] == "teacher_credit_labels"
