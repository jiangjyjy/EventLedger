import json

from carve.schemas import Event, Trace
from carve.student_lora.data import build_examples, make_task_split
from experiments.run_student_lora import build_config


def test_spider_labeled_event_target_is_mean_credit(tmp_path):
    event = Event("e2", "tr1", "task1", 0, "revise", "sql_writer_a", "writer-1", "SELECT 1")
    trace = Trace("tr1", "task1", "spider", "trace", [event], "SELECT 1", verifier_score=1.0, success=True)
    (tmp_path / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n")
    (tmp_path / "credit_labels.jsonl").write_text(
        "".join(
            json.dumps({"trace_id": "tr1", "event_id": "e2", "operator_family": "sql_writer_a", "operator_name": name, "delta_mean": value, "abstained": False}) + "\n"
            for name, value in (("one", 0.5), ("two", 1.0))
        )
    )

    bundle = build_examples(tmp_path, make_task_split(["task1"], seed=0, train_count=1, val_count=0, test_count=0))

    assert bundle.factual_examples[0].target == 0.75


def test_student_config_preserves_requested_lora_hyperparameters():
    config = build_config(
        source_run="source",
        model_path="model",
        output_dir="output",
        smoke=False,
        lora_r=8,
        lora_alpha=16,
    )

    assert config.lora_r == 8
    assert config.lora_alpha == 16


def test_spider_distillation_excludes_deterministic_verifier_and_resolver_events(tmp_path):
    events = [
        Event("e1", "tr1", "task1", 0, "assign", "planner", "planner-1", "plan", model="glm-5.2"),
        Event("e2", "tr1", "task1", 1, "revise", "sql_writer_a", "writer-a", "SELECT 1", model="glm-5.2"),
        Event("e3", "tr1", "task1", 2, "revise", "sql_writer_b", "writer-b", "SELECT 1", model="glm-5.2"),
        Event("e4", "tr1", "task1", 3, "tool", "public_sql_verifier_a", "verifier-a", "{}"),
        Event("e5", "tr1", "task1", 4, "tool", "public_sql_verifier_b", "verifier-b", "{}"),
        Event("e6", "tr1", "task1", 5, "aggregate", "selector", "selector-1", "candidate_a", model="glm-5.2"),
        Event("e7", "tr1", "task1", 6, "aggregate", "final_resolver", "resolver-1", "SELECT 1"),
        Event("e8", "tr1", "task1", 7, "tool", "hidden_sql_verifier", "hidden-1", "{}"),
    ]
    trace = Trace("tr1", "task1", "spider", "trace", events, "SELECT 1", verifier_score=1.0, success=True)
    (tmp_path / "traces.jsonl").write_text(json.dumps(trace.to_dict()) + "\n")

    bundle = build_examples(tmp_path, make_task_split(["task1"], seed=0, train_count=1, val_count=0, test_count=0))

    assert [example.event_id for example in bundle.factual_examples] == ["e1", "e2", "e3", "e6"]
