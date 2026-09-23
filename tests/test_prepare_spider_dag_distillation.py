import json

from carve.schemas import Event, Trace
from experiments.prepare_spider_dag_distillation import prepare_distillation


def test_prepare_distillation_writes_a_deterministic_task_partition(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    traces = []
    labels = []
    for index in range(10):
        trace_id = f"trace-{index}"
        task_id = f"task-{index}"
        event = Event("e2", trace_id, task_id, 0, "revise", "sql_writer_a", "writer-a", "SELECT 1")
        traces.append(Trace(trace_id, task_id, "spider", "run", [event], "SELECT 1", verifier_score=1.0, success=True))
        labels.append(
            {
                "trace_id": trace_id,
                "event_id": "e2",
                "operator_name": "predicate_drop",
                "delta_mean": 0.5,
                "abstained": False,
                "metadata": {"credit_scheme": "paired_structural_shapley"},
            }
        )
    (bundle / "traces.jsonl").write_text("".join(json.dumps(trace.to_dict()) + "\n" for trace in traces))
    (bundle / "credit_labels.jsonl").write_text("".join(json.dumps(row) + "\n" for row in labels))

    summary = prepare_distillation(bundle, tmp_path / "prepared", seed=17, train_count=7, validation_count=2, test_count=1)

    split = json.loads((tmp_path / "prepared" / "split.json").read_text())
    assert [len(split[name]) for name in ("train", "validation", "test")] == [7, 2, 1]
    assert len(set().union(*[set(split[name]) for name in split])) == 10
    assert summary["traces"] == 10
    assert summary["credit_labels"] == 10
    assert summary["event_credit_targets"] == 10
    assert summary["trainable_factual_events"] == 10


def test_prepare_distillation_stratifies_factual_success(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    traces = []
    for index in range(20):
        trace_id = f"trace-{index}"
        task_id = f"task-{index}"
        event = Event("e2", trace_id, task_id, 0, "revise", "sql_writer_a", "writer-a", "SELECT 1")
        traces.append(Trace(trace_id, task_id, "spider", "run", [event], "SELECT 1", verifier_score=float(index < 16), success=index < 16))
    (bundle / "traces.jsonl").write_text("".join(json.dumps(trace.to_dict()) + "\n" for trace in traces))
    (bundle / "credit_labels.jsonl").write_text("")

    prepare_distillation(bundle, tmp_path / "prepared", seed=17, train_count=12, validation_count=4, test_count=4)

    split = json.loads((tmp_path / "prepared" / "split.json").read_text())
    failing = {f"task-{index}" for index in range(16, 20)}
    assert {name: len(set(tasks) & failing) for name, tasks in split.items()} == {"train": 2, "validation": 1, "test": 1}
