import json

from experiments.merge_spider_dag_training_bundles import merge_training_bundles


def test_merge_training_bundles_deduplicates_labels_and_traces(tmp_path):
    def bundle(name, trace_id, labels):
        path = tmp_path / name
        path.mkdir()
        (path / "traces.jsonl").write_text(json.dumps({"trace_id": trace_id, "task_id": trace_id}) + "\n")
        (path / "credit_labels.jsonl").write_text("".join(json.dumps(label) + "\n" for label in labels))
        return path

    first = bundle("first", "t1", [{"trace_id": "t1", "event_id": "e2", "operator_name": "op", "metadata": {}}])
    second = bundle("second", "t2", [{"trace_id": "t1", "event_id": "e2", "operator_name": "op", "metadata": {}}, {"trace_id": "t2", "event_id": "e6", "operator_name": "stop", "metadata": {}}])

    summary = merge_training_bundles([first, second], tmp_path / "merged")

    assert summary == {"source_bundles": 2, "traces": 2, "credit_labels": 2, "duplicate_labels_dropped": 1}
