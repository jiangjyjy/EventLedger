import json

from experiments.prepare_nq_openqa_training_bundle import build_bundle


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_bundle_merges_unique_traces_and_labels(tmp_path):
    first, second, output = tmp_path / "first", tmp_path / "second", tmp_path / "bundle"
    _write(first / "traces.jsonl", [{"trace_id": "t1", "task_id": "a"}])
    _write(second / "traces.jsonl", [{"trace_id": "t2", "task_id": "b"}])
    _write(first / "credit_labels.jsonl", [{"trace_id": "t1", "event_id": "e2", "operator_name": "op"}])
    _write(second / "credit_labels.jsonl", [{"trace_id": "t2", "event_id": "e3", "operator_name": "op"}])

    summary = build_bundle([first, second], output)

    assert summary == {"traces": 2, "labels": 2}
    assert len((output / "traces.jsonl").read_text().splitlines()) == 2
    assert len((output / "credit_labels.jsonl").read_text().splitlines()) == 2


def test_build_bundle_accepts_label_only_source_when_trace_is_already_present(tmp_path):
    traces, labels, output = tmp_path / "traces", tmp_path / "labels", tmp_path / "bundle"
    _write(traces / "traces.jsonl", [{"trace_id": "t1", "task_id": "a"}])
    _write(labels / "credit_labels.jsonl", [{"trace_id": "t1", "event_id": "e7", "operator_name": "op"}])

    summary = build_bundle([traces, labels], output)

    assert summary == {"traces": 1, "labels": 1}
