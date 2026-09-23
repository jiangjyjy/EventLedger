import json

from experiments.build_spider_dag_training_bundle import build_training_bundle


def test_training_bundle_merges_non_abstained_labels_and_preserves_schemes(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"trace_id": "t1", "task_id": "case1"}) + "\n")
    paired = tmp_path / "paired.jsonl"
    paired.write_text(json.dumps({"trace_id": "t1", "event_id": "e2", "operator_name": "join_condition_delete", "abstained": False, "delta_mean": 0.5, "metadata": {"operator_set": "spider_dag_paired_structural_cf_v1"}}) + "\n")
    selector = tmp_path / "selector.jsonl"
    selector.write_text(
        json.dumps({"trace_id": "t1", "event_id": "e6", "operator_name": "force_abstain", "abstained": False, "delta_mean": -1.0, "metadata": {"operator_set": "spider_dag_selector_v2"}}) + "\n"
        + json.dumps({"trace_id": "t1", "event_id": "e6", "operator_name": "hide_candidate_b", "abstained": True, "delta_mean": 0.0, "metadata": {}}) + "\n"
    )

    summary = build_training_bundle(traces, paired, selector, tmp_path / "bundle")

    rows = [json.loads(line) for line in (tmp_path / "bundle" / "credit_labels.jsonl").read_text().splitlines()]
    assert summary == {"traces": 1, "credit_labels": 2, "abstained_excluded": 1}
    assert [row["metadata"]["credit_scheme"] for row in rows] == ["paired_structural_shapley", "selector_direct_delta"]
