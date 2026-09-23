import json

from formal_ablation import VARIANT_NAMES, aggregate_label_scores, validate_formal_contract
from experiments.compile_formal_ablation_table import collect
from experiments.materialize_formal_ablation_subset import materialize_subset


def test_formal_registry_contains_all_table_rows():
    assert len(VARIANT_NAMES) == 10
    assert "no_leave_one_out" in VARIANT_NAMES
    assert "no_oracle_calibration" in VARIANT_NAMES
    assert "no_conformal_abstention" in VARIANT_NAMES
    assert "no_ranking_loss" in VARIANT_NAMES


def test_no_leave_one_out_uses_raw_delta_while_full_uses_effective_credit():
    rows = [
        {
            "trace_id": "t1",
            "event_id": "e1",
            "operator_family": "msg",
            "operator_name": "typed",
            "delta_mean": 0.25,
            "abstained": False,
            "metadata": {"rescaled_delta": 1.0, "conservation_scale": 4.0},
        }
    ]
    assert aggregate_label_scores(rows, variant="full_carve") == {"t1::e1": 1.0}
    assert aggregate_label_scores(rows, variant="no_leave_one_out") == {"t1::e1": 0.25}


def test_formal_manifest_requires_three_regimes_of_one_hundred():
    manifest = {"regimes": {regime: {"task_ids": list(range(100))} for regime in ("code_math", "sql", "openqa")}}
    validate_formal_contract(manifest)


def test_random_budget_selection_is_stable_for_a_fixed_seed():
    rows = [
        {"trace_id": "t1", "event_id": f"e{i}", "operator_family": "msg", "operator_name": "typed", "delta_mean": 1.0, "abstained": False, "metadata": {}}
        for i in range(5)
    ]
    first = aggregate_label_scores(rows, variant="random_budgeted_selection", seed=11, budget_per_trace=2)
    assert first == aggregate_label_scores(rows, variant="random_budgeted_selection", seed=11, budget_per_trace=2)
    assert len(first) == 2


def test_compiler_uses_completed_api_no_crn_evaluation(tmp_path):
    result_path = tmp_path / "api_recollection" / "code_math_no_crn" / "no_crn_code_math_evaluation.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "regime": "code_math",
                "variant": "no_crn_pairing",
                "status": "measured_api_counterfactual",
                "tasks": 100,
                "success_rate": 0.92,
                "verifier_score": 0.92,
                "provenance": {"api_counterfactual_jobs": 500},
            }
        ),
        encoding="utf-8",
    )

    rows = collect(tmp_path)["rows"]
    row = next(item for item in rows if item["regime"] == "code_math" and item["variant"] == "no_crn_pairing")

    assert row["status"] == "measured_api_counterfactual"
    assert row["metrics"]["success_rate"] == 0.92
    assert row["metrics"]["verifier_score"] == 0.92


def test_compiler_uses_completed_api_message_only_evaluation(tmp_path):
    result_path = tmp_path / "api_recollection" / "code_math_message_only" / "no_typed_code_math_evaluation.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "regime": "code_math",
                "variant": "no_typed_operators",
                "status": "measured_api_counterfactual",
                "tasks": 100,
                "success_rate": 0.92,
                "verifier_score": 0.92,
                "provenance": {"api_counterfactual_jobs": 100},
            }
        ),
        encoding="utf-8",
    )

    rows = collect(tmp_path)["rows"]
    row = next(item for item in rows if item["regime"] == "code_math" and item["variant"] == "no_typed_operators")

    assert row["status"] == "measured_api_counterfactual"
    assert row["metrics"]["events_scored"] == 100


def test_compiler_uses_completed_domain_specific_spider_and_openqa_rows(tmp_path):
    for regime, variant in (("sql", "no_typed_operators"), ("sql", "no_crn_pairing"), ("openqa", "no_typed_operators"), ("openqa", "no_crn_pairing")):
        path = tmp_path / "api_recollection" / f"{regime}_{variant}" / f"{variant}_{regime}_evaluation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"regime": regime, "variant": variant, "status": "measured_api_counterfactual", "tasks": 100, "success_rate": 0.5, "verifier_score": 0.5, "provenance": {"api_counterfactual_jobs": 200}}), encoding="utf-8")

    rows = collect(tmp_path)["rows"]
    for regime, variant in (("sql", "no_typed_operators"), ("sql", "no_crn_pairing"), ("openqa", "no_typed_operators"), ("openqa", "no_crn_pairing")):
        row = next(item for item in rows if item["regime"] == regime and item["variant"] == variant)
        assert row["status"] == "measured_api_counterfactual"
        assert row["metrics"]["verifier_score"] == 0.5


def test_compiler_uses_openqa_oracle_policy_rows_and_marks_other_domains_not_applicable(tmp_path):
    root = tmp_path / "oracle_policy"
    root.mkdir()
    for variant in ("no_oracle_calibration", "no_conformal_abstention"):
        path = root / f"{variant}_openqa_evaluation.json"
        path.write_text(
            json.dumps(
                {
                    "regime": "openqa",
                    "variant": variant,
                    "status": "measured_oracle_policy",
                    "subset_tasks": 100,
                    "tasks": 15,
                    "oracle_agreement": 0.7333333333,
                    "verifier_score": 0.7333333333,
                    "provenance": {"judge_scores": 100},
                }
            ),
            encoding="utf-8",
        )

    rows = collect(tmp_path)["rows"]
    for variant in ("no_oracle_calibration", "no_conformal_abstention"):
        openqa = next(row for row in rows if row["regime"] == "openqa" and row["variant"] == variant)
        assert openqa["status"] == "measured_oracle_policy"
        assert openqa["metrics"]["success_rate"] == 0.7333333333
        for regime in ("code_math", "sql"):
            row = next(row for row in rows if row["regime"] == regime and row["variant"] == variant)
            assert row["status"] == "not_applicable"


def test_materialize_subset_keeps_only_selected_traces_labels_and_fixed_split(tmp_path):
    traces_path = tmp_path / "source_traces.jsonl"
    labels_path = tmp_path / "source_labels.jsonl"
    traces = [
        {"trace_id": "trace-a", "task_id": "task-a"},
        {"trace_id": "trace-b", "task_id": "task-b"},
        {"trace_id": "trace-c", "task_id": "task-c"},
    ]
    labels = [
        {"trace_id": "trace-a", "event_id": "e1"},
        {"trace_id": "trace-b", "event_id": "e1"},
        {"trace_id": "trace-c", "event_id": "e1"},
    ]
    traces_path.write_text("".join(json.dumps(row) + "\n" for row in traces), encoding="utf-8")
    labels_path.write_text("".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8")

    result = materialize_subset(
        task_ids=["task-b", "task-a"],
        traces_path=traces_path,
        labels_path=labels_path,
        output_dir=tmp_path / "subset",
        seed=7,
        train_count=1,
        validation_count=1,
        test_count=0,
    )

    selected = [json.loads(line) for line in (tmp_path / "subset" / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    selected_labels = [json.loads(line) for line in (tmp_path / "subset" / "credit_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    split = json.loads((tmp_path / "subset" / "split.json").read_text(encoding="utf-8"))
    assert [row["task_id"] for row in selected] == ["task-b", "task-a"]
    assert {row["trace_id"] for row in selected_labels} == {"trace-a", "trace-b"}
    assert set(split["train"]) | set(split["validation"]) | set(split["test"]) == {"task-a", "task-b"}
    assert result["trace_count"] == 2
    assert result["label_count"] == 2
