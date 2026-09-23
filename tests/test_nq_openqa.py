from carve.agents.nq_openqa_runner import NQOpenQARunner, _answer_shape_guidance
from carve.agents.api_client import APIClientConfig
from carve.datasets.nq_openqa import NQOpenQACase
from carve.verifiers.openqa import OpenQAExactMatchVerifier
from experiments.run_nq_openqa_traces import run_cases
from experiments.run_nq_openqa_traces import configure_openqa_api
from experiments.run_nq_openqa_counterfactuals import run_reader_jobs, run_router_jobs, run_selector_jobs, select_reader_jobs, select_router_jobs, select_selector_jobs
from experiments.openqa_dag_rl import ACTION_USE_A, ACTION_USE_B, evaluate_openqa_action
from experiments.evaluate_nq_openqa_student_control import evaluate_openqa_branch_selection
from experiments.evaluate_nq_openqa_student_control import evaluate_openqa_selective_branch_selection


class ScriptedClient:
    def complete(self, role, prompt, seed):
        return {"reader_a": "Linda Davis", "reader_b": "Reba McEntire", "selector": "candidate_a"}[role]


class RecordingClient(ScriptedClient):
    def __init__(self):
        self.prompts = {}

    def complete(self, role, prompt, seed):
        self.prompts[role] = prompt
        if role == "reader_a":
            return "Answer: Linda Davis\nEvidence: 99\nConfidence: 0.9"
        if role == "reader_b":
            return "Answer: Linda Davis\nEvidence: 1\nConfidence: 0.9"
        return "candidate_a"


class AbstainingSelectorClient:
    def complete(self, role, prompt, seed):
        if role == "reader_a":
            return "Answer: Linda Davis\nEvidence: 1\nConfidence: 0.9"
        if role == "reader_b":
            return "Answer: Unknown\nEvidence: none\nConfidence: 0.1"
        return "abstain"


def test_openqa_api_budget_overrides_the_generic_default_and_rejects_tiny_limits():
    config = APIClientConfig(api_key="test", max_tokens=2048)

    configured = configure_openqa_api(config, max_tokens=512)

    assert configured.max_tokens == 512
    assert configured.retries_per_url == 2
    assert config.max_tokens == 2048
    assert config.retries_per_url == 1
    import pytest
    with pytest.raises(ValueError, match="at least 64"):
        configure_openqa_api(config, max_tokens=2)


def test_openqa_verifier_normalizes_aliases_and_articles():
    result = OpenQAExactMatchVerifier().verify("The Linda Davis.", ["Linda Davis"])

    assert result.success is True
    assert result.score == 1.0


def test_nq_openqa_runner_retrieves_then_reverifies_selected_reader_answer():
    case = NQOpenQACase(
        task_id="nq_open_dev_0000",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )

    trace = NQOpenQARunner(ScriptedClient()).run(case, seed=7)

    assert trace.success is True
    assert trace.final_answer == "Linda Davis"
    assert trace.manifest["telemetry"]["api_calls"] == 3
    assert [event.event_id for event in trace.events] == [f"e{index}" for index in range(1, 10)]
    assert trace.get_event("e5").agent_role == "evidence_check_a"
    assert trace.get_event("e9").metadata["verifier_success"] is True


def test_nq_openqa_router_gives_readers_distinct_evidence_after_a_shared_top_passage():
    case = NQOpenQACase(
        task_id="nq_open_dev_0001",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=tuple((f"Title {index}", f"Evidence {index}") for index in range(1, 6)),
        evidence_ids=(),
    )
    trace = NQOpenQARunner(ScriptedClient()).run(case, seed=7)

    reader_a = trace.get_event("e3").metadata["context_indices"]
    reader_b = trace.get_event("e4").metadata["context_indices"]
    assert reader_a[0] == reader_b[0] == 1
    assert set(reader_a) != set(reader_b)


def test_nq_openqa_full_context_mode_gives_both_readers_all_retrieved_evidence():
    case = NQOpenQACase(
        task_id="nq_open_dev_full_context",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=tuple((f"Title {index}", f"Evidence {index}") for index in range(1, 6)),
        evidence_ids=(),
    )

    trace = NQOpenQARunner(ScriptedClient(), reader_context_mode="full_top_k").run(case, seed=7)

    assert trace.get_event("e3").metadata["context_indices"] == [1, 2, 3, 4, 5]
    assert trace.get_event("e4").metadata["context_indices"] == [1, 2, 3, 4, 5]


def test_nq_openqa_invalid_reader_citation_is_not_silently_replaced_with_top_evidence():
    case = NQOpenQACase(
        task_id="nq_open_dev_0002",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)

    assert trace.get_event("e3").metadata["citation"] is None
    assert trace.get_event("e5").metadata["citation_exists"] is False


def test_nq_openqa_readers_receive_distinct_reasoning_roles():
    case = NQOpenQACase(
        task_id="nq_open_dev_0003",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )
    client = RecordingClient()
    NQOpenQARunner(client).run(case, seed=7)

    assert "precision evidence extractor" in client.prompts["reader_a"]
    assert "independent verifier" in client.prompts["reader_b"]


def test_nq_openqa_answer_shape_guidance_preserves_requested_qualifiers():
    assert "full date" in _answer_shape_guidance("When was it released?")
    assert "edition/version qualifiers" in _answer_shape_guidance("Which book edition was it?")


def test_nq_openqa_selector_falls_back_to_the_only_citation_supported_candidate():
    case = NQOpenQACase(
        task_id="nq_open_dev_selector_fallback",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )

    trace = NQOpenQARunner(AbstainingSelectorClient()).run(case, seed=7)

    assert trace.final_answer == "Linda Davis"
    assert trace.get_event("e8").metadata["choice"] == "candidate_a"


def test_nq_openqa_factual_runner_fsyncs_rows_and_resumes_completed_cases(tmp_path):
    case = NQOpenQACase(
        task_id="nq_open_dev_0004",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )
    output = tmp_path / "traces.jsonl"

    first = run_cases([case], output, RecordingClient(), seed=7, model="test")
    second = run_cases([case], output, RecordingClient(), seed=7, model="test", resume=True)

    assert first == {"input_rows": 1, "trace_rows": 1, "api_calls": 3, "failed_rows": 0}
    assert second == {"input_rows": 1, "trace_rows": 0, "api_calls": 0, "failed_rows": 0}
    assert len(output.read_text().splitlines()) == 1


def test_nq_openqa_selector_jobs_cover_stop_opposite_and_hidden_unselected_branch():
    case = NQOpenQACase(
        task_id="nq_open_dev_0005",
        question="Who sang the duet?",
        answers=("Linda Davis",),
        contexts=(("Linda Davis", "Linda Davis sang the duet."),),
        evidence_ids=("1",),
    )
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)

    jobs = select_selector_jobs(trace)

    assert [(job.event_id, job.operator_name) for job in jobs] == [
        ("e7", "force_abstain"),
        ("e7", "force_candidate_b"),
        ("e7", "hide_candidate_b"),
    ]
    assert jobs[-1].metadata["requires_api"] is True


def test_nq_openqa_selector_jobs_write_resumable_verifier_credit_labels(tmp_path):
    case = NQOpenQACase("nq_open_dev_0006", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)
    output = tmp_path / "labels"

    first = run_selector_jobs([trace], {case.task_id: case}, RecordingClient(), output, seed=7)
    second = run_selector_jobs([trace], {case.task_id: case}, RecordingClient(), output, seed=7, resume=True)

    assert len(first) == 3
    assert second == []
    rows = [__import__("json").loads(line) for line in (output / "credit_labels.jsonl").read_text().splitlines()]
    assert {row["operator_name"] for row in rows} == {"force_abstain", "force_candidate_b", "hide_candidate_b"}
    assert all(row["score_source"] == "verifier" for row in rows)


def test_nq_openqa_selector_resume_retries_only_abstained_labels(tmp_path):
    case = NQOpenQACase("retry", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)
    class Failing(RecordingClient):
        def complete(self, role, prompt, seed):
            if role == "selector": raise TimeoutError("temporary")
            return super().complete(role, prompt, seed)
    output = tmp_path / "labels"
    run_selector_jobs([trace], {case.task_id: case}, Failing(), output, seed=7)
    retried = run_selector_jobs([trace], {case.task_id: case}, RecordingClient(), output, seed=7, resume=True, retry_abstained=True)
    assert len(retried) == 1


def test_nq_openqa_reader_structural_jobs_replay_selector_and_persist_labels(tmp_path):
    case = NQOpenQACase("nq_open_dev_0007", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)

    jobs = select_reader_jobs(trace)
    rows = run_reader_jobs([trace], {case.task_id: case}, RecordingClient(), tmp_path / "labels", seed=7)

    assert [(job.event_id, job.operator_name) for job in jobs] == [("e3", "citation_remove"), ("e3", "answer_truncate"), ("e4", "citation_remove"), ("e4", "answer_truncate")]
    assert len(rows) == 4
    assert all(row["operator_family"] == "reader" for row in rows)


def test_nq_openqa_router_jobs_target_each_reader_branch():
    case = NQOpenQACase("nq_open_dev_0008", "Who sang?", ("Linda Davis",), tuple((str(i), "Linda Davis sang the duet.") for i in range(1, 5)), ())
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)

    jobs = select_router_jobs(trace)

    assert [(job.event_id, job.operator_name) for job in jobs] == [("e2", "drop_passage_a"), ("e2", "drop_passage_b"), ("e2", "swap_branch_evidence")]
    assert all(job.metadata["requires_api"] for job in jobs)


def test_nq_openqa_router_jobs_replay_downstream_and_persist_labels(tmp_path):
    case = NQOpenQACase("nq_open_dev_0009", "Who sang?", ("Linda Davis",), tuple((str(i), "Linda Davis sang the duet.") for i in range(1, 5)), ())
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)
    rows = run_router_jobs([trace], {case.task_id: case}, RecordingClient(), tmp_path / "labels", seed=7)

    assert len(rows) == 3
    assert all(row["operator_family"] == "router" for row in rows)
    assert all(row["metadata"]["replay_evidence"]["reexecuted_roles"] for row in rows)
    evidence = rows[0]["metadata"]["replay_evidence"]
    assert evidence["api_telemetry"]["api_calls"] == 2
    assert set(evidence["mutated_readers"]) == {"a"}


def test_openqa_rl_actions_use_saved_candidates_and_em_verifier():
    case = NQOpenQACase("rl", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)
    a, b = evaluate_openqa_action(trace, case, ACTION_USE_A), evaluate_openqa_action(trace, case, ACTION_USE_B)
    assert a.success is True and a.saved_api_calls == 2
    assert b.success is True and b.saved_api_calls == 2


def test_openqa_student_control_selects_higher_scored_reader_and_reverifies_it():
    case = NQOpenQACase("control", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)

    record = evaluate_openqa_branch_selection(
        trace,
        case,
        {f"{trace.trace_id}::e3": 1.0, f"{trace.trace_id}::e4": -1.0},
    )

    assert record["action"] == "use_a"
    assert record["success"] is True
    assert record["saved_api_calls"] == 2


def test_openqa_selective_control_falls_back_to_factual_below_the_margin_threshold():
    case = NQOpenQACase("selective", "Who sang?", ("Linda Davis",), (("Linda", "Linda Davis sang the duet."),), ("1",))
    trace = NQOpenQARunner(RecordingClient()).run(case, seed=7)
    scores = {f"{trace.trace_id}::e3": 1.0, f"{trace.trace_id}::e4": -1.0}

    record = evaluate_openqa_selective_branch_selection(trace, case, scores, threshold=3.0)

    assert record["action"] == "use_factual_selector"
    assert record["shortcut"] is False
    assert record["saved_api_calls"] == 0
